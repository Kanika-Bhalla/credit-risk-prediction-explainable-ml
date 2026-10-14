"""Standalone, reproducible German Credit risk ML experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             log_loss, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def load_data():
    """Fetch UCI German Credit; original labels 1=good, 2=bad -> 0/1."""
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=144)
    X = dataset.data.features.copy()
    labels = dataset.data.targets.squeeze().astype(int)
    assert set(labels.unique()) == {1, 2}, "Unexpected target coding"
    y = (labels == 2).astype(int)
    return X, y

def make_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.columns.difference(numeric).tolist()
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
    ], sparse_threshold=0)

def choose_threshold(y_true, probabilities):
    choices = np.linspace(0.10, 0.90, 81)
    f1s = [f1_score(y_true, probabilities >= t, zero_division=0) for t in choices]
    return float(choices[int(np.argmax(f1s))])

def score_predictions(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def run(output_dir="results", shap_enabled=True):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    preprocessing = make_preprocessor(X_train)
    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"model__C": [0.1, 1, 10], "model__class_weight": [None, "balanced"]}),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {"model__n_estimators": [150],
             "model__min_samples_leaf": [2, 8],
             "model__class_weight": [None, "balanced"]})
    }
    comparisons, best = [], None
    for name, (estimator, grid) in candidates.items():
        pipeline = Pipeline([("prep", clone(preprocessing)), ("model", estimator)])
        search = GridSearchCV(pipeline, grid, scoring="average_precision", cv=CV,
                              n_jobs=1, refit=True, error_score="raise")
        search.fit(X_train, y_train)
        comparisons.append({"model": name, "cv_average_precision": float(search.best_score_),
                            "best_params": search.best_params_})
        if best is None or search.best_score_ > best[0]:
            best = (search.best_score_, name, search.best_estimator_)
    pd.DataFrame(comparisons).to_json(out / "model_comparison.json", orient="records", indent=2)
    _, winner, fitted_base = best

    # Nested calibration takes place within training folds.
    template = CalibratedClassifierCV(estimator=clone(fitted_base), method="sigmoid", cv=3)
    oof_prob = cross_val_predict(template, X_train, y_train, cv=CV,
                                 method="predict_proba", n_jobs=1)[:, 1]
    threshold = choose_threshold(y_train, oof_prob)
    calibrated = clone(template).fit(X_train, y_train)
    p_test = calibrated.predict_proba(X_test)[:, 1]
    report = {"dataset": "UCI Statlog German Credit (CC BY 4.0)",
              "rows": int(len(X)), "bad_risk_fraction": float(y.mean()),
              "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
              "selected_model": winner, "threshold": threshold,
              "validation_oof_brier": float(brier_score_loss(y_train, oof_prob)),
              "test": score_predictions(y_test, p_test, threshold),
              "warning": "Educational benchmark, not a deployment-ready lending decision system."}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    CalibrationDisplay.from_predictions(y_test, p_test, n_bins=7, ax=axes[0])
    axes[0].set_title("Test reliability (descriptive)")
    ConfusionMatrixDisplay.from_predictions(y_test, p_test >= threshold,
                                             ax=axes[1], colorbar=False)
    axes[1].set_title(f"Test confusion matrix (threshold {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(out / "evaluation.png", dpi=160)
    plt.close(fig)

    if shap_enabled:
        try:
            import shap
            prep = fitted_base.named_steps["prep"]
            classifier = fitted_base.named_steps["model"]
            bg_raw = X_train.sample(n=min(30, len(X_train)), random_state=SEED)
            sample_raw = X_test.sample(n=min(10, len(X_test)), random_state=SEED)
            background = np.asarray(prep.transform(bg_raw), dtype=float)
            rows = np.asarray(prep.transform(sample_raw), dtype=float)
            names = prep.get_feature_names_out()
            explainer = shap.Explainer(lambda a: classifier.predict_proba(a)[:, 1],
                                      background, algorithm="permutation", feature_names=names)
            explanations = explainer(rows, max_evals=max(2 * rows.shape[1] + 1, 200))
            importance = pd.DataFrame({"feature": names,
                 "mean_abs_shap": np.abs(explanations.values).mean(axis=0)})
            importance.sort_values("mean_abs_shap", ascending=False).to_csv(
                out / "shap_importance.csv", index=False)
            shap.plots.bar(explanations, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(out / "shap_bar.png", dpi=160, bbox_inches="tight")
            plt.close("all")
            report["shap_note"] = "Explains uncalibrated base classifier on 10 sampled test rows."
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            print("Optional SHAP analysis skipped:", type(exc).__name__, str(exc)[:240])

    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, shap_enabled=not args.no_shap)
