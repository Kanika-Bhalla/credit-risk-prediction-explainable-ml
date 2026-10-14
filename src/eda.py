"""Generate exploratory plots for public UCI German Credit data."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

FEATURE_LABELS = {"Attribute1":"Checking account status","Attribute2":"Loan duration (months)","Attribute3":"Credit history","Attribute4":"Loan purpose","Attribute5":"Credit amount","Attribute6":"Savings account","Attribute7":"Employment duration","Attribute8":"Installment rate","Attribute9":"Personal status / sex","Attribute10":"Other debtors","Attribute11":"Residence duration","Attribute12":"Property","Attribute13":"Age","Attribute14":"Other installment plans","Attribute15":"Housing","Attribute16":"Existing credits","Attribute17":"Job category","Attribute18":"Dependents","Attribute19":"Telephone","Attribute20":"Foreign worker"}

def run(output_dir="results/eda"):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    ds=fetch_ucirepo(id=144)
    X=ds.data.features.copy()
    y=(ds.data.targets.squeeze().astype(int)==2).astype(int)
    fig,ax=plt.subplots(figsize=(6,4))
    counts=y.value_counts().reindex([0,1],fill_value=0)
    bars=ax.bar(["Good credit risk","Bad credit risk"],counts.values,color=["#3987be","#db8055"])
    ax.bar_label(bars)
    ax.set_ylabel("Number of records");ax.set_title("Credit risk class distribution")
    fig.tight_layout();fig.savefig(out/"target_distribution.png",dpi=160);plt.close(fig)

    missing=X.isna().sum().sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(9,4))
    nonzero=missing[missing>0]
    if len(nonzero):
        nonzero.plot.bar(ax=ax,color="#3987be");ax.set_ylabel("Missing values")
        ax.set_title("Missing values by feature")
    else:
        ax.text(.5,.5,"No missing feature values",ha="center",va="center",transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout();fig.savefig(out/"missing_values.png",dpi=160);plt.close(fig)

    numeric=X.select_dtypes(include=np.number)
    cols=numeric.columns[:6]
    if len(cols):
        fig,axs=plt.subplots(2,3,figsize=(13,7))
        for ax in axs.flat: ax.set_visible(False)
        for ax,col in zip(axs.flat,cols):
            ax.set_visible(True)
            ax.hist(numeric[col].dropna(),bins=20,color="#3987be",edgecolor="white")
            ax.set_title(FEATURE_LABELS.get(col,col));ax.set_ylabel("Count")
        fig.suptitle("Numerical feature distributions")
        fig.tight_layout();fig.savefig(out/"numeric_distributions.png",dpi=160);plt.close(fig)
        c=numeric.corr()
        fig,ax=plt.subplots(figsize=(max(7,len(c)*.65),max(6,len(c)*.6)))
        im=ax.imshow(c,cmap="coolwarm",vmin=-1,vmax=1)
        ax.set_xticks(range(len(c)));ax.set_xticklabels([FEATURE_LABELS.get(x,x) for x in c.columns],rotation=60,ha="right",fontsize=8)
        ax.set_yticks(range(len(c)));ax.set_yticklabels([FEATURE_LABELS.get(x,x) for x in c.index],fontsize=8)
        fig.colorbar(im,ax=ax,label="Pearson r")
        ax.set_title("Numeric feature correlations (descriptive)")
        fig.tight_layout();fig.savefig(out/"correlation_heatmap.png",dpi=160);plt.close(fig)

    results=[]
    for col in X.columns:
        if X[col].dtype=="object" or str(X[col].dtype)=="category":
            rates=pd.DataFrame({"group":X[col].astype("string").fillna("Missing"),"bad":y.to_numpy()}).groupby("group",dropna=False).agg(n=("bad","size"),rate=("bad","mean"))
            rates=rates[rates.n>=20].sort_values("n",ascending=False).head(8)
            if len(rates)>1:
                results.append((col,rates))
    for col,rates in results[:3]:
        fig,ax=plt.subplots(figsize=(9,4))
        ax.bar(rates.index.astype(str),rates.rate,color="#db8055")
        ax.set_ylim(0,1);ax.set_ylabel("Observed bad-credit proportion")
        ax.set_title(f"Observed risk by {FEATURE_LABELS.get(col,col)} (n ≥ 20)")
        ax.tick_params(axis="x",rotation=35)
        fig.tight_layout();fig.savefig(out/f"risk_by_{col}.png",dpi=160);plt.close(fig)

    summary={"total_records":len(X),"num_features":X.shape[1],"bad_credit_rate":round(float(y.mean()),4),"missing_by_feature":missing.to_dict()}
    import json
    (out/"eda_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Saved EDA charts to",out)
if __name__=="__main__":
    run()
