from pathlib import Path
import json
import numpy as np
import pandas as pd
from .core import read,save
from .evaluation import summarize

def build(c,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df=summarize(folder)
    dest=folder/"figures"
    dest.mkdir(exist_ok=True)
    def finish(name,fig):
        fig.tight_layout()
        fig.savefig(dest/(name+".png"),dpi=150,bbox_inches="tight")
        plt.close(fig)
    mean=df.groupby("model")
    for col,title in [("f1","Binary F1"),("parameters","Trainable parameters"),("training_s","Training time (s)"),("head_ms","Prediction latency (ms)")]:
        fig,ax=plt.subplots(figsize=(7.5,5))
        mean[col].mean().dropna().sort_values().plot.barh(ax=ax,color="#1c7064",xerr=mean[col].std().fillna(0) if len(df.fold.unique())>1 else None)
        ax.set(xlabel=title,ylabel="",title=f"{title} by model")
        finish(col,fig)
    audit=read(c.processed/"audit.json")
    for name in ("class_counts","speakers"):
        fig,ax=plt.subplots(figsize=(8,4))
        pd.Series(audit[name]).sort_values(ascending=False).head(16).plot.bar(ax=ax,color="#26786b")
        ax.set(ylabel="Utterances",title="Class distribution (1 = sarcastic)" if name=="class_counts" else "Most frequent speakers")
        finish(name,fig)
    entries=[read(p) for p in sorted(folder.glob("fold-0/*/evaluation.json"))]
    primary=[e for e in entries if e["model"] in ("text","speech","concat","proposed")]
    if primary:
        fig,axs=plt.subplots(1,len(primary),figsize=(3*len(primary),3),squeeze=False)
        for ax,e in zip(axs[0],primary):
            cm=e["metrics"]["confusion"]
            ax.imshow(cm,cmap="Greens")
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i][j]),ha="center",va="center",bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
            ax.set(title=e["model"],xlabel="Predicted",ylabel="Actual",xticks=[0,1],yticks=[0,1])
        finish("confusion-fold-1",fig)
        for name,ylabel,xlabel in [("roc","True-positive rate","False-positive rate"),("pr","Precision","Recall")]:
            fig,ax=plt.subplots()
            for e in primary:
                ax.plot(e[name]["x"],e[name]["y"],label=e["model"])
            ax.legend()
            ax.set(xlabel=xlabel,ylabel=ylabel,title="Fold 1")
            finish(name,fig)
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        history=next((e for e in primary if e["model"]=="proposed"),primary[0])["history"]
        h=pd.DataFrame(history).set_index("epoch")
        for ax,col in zip(axs,["loss","f1"]):
            h[[f"train_{col}",f"val_{col}"]].plot(ax=ax)
            ax.set_ylabel(col)
        finish("learning-curves",fig)
    p=pd.read_csv(folder/"held_out_predictions.csv")
    comparisons=p.pivot(index=["fold","id"],columns="model",values="prediction")
    lines=["# Held-out error analysis","", "Fold means weight each fold equally. Example selection below follows sorted fold and sample ID, never score magnitude.",""]
    if "proposed" in comparisons:
        base=p[p.model=="proposed"].set_index(["fold","id"]).sort_index()
        for outcome in ("TP","TN","FP","FN"):
            sub=base[base.outcome==outcome]
            lines.extend([f"## {outcome} ({len(sub)})",""])
            if len(sub):
                ix=sub.index[0]
                r=sub.iloc[0]
                lines.extend([f"{ix[0]}, {ix[1]} · {r['text']}","",f"Reference {r['label']}, score {r['score']:.3f}. {r['subset']}.",""])
        if "text" in comparisons:
            for title,mask in [("Corrected with multimodal input",(comparisons["text"]!=base.label)&(comparisons.proposed==base.label)),
                               ("Multimodal error after a correct text decision",(comparisons["text"]==base.label)&(comparisons.proposed!=base.label))]:
                selected=base.loc[mask]
                lines.extend([f"## {title} ({len(selected)})",""])
                for ix,r in selected.head(3).iterrows():
                    lines.append(f"- {ix[1]} · {r['text']} (label {r.label}, proposed score {r.score:.3f})")
                lines.append("")
        lines+=["These comparisons change the entire classifier. They cannot isolate the causal effect of the voice on any individual decision. Laughter, other speakers, ambiguous words and acting are plausible confounders to examine during playback."]
    (folder/"error-analysis.md").write_text("\n".join(lines),encoding="utf-8")
