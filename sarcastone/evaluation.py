import time
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,
                             confusion_matrix,roc_curve,precision_recall_curve)
from .core import atomic,read,save

def metrics(y,p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    cls=(p>=.5).astype(int)
    return dict(n=len(y),accuracy=float(accuracy_score(y,cls)),precision=float(precision_score(y,cls,zero_division=0)),
                recall=float(recall_score(y,cls,zero_division=0)),f1=float(f1_score(y,cls,zero_division=0)),
                macro_f1=float(f1_score(y,cls,average="macro",labels=[0,1],zero_division=0)),
                weighted_f1=float(f1_score(y,cls,average="weighted",zero_division=0)),
                auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
                confusion=confusion_matrix(y,cls,labels=[0,1]).tolist())

def evaluate(kind,rows,labels,scores,meta):
    y=np.asarray(labels)
    scores=np.asarray(scores)
    fpr,tpr,_=roc_curve(y,scores)
    precision,recall,_=precision_recall_curve(y,scores)
    outputs=[]
    for r,t,p in zip(rows,y,scores):
        cls=int(p>=.5)
        outputs.append(dict(id=r["id"],text=r["text"],speaker=r["speaker"],show=r["show"],label=int(t),
                            score=float(p),prediction=cls,outcome=("TP" if t else "FP") if cls else ("FN" if t else "TN"),
                            subset=r.get("subset","unavailable")))
    sub={}
    for group in sorted({r["subset"] for r in outputs}):
        subset=[r for r in outputs if r["subset"]==group]
        sub[group]=metrics([r["label"] for r in subset],[r["score"] for r in subset])
    return dict(model=kind,metrics=metrics(y,scores),roc=dict(x=fpr.tolist(),y=tpr.tolist()),
                pr=dict(x=recall.tolist(),y=precision.tolist()),subgroups=sub,predictions=outputs,**meta)

def csv_save(path,df):
    with atomic(path) as tmp:
        df.to_csv(tmp,index=False)

def summarize(folder):
    entries=[(p,read(p)) for p in sorted(folder.glob("fold-*/*/evaluation.json"))]
    if not entries:
        raise ValueError("No evaluated folds in this run.")
    flat=[]
    predictions=[]
    for p,e in entries:
        fold=p.parents[1].name
        flat.append(dict(model=e["model"],fold=fold,**{k:v for k,v in e["metrics"].items() if k!="confusion"},
                         parameters=e["parameters"],training_s=e["training_s"],head_ms=e["head_ms"]))
        predictions.extend(dict(model=e["model"],fold=fold,**r) for r in e["predictions"])
    df=pd.DataFrame(flat)
    csv_save(folder/"fold_metrics.csv",df)
    csv_save(folder/"held_out_predictions.csv",pd.DataFrame(predictions))
    long=df.drop(columns=["fold"]).groupby("model").agg(["mean","std","count"])
    long.columns=["_".join(c) for c in long.columns]
    csv_save(folder/"summary.csv",long.reset_index())
    save(folder/"summary.json",__import__("json").loads(long.reset_index().to_json(orient="records")))
    return df
