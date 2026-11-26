"""Write a portable research snapshot from the measured local runs, with no weights or recordings."""
import csv
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sarcastone.core import Config,PROJECT,read

def publish(c):
    dest=PROJECT/"research"
    folder=c.runs/"benchmark"
    friends=c.runs/"friends"
    a=pd.read_csv(folder/"fold_metrics.csv")
    b=pd.read_csv(friends/"fold_metrics.csv").set_index("model")
    scores=a.groupby("model")
    proposed=scores.get_group("proposed")
    if set(proposed.fold)!={f"fold-{i}" for i in range(5)}:
        raise ValueError("All five primary folds are needed for the report.")
    lines=["# MUStARD benchmark","", "The 690 held-out decisions per model come from the published outer test folds. The development partition excludes repeated transcripts or identical audio shared with its test fold. Friends has a separate fixed test set of 356 utterances. The two protocols share examples and must be interpreted separately.","",
           "| Model | Accuracy | Precision | Recall | Binary F1 ± SD | Macro F1 | Weighted F1 | AUC | Friends F1 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k,q in scores:
        m=q.mean(numeric_only=True)
        lines.append(f"| {k} | {m.accuracy:.3f} | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} ± {q.f1.std():.3f} | {m.macro_f1:.3f} | {m.weighted_f1:.3f} | {m.auc:.3f} | {b.loc[k,'f1']:.3f} |")
    fp=pd.read_csv(folder/"held_out_predictions.csv")
    prim=fp[fp.model.isin(["text","speech","concat","proposed"])]
    from sarcastone.evaluation import metrics
    lines+=["","## Does vocal information help when estimated valences disagree?","",
            "Each row below pools out-of-fold examples for one model and subgroup. The subgroup is defined once per utterance by frozen model estimates. Missing independent modality-specific human labels make this an exploratory test.","",
            "| Subgroup | n | Positive labels | Text F1 | Speech F1 | Concat F1 | Proposed F1 |","|---|---:|---:|---:|---:|---:|---:|"]
    subgroup_rows=[]
    for group,p in prim.groupby("subset"):
        values={}
        for k,q in p.groupby("model"):
            values[k]=metrics(q.label,q.score)["f1"]
        sample=p[p.model=="proposed"]
        subgroup_rows.append(dict(subset=group,n=len(sample),positive=int(sample.label.sum()),**values))
        lines.append(f"| {group} | {len(sample)} | {sample.label.sum()} | "+" | ".join(f"{values[k]:.3f}" for k in ["text","speech","concat","proposed"])+" |")
    pd.DataFrame(subgroup_rows).to_csv(dest/"agreement-subgroups.csv",index=False)
    means=scores.f1.mean()
    lines+=["",f"The proposed head differed from concatenation by {means.proposed-means.concat:+.3f} mean binary F1 and from the text-only head by {means.proposed-means.text:+.3f}. The Friends difference versus concatenation was {b.loc['proposed','f1']-b.loc['concat','f1']:+.3f}. Fold means and pooled subgroup metrics answer different questions. These descriptive contrasts do not establish statistical significance.","","## Split audit","",
            "| Fold | Train | Validation | Test | Purged |","|---|---:|---:|---:|---:|"]
    for name,path in [("CV",folder),("Friends",friends)]:
        for i,s in enumerate(read(path/"splits.json")):
            lines.append(f"| {name} {i+1} | "+" | ".join(str(len(s[k])) for k in ("train","validation","test","purged"))+" |")
    lines+=["","## Show and label association","","| Source | Non-sarcastic | Sarcastic |","|---|---:|---:|"]
    for show,q in fp[fp.model=="proposed"].groupby("show"):
        lines.append(f"| {show} | {sum(q.label==0)} | {sum(q.label==1)} |")
    lines+=["","Label frequencies vary substantially by show. Even an audio-only model can exploit production or voice cues associated with a show. Prosodic interpretation requires reviewing this confound alongside the holdout results.","","## Saved evidence","","The accompanying results folders contain fold-level metrics, prediction-level evidence, error reports and the split manifests. Neural head latency is warm single-item timing. Classical latency is amortized over the test batch. Both exclude frozen feature extraction. Full live latency is measured separately. Volunteers have not yet contributed an external evaluation set."]
    (dest/"BENCHMARK.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    for name,path in [("five-fold",folder),("friends",friends)]:
        out=dest/"results"/name
        out.mkdir(parents=True,exist_ok=True)
        for f in ("fold_metrics.csv","summary.csv","summary.json","splits.json","error-analysis.md","run.json"):
            shutil.copy2(path/f,out/f)
        pd.read_csv(path/"held_out_predictions.csv").drop(columns="text").to_csv(out/"held_out_predictions.csv",index=False)
        shutil.copytree(path/"figures",out/"figures",dirs_exist_ok=True)
if __name__=="__main__":
    publish(Config())
