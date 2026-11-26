"""Keep official test IDs. Purge duplicates from the development side of each boundary."""
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from .core import PROJECT,read
from .data import norm

def groups(rows,context=False,speakers=False):
    parents=list(range(len(rows)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]
            i=parents[i]
        return i
    seen={}
    for i,r in enumerate(rows):
        tokens=[("line",norm(r["text"]))]
        if r.get("audio_hash"):
            tokens.append(("audio",r["audio_hash"]))
        if context:
            tokens += [("line",norm(s)) for s in r["context"] if norm(s)]
        if speakers:
            tokens.append(("speaker",norm(r["speaker"])))
        for tok in tokens:
            if tok in seen:
                parents[root(i)]=root(seen[tok])
            else:
                seen[tok]=i
    return np.array([root(i) for i in range(len(rows))])

def partitions(rows,protocol,seed,context=False):
    y=np.array([r["label"] for r in rows])
    idx=np.arange(len(rows))
    g=groups(rows,context)
    if protocol=="official":
        folds=[idx[[r["id"] in ids for r in rows]] for ids in map(set,read(PROJECT/"configs/official_folds.json"))]
    elif protocol=="friends":
        folds=[idx[[r["show"]=="FRIENDS" for r in rows]]]
    elif protocol=="speaker":
        g=groups(rows,context,True)
        folds=[te for _,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(idx,y,g)]
    else:
        raise ValueError("Unknown evaluation protocol")
    result=[]
    for test in folds:
        dev=idx[~np.isin(g,g[test])]
        if protocol=="friends":
            seen={norm(rows[i]["speaker"]) for i in test}
            dev=np.array([i for i in dev if norm(rows[i]["speaker"]) not in seen],dtype=int)
        # Use speaker-disjoint validation when measuring new-speaker generalization.
        inner=groups([rows[i] for i in dev],context,protocol in ("friends","speaker"))
        for tr,va in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(dev,y[dev],inner):
            if len(set(y[dev[tr]]))==len(set(y[dev[va]]))==2:
                break
        else:
            raise ValueError("Unable to construct two-class group-safe validation.")
        tr,va=dev[tr],dev[va]
        if len(set(y[test]))!=2:
            raise ValueError("Test fold lacks a class.")
        result.append({k:[rows[i]["id"] for i in ids] for k,ids in
                       dict(train=tr,validation=va,test=test,purged=np.setdiff1d(idx,np.r_[tr,va,test])).items()})
    return result
