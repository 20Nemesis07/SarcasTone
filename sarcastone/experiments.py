import platform
import time
import importlib.metadata
import numpy as np
import torch
from filelock import FileLock
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .core import key,read,save,stable
from .evaluation import evaluate,summarize
from .features import DIMS,cache_path,load_feature,fingerprint
from .model import PRIMARY,ABLATIONS,model_inputs
from .splits import partitions
from .training import normalize,train

KINDS=["tfidf","svm",*PRIMARY,*ABLATIONS]
def run(c,rows,name,kinds=None,protocol=None):
    folder=c.runs/key(name)
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/"run.lock"),timeout=1):
        saved=read(folder/"run.json") if (folder/"run.json").exists() else {}
        _run(c,[dict(r) for r in rows],folder,kinds or KINDS,protocol or saved.get("protocol",c.protocol))

def _run(c,rows,folder,kinds,protocol):
    torch.set_num_threads(c.train["threads"])
    splits=partitions(rows,protocol,c.train["seed"],c.features["context"])
    meta=dict(protocol=protocol,training=c.train,features={k:fingerprint(c,k) for k in DIMS},dataset=stable(rows),
              split_hash=stable(splits),format=2,python=platform.python_version(),
              versions={p:importlib.metadata.version(p) for p in ("torch","scikit-learn","numpy","transformers")})
    signature=stable(meta)
    if (folder/"run.json").exists() and read(folder/"run.json")!=meta:
        raise ValueError("Run metadata changed. Choose a new --run name.")
    save(folder/"run.json",meta)
    save(folder/"splits.json",splits)
    ids={r["id"]:i for i,r in enumerate(rows)}
    labels=torch.tensor([r["label"] for r in rows],dtype=torch.float32)
    x={k:torch.from_numpy(np.stack([load_feature(cache_path(c,k,r),k) for r in rows])) for k in DIMS}
    for i,r in enumerate(rows):
        s=int(x["sentiment"][i].argmax())
        neu,hap,ang,sad=x["emotion"][i].tolist()
        e=int(np.argmax([ang+sad,neu,hap]))
        r["subset"]="predicted agreement" if s==e else ("predicted opposite valence" if {s,e}=={0,2} else "predicted neutral mismatch")
    for fold,split in enumerate(splits):
        tr,va,te=([ids[uid] for uid in split[s]] for s in ("train","validation","test"))
        vocab=sorted({rows[i]["emotion"] for i in tr if rows[i].get("emotion")})
        aux=torch.tensor([vocab.index(r["emotion"]) if r.get("emotion") in vocab else -1 for r in rows])
        for kind in kinds:
            path=folder/f"fold-{fold}"/key(kind)
            target=path/"evaluation.json"
            if target.exists() and (kind in ("tfidf","svm") or (path/"inference.pt").exists()):
                continue
            print(f"{folder.name} • fold {fold+1}/{len(splits)} • {kind}",flush=True)
            start=time.perf_counter()
            history=[]
            if kind in ("tfidf","svm"):
                if kind=="tfidf":
                    estimator=make_pipeline(TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True),
                                            LogisticRegression(max_iter=500,random_state=c.train["seed"]))
                    inputs=np.array([r["text"] for r in rows])
                else:
                    estimator=make_pipeline(StandardScaler(),SVC(C=1,probability=True,random_state=c.train["seed"]))
                    inputs=x["audio"].numpy()
                estimator.fit(inputs[tr],labels[tr].numpy())
                seconds=time.perf_counter()-start
                start=time.perf_counter()
                probs=estimator.predict_proba(inputs[te])[:,1]
                latency=(time.perf_counter()-start)*1000/len(te)
                nparams=None
            else:
                model,stats,history,seconds=train(kind,x,labels,aux,tr,va,path,c,stable([signature,fold,kind]))
                norm=model_inputs(kind,normalize({k:v[te] for k,v in x.items()},stats))
                with torch.inference_mode():
                    probs=model(norm)[0].sigmoid().numpy()
                    single={k:v[:1] for k,v in norm.items()}
                    model(single)
                    start=time.perf_counter()
                    for _ in range(40):
                        model(single)
                    latency=(time.perf_counter()-start)*25
                nparams=sum(p.numel() for p in model.parameters())
            result=evaluate(kind,[rows[i] for i in te],labels[te].numpy(),probs,
                            dict(history=history,training_s=seconds,head_ms=latency,parameters=nparams))
            save(target,result)
    summarize(folder)
