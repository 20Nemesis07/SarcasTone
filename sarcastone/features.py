import gc
import time
import zipfile
import numpy as np
from tqdm import tqdm
from filelock import FileLock
from . import audio
from .core import atomic, read, save, stable

DIMS={"text":768,"audio":len(audio.NAMES),"sentiment":3,"emotion":4}
LABELS={"sentiment":["negative","neutral","positive"],"emotion":["neu","hap","ang","sad"]}

def fingerprint(c, branch):
    meta=dict(version=c.features["version"],branch=branch,dim=DIMS[branch])
    if branch!="audio":
        meta["model"]=c.locks[branch]["revision"]
    if branch in ("text","sentiment"):
        meta["tokens"]=c.features["tokens"]
    if branch=="text":
        meta["context"]=c.features["context"]
    if branch in ("emotion","audio"):
        meta["rate"]=c.features["rate"]
    return meta

def input_text(c, branch, row):
    if branch=="text" and c.features["context"]:
        return " [SEP] ".join([row["text"],*row.get("context",[])[-2:]])
    return row["text"]

def cache_path(c,branch,row):
    data=input_text(c,branch,row) if branch in ("text","sentiment") else row["audio_hash"]
    return c.cache/branch/stable(fingerprint(c,branch))[:16]/f"{row['id']}-{stable(data)[:20]}.npz"

def load_feature(path, branch):
    with np.load(path,allow_pickle=False) as f:
        x=f["x"]
    if x.dtype!=np.float32 or x.shape!=(DIMS[branch],) or not np.isfinite(x).all():
        raise ValueError(f"Invalid vector in {path}")
    if branch in LABELS and (np.min(x)<0 or not np.isclose(x.sum(),1,atol=1e-4)):
        raise ValueError(f"Invalid probabilities in {path}")
    return x

class Encoder:
    def __init__(self,c,branch):
        import torch
        from transformers import (AutoTokenizer,AutoModel,AutoModelForSequenceClassification,
                                  AutoFeatureExtractor,AutoModelForAudioClassification)
        self.branch,self.tokens=branch,c.features["tokens"]
        self.device=c.train["device"]
        if self.device=="auto":
            self.device="cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(c.train["threads"])
        path=c.model(branch)
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {branch} weights. Run setup: {path}")
        kwargs=dict(local_files_only=True,trust_remote_code=False)
        if branch=="emotion":
            self.processor=AutoFeatureExtractor.from_pretrained(path,**kwargs)
            model=AutoModelForAudioClassification
        else:
            self.processor=AutoTokenizer.from_pretrained(path,**kwargs)
            model=AutoModel if branch=="text" else AutoModelForSequenceClassification
        self.model=model.from_pretrained(path,**kwargs).eval().requires_grad_(False).to(self.device)
        if branch in LABELS:
            ids={str(v).lower():int(k) for k,v in self.model.config.id2label.items()}
            self.order=[ids[label] for label in LABELS[branch]]

    def __call__(self,value):
        import torch
        with torch.inference_mode():
            if self.branch=="emotion":
                batch=self.processor(value,sampling_rate=audio.RATE,return_tensors="pt")
            else:
                batch=self.processor(value,truncation=True,max_length=self.tokens,return_tensors="pt")
            batch=batch.to(self.device)
            out=self.model(**batch)
            if self.branch=="text":
                mask=batch["attention_mask"].unsqueeze(-1)
                vec=(out.last_hidden_state*mask).sum(1)/mask.sum(1).clamp(min=1)
            else:
                vec=out.logits.softmax(-1)[:,self.order]
            return vec[0].cpu().numpy().copy()

def extract(c,rows,force=False,branches=None):
    c.cache.mkdir(parents=True,exist_ok=True)
    with FileLock(str(c.cache/"extraction.lock"),timeout=2):
        return _extract(c,rows,force,branches)

def _extract(c,rows,force,branches):
    timings={}
    for branch in branches or DIMS:
        misses=[]
        for row in rows:
            try:
                if force:
                    raise ValueError()
                load_feature(cache_path(c,branch,row),branch)
            except (FileNotFoundError,ValueError,OSError,EOFError,zipfile.BadZipFile,KeyError):
                misses.append(row)
        if not misses:
            continue
        start=time.perf_counter()
        encoder=None if branch=="audio" else Encoder(c,branch)
        for r in tqdm(misses,desc=f"{branch} ({len(rows)-len(misses)} reused)"):
            if branch in ("audio","emotion"):
                value=audio.read_pcm(c.processed/r["audio"],c.features["max_seconds"])
            else:
                value=input_text(c,branch,r)
            x=audio.inspect(value)[0] if branch=="audio" else encoder(value)
            path=cache_path(c,branch,r)
            with atomic(path) as tmp:
                np.savez_compressed(tmp,x=np.asarray(x,dtype=np.float32))
                load_feature(tmp,branch)
        save(path.parent/"metadata.json",fingerprint(c,branch))
        del encoder
        gc.collect()
        timings[branch]=dict(samples=len(misses),seconds=time.perf_counter()-start)
    if timings:
        save(c.cache/f"extraction-{time.time_ns()}.json",timings)
    return timings
