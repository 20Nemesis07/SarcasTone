import time
import threading
from pathlib import Path
import numpy as np
import torch
from . import audio
from .core import read
from .features import Encoder,DIMS,fingerprint,input_text
from .model import Net,PRIMARY,DISAGREE,model_inputs
from .training import normalize

def load_asr(c):
    from faster_whisper import WhisperModel
    if not (c.model("asr")/"model.bin").is_file():
        raise FileNotFoundError("Local ASR weights unavailable. Enter the transcript manually.")
    return WhisperModel(str(c.model("asr")),device="cpu",compute_type="int8",cpu_threads=c.train["threads"],local_files_only=True)

class LiveModels:
    """Share immutable local weights between takes, subject to available memory."""
    def __init__(self):
        self.loaded={}
        self.lock=threading.Lock()

    def get(self,c,branch):
        import psutil
        key=(str(c.model(branch)),c.train["device"],c.train["threads"],c.features["tokens"])
        reserve=2_500_000_000
        with self.lock:
            if psutil.virtual_memory().available<reserve:
                self.loaded.clear()
            if key in self.loaded:
                return self.loaded[key]
            model=load_asr(c) if branch=="asr" else Encoder(c,branch)
            if psutil.virtual_memory().available>=reserve:
                self.loaded[key]=model
            return model

def transcribe(c,y,bank=None):
    asr=bank.get(c,"asr") if bank else load_asr(c)
    segments,_=asr.transcribe(y,language="en",beam_size=1,condition_on_previous_text=False,vad_filter=False)
    return " ".join(s.text.strip() for s in segments).strip()

def explain(model,z):
    with torch.inference_mode():
        base=float(model(z)[0].sigmoid().item())
        changes={}
        for name in ("text","audio",*audio.GROUPS,"incongruity"):
            mod=dict(z)
            if name in ("text","audio"):
                mod[name]=torch.zeros_like(z[name])
            if name in audio.GROUPS:
                v=z["audio"].clone()
                v[:,[audio.NAMES.index(n) for n in audio.GROUPS[name]]]=0
                mod["audio"]=v
            changes[name]=base-float(model(mod,omit_conflict=name=="incongruity")[0].sigmoid().item())
        return changes

def analyze(c,fold,text="",y=None,context=None,bank=None):
    started=time.perf_counter()
    torch.set_num_threads(c.train["threads"])
    meta=read(fold.parent/"run.json")
    if meta["features"]!={k:fingerprint(c,k) for k in DIMS}:
        raise ValueError("This run used a different feature configuration. Select the matching config.")
    if c.features["context"] and not context:
        raise ValueError("This checkpoint requires dialogue context. Select a target-only experiment for live use.")
    available=[k for k in PRIMARY if (fold/k/"inference.pt").is_file() and
               ((k=="text" and text.strip()) or (k=="speech" and y is not None) or (text.strip() and y is not None))]
    if not available:
        raise ValueError("No compatible trained checkpoint for these inputs. Train a run or change the input.")
    out=dict(text=text,predictions={},gate=None,conflict={},explanation={},seconds=0)
    X={}
    def encode(branch,value):
        model=bank.get(c,branch) if bank else Encoder(c,branch)
        return model(value)
    if y is not None:
        x,traces=audio.inspect(y)
        X["audio"]=x
        out.update(traces=traces,acoustics=dict(zip(audio.NAMES,x.tolist())),wave=y)
    if text.strip():
        X["text"]=encode("text",input_text(c,"text",dict(text=text,context=context or [])))
        X["sentiment"]=encode("sentiment",text)
        out["sentiment"]=X["sentiment"]
    if y is not None:
        X["emotion"]=encode("emotion",y)
        out["emotion"]=X["emotion"]
    with torch.inference_mode():
        for kind in available:
            ck=torch.load(fold/kind/"inference.pt",map_location="cpu",weights_only=True)
            model=Net(**ck["args"]).eval()
            model.load_state_dict(ck["weights"])
            z=normalize({k:torch.from_numpy(v[None]) for k,v in X.items()},ck["stats"])
            logits,info=model(model_inputs(kind,z))
            out["predictions"][kind]=float(logits.sigmoid().item())
            if kind=="proposed":
                out["gate"]=info["gate"][0].tolist()
                out["conflict"]=dict(zip(DISAGREE,info["conflict"][0].tolist()))
                out["explanation"]=explain(model,z)
    out["seconds"]=time.perf_counter()-started
    return out
