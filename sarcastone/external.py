import time
import uuid
from pathlib import Path
from scipy.io import wavfile
from . import audio
from .core import atomic,read,save
from .evaluation import metrics

def collect(c,y,text,speaker,label,condition,consent,pair="",accent="",literal="unspecified"):
    if not consent or not text.strip() or not speaker.strip() or label not in (0,1):
        raise ValueError("Supply consent, a transcript, a speaker alias and the intended delivery.")
    uid=uuid.uuid4().hex
    path=c.root/"private"/"recordings"/uid
    with atomic(path.with_suffix(".wav")) as tmp:
        wavfile.write(tmp,audio.RATE,audio.validate(y,c.features["max_seconds"]))
    save(path.with_suffix(".json"),dict(id=uid,text=text,speaker=speaker,label=label,condition=condition,
                                       pair=pair,accent=accent,literal=literal,consent=True))
    return uid

def evaluate_private(c,folder):
    from .inference import analyze
    fold=folder/"fold-0"
    rows=[]
    for path in sorted((c.root/"private/recordings").glob("*.json")):
        r=read(path)
        if not r["consent"]:
            continue
        y=audio.read_pcm(path.with_suffix(".wav"),c.features["max_seconds"])
        result=analyze(c,fold,r["text"],y)
        rows.append(dict(**r,scores=result["predictions"]))
    if not rows:
        raise ValueError("No consented recordings found. Use the Live analysis collection form.")
    summary={}
    for k in rows[0]["scores"]:
        summary[k]=metrics([r["label"] for r in rows],[r["scores"][k] for r in rows])
    dest=c.root/"private/evaluations"/f"{folder.name}-{time.time_ns()}.json"
    save(dest,dict(run=folder.name,fold=0,transcript="participant-supplied",metrics=summary,rows=rows))
    print(dest)
