import collections
import csv
import re
import zipfile
import tempfile
from pathlib import Path, PurePosixPath
from scipy.io import wavfile
from tqdm import tqdm
from filelock import FileLock
from . import audio
from .core import PROJECT, atomic, digest, key, read, save, stable

def norm(s):
    return " ".join(re.findall(r"\w+",s.casefold()))

def metadata(c):
    rows=read(c.raw/"sarcasm_data.json")
    extra={}
    with (c.raw/"mustardpp.csv").open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if r["Sarcasm"].strip() in ("0","1"):
                extra.setdefault(r["SCENE"].strip(),[]).append(r)
    result=[]
    for uid,r in rows.items():
        key(uid)
        if type(r["sarcasm"]) is not bool or not all(isinstance(r[k],str) and r[k].strip() for k in ("utterance","speaker","show")):
            raise ValueError(f"Invalid record {uid}")
        if len(r["context"])!=len(r["context_speakers"]):
            raise ValueError(f"Context alignment error {uid}")
        matches=[x for x in extra.get(uid,[]) if norm(x["SENTENCE"])==norm(r["utterance"]) and norm(x["SPEAKER"])==norm(r["speaker"]) and norm(x["SHOW"])==norm(r["show"]) and int(x["Sarcasm"])==r["sarcasm"]]
        e=matches[0] if len(matches)==1 else {}
        result.append(dict(id=uid,text=r["utterance"],label=int(r["sarcasm"]),speaker=r["speaker"].strip(),show=r["show"],
                           context=r["context"],context_speakers=r["context_speakers"],sentiment=None,
                           emotion=e.get("Explicit_Emotion"),implicit_emotion=e.get("Implicit_Emotion"),valence=e.get("Valence")))
    return result

def prepare(c,force=False):
    c.processed.mkdir(parents=True,exist_ok=True)
    with FileLock(str(c.processed/"preparation.lock"),timeout=2):
        return _prepare(c,force)

def _prepare(c,force):
    rows=metadata(c)
    audit=[]
    wanted={r["id"] for r in rows}
    mp=c.processed/"manifest.json"
    past={r["id"]:r for r in read(mp)} if mp.exists() else {}
    progress=c.processed/"preparation-progress.json"
    if progress.exists():
        past.update({r["id"]:r for r in read(progress)})
    with zipfile.ZipFile(c.archive) as z:
        clips={}
        for member in z.infolist():
            p=PurePosixPath(member.filename.replace("\\","/"))
            if p.is_absolute() or ".." in p.parts or "__MACOSX" in p.parts:
                continue
            if p.parent.name=="utterances_final" and p.suffix.lower()==".mp4" and p.stem in wanted:
                if p.stem in clips:
                    raise ValueError(f"Ambiguous media ID {p.stem}")
                clips[p.stem]=member
        accepted=[]
        for r in tqdm(rows,desc="Pair target WAVs"):
            uid=r["id"]
            wav=c.processed/"audio"/(uid+".wav")
            media=clips.get(uid)
            try:
                if media is None or media.file_size>128*1024**2:
                    raise ValueError("Target clip missing or exceeds size limit")
                old=past.get(uid,{})
                if not force and old.get("crc")==media.CRC and wav.exists() and digest(wav)==old.get("audio_hash"):
                    seconds,sha=old["duration"],old["audio_hash"]
                else:
                    # MP4 containers may require seeking to a trailing index.
                    c.processed.mkdir(parents=True,exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=c.processed) as temp:
                        clip=Path(temp)/"target.mp4"
                        clip.write_bytes(z.read(media))
                        y=audio.decode(clip,c.features["max_seconds"])
                    with atomic(wav) as tmp:
                        wavfile.write(tmp,audio.RATE,y)
                    seconds,sha=len(y)/audio.RATE,digest(wav)
                if seconds>c.features["max_seconds"]:
                    raise ValueError("Recording exceeds configured length")
                accepted.append(dict(r,audio=f"audio/{uid}.wav",duration=seconds,audio_hash=sha,crc=media.CRC,member=media.filename))
            except (ValueError,OSError,zipfile.BadZipFile) as exc:
                audit.append(dict(id=uid,error=str(exc)))
            if len(accepted)%25==0:
                past.update({v["id"]:v for v in accepted[-25:]})
                save(progress,list(past.values()))
    if not accepted:
        raise ValueError("No valid audio pairs. Check archive layout and validation errors: "+str(audit[:3]))
    save(mp,accepted)
    counts=collections.Counter(norm(r["text"]) for r in accepted)
    summary=dict(source_count=len(rows),valid=len(accepted),class_counts=dict(collections.Counter(r["label"] for r in accepted)),
                 speakers=dict(collections.Counter(r["speaker"] for r in accepted)),excluded=audit,
                 duplicate_transcripts={k:v for k,v in counts.items() if v>1},emotion_matches=sum(bool(r["emotion"]) for r in accepted))
    save(c.processed/"audit.json",summary)
    return accepted
