"""Spot-check publisher ID alignment using local ASR. Retain every sampled case."""
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sarcastone import audio
from sarcastone.core import Config,read,save
from sarcastone.data import norm
from sarcastone.inference import transcribe

def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+int(x!=y)))
        prev=cur
    return prev[-1]

def main():
    c=Config()
    rows=read(c.processed/"manifest.json")
    rng=random.Random(42)
    output=[]
    for show,label in sorted({(r["show"],r["label"]) for r in rows}):
        row=rng.choice([r for r in rows if (r["show"],r["label"])==(show,label)])
        y=audio.read_pcm(c.processed/row["audio"],c.features["max_seconds"])
        words=transcribe(c,y)
        ref=norm(row["text"]).split()
        wer=distance(ref,norm(words).split())/max(1,len(ref))
        output.append(dict(id=row["id"],show=show,label=label,reference=row["text"],asr=words,word_error_rate=wer))
    save(c.processed/"alignment-spotcheck.json",output)
    print("Compared",len(output),"source-aligned clips. See alignment-spotcheck.json for possible mismatches.")
if __name__=="__main__":
    main()
