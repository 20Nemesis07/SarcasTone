"""Time the first and two successive four-model takes on the same reference clip."""
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sarcastone.core import Config,read,save
from sarcastone import audio
from sarcastone.inference import LiveModels,transcribe,analyze

def main():
    c=Config()
    run=c.runs/"benchmark"
    uid=read(run/"splits.json")[0]["test"][0]
    row=next(r for r in read(c.processed/"manifest.json") if r["id"]==uid)
    raw=(c.processed/row["audio"]).read_bytes()
    bank=LiveModels()
    timings=[]
    for i in range(3):
        tick=time.perf_counter()
        y=audio.decode(raw)
        words=transcribe(c,y,bank)
        result=analyze(c,run/"fold-0",words,y,bank=bank)
        timings.append(dict(take=i+1,total_seconds=time.perf_counter()-tick,score=result["predictions"]["proposed"],cached_models=len(bank.loaded)))
    save(run/"latency.json",dict(sample=uid,duration=row["duration"],runs=timings))
    print(timings)
if __name__=="__main__":
    main()
