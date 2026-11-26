"""Exercise actual local resources using one published fold-1 test recording."""
import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sarcastone.core import Config,read,save
from sarcastone import audio
from sarcastone.features import DIMS,extract
from sarcastone.inference import LiveModels,analyze,transcribe
from sarcastone.resources import setup,verify

def main():
    import psutil
    c=Config()
    folder=c.runs/"benchmark"
    held=read(folder/"splits.json")[0]["test"]
    rows=read(c.processed/"manifest.json")
    row=next(r for r in rows if r["id"]==held[0])
    import socket
    with patch.object(socket.socket,"connect",side_effect=AssertionError("Offline test attempted a network connection")):
        verify(c)
        setup(c)
        assert not extract(c,rows),"Unexpected feature cache misses"
        start=time.perf_counter()
        y=audio.decode((c.processed/row["audio"]).read_bytes())
        t=time.perf_counter()
        words=transcribe(c,y)
        asr=time.perf_counter()-t
        live=analyze(c,folder/"fold-0",words,y)
        elapsed=time.perf_counter()-start
        gold=analyze(c,folder/"fold-0",row["text"],y)
        for kind,p in gold["predictions"].items():
            expected=next(r["score"] for r in read(folder/"fold-0"/kind/"evaluation.json")["predictions"] if r["id"]==row["id"])
            if abs(p-expected)>1e-5:
                raise AssertionError(f"Offline training/inference mismatch for {kind}: {p} versus {expected}")
        bank=LiveModels()
        pair=[]
        for uid in ("2_549","2_209"):
            item=next(r for r in rows if r["id"]==uid)
            recording=audio.read_pcm(c.processed/item["audio"],c.features["max_seconds"])
            paired=analyze(c,folder/"fold-0","Really?",recording,bank=bank)
            pair.append(dict(id=uid,label=item["label"],predictions=paired["predictions"],analysis_s=paired["seconds"]))
        assert abs(pair[0]["predictions"]["text"]-pair[1]["predictions"]["text"])<1e-7
        del bank
        seconds={}
        for branch in ("text","sentiment","emotion"):
            from sarcastone.features import Encoder
            t=time.perf_counter()
            model=Encoder(c,branch)
            seconds[branch]=time.perf_counter()-t
            del model
    mem=psutil.Process().memory_info()
    report=dict(id=row["id"],reference=row["text"],transcription=words,duration=row["duration"],
                asr_seconds=asr,analysis_seconds=live["seconds"],total_seconds=elapsed,
                manual_scores=gold["predictions"],asr_scores=live["predictions"],label=row["label"],
                loader_seconds=seconds,process_peak_bytes=getattr(mem,"peak_wset",None),
                zero_cache_misses=True,network_blocked=True,all_four_match_offline_test=True,paired_uploads=pair)
    save(folder/"offline-smoke.json",report)
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    main()
