from pathlib import Path
import json,sys
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from sarcastone.core import Config,read,save
from sarcastone.data import prepare,norm
from sarcastone.features import DIMS,extract
from sarcastone.experiments import KINDS
from unittest.mock import patch
c=Config()
def mtimes():
    return {f.name:f.stat().st_mtime_ns for f in (c.processed/"audio").glob("*.wav")}
before=mtimes()
rows=prepare(c)
assert mtimes()==before
with patch("sarcastone.features.Encoder",side_effect=AssertionError("unexpected recomputation")):
    assert extract(c,rows)=={}
lookup={r["id"]:r for r in rows}
count=0
for name in ("benchmark","friends"):
    folder=c.runs/name
    for i,s in enumerate(read(folder/"splits.json")):
        splits=[s[k] for k in ("train","validation","test")]
        for j,x in enumerate(splits):
            for z in splits[j+1:]:
                assert not set(x).intersection(z)
                for field in ("text","audio_hash",*(("speaker",) if name=="friends" else ())):
                    def keys(ids):
                        return {norm(lookup[k][field]) for k in ids}
                    assert not keys(x).intersection(keys(z)),(name,i,field)
        for kind in KINDS:
            ev=read(folder/f"fold-{i}"/kind/"evaluation.json")
            assert [r["id"] for r in ev["predictions"]]==s["test"],kind
            assert sum(map(sum,ev["metrics"]["confusion"]))==len(s["test"])
        count+=1
save(c.root/"reports/recheck.json",dict(wav_reused=len(before),folds_checked=count,vectors_reused=len(rows)*len(DIMS)))
print("Audited partitions, model test IDs, zero recomputation and unchanged PCM",count,len(before))
