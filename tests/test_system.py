"""Only synthetic fixtures in this file. No outputs here are benchmark results."""
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scipy.io import wavfile
from sarcastone import audio
from sarcastone.core import Config,atomic,digest
from sarcastone.resources import fetch
from sarcastone.features import DIMS,extract,cache_path,load_feature
from sarcastone.model import Net,PRIMARY,ABLATIONS,conflict
from sarcastone.training import train
from sarcastone.evaluation import metrics
from sarcastone.splits import groups
from sarcastone.external import collect

class SystemTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch.dict(os.environ,SARCASTONE_HOME=self.temp.name):
            self.c=Config()
        self.c.train.update(epochs=4,threads=1,patience=6,batch=6,width=8)
        self.y=(.2*np.sin(2*np.pi*180*np.arange(16000)/16000)).astype("float32")
        self.c.processed.mkdir(parents=True)
        self.wav=self.c.processed/"sample.wav"
        wavfile.write(self.wav,16000,self.y)

    def test_audio_and_features(self):
        decoded=audio.decode(self.wav.read_bytes())
        x,trace=audio.inspect(decoded)
        self.assertEqual(x.shape,(44,))
        self.assertTrue(np.isfinite(x).all())
        self.assertAlmostEqual(np.nanmedian(trace["pitch"]),180,delta=4)
        self.assertAlmostEqual(x[audio.NAMES.index("rms_mean")],.2/np.sqrt(2),places=3)
        np.testing.assert_allclose(audio.inspect(audio.read_pcm(self.wav,45))[0],x,rtol=1e-4,atol=1e-4)

    def test_silence_corruption_and_duration(self):
        for v in [np.zeros(16000),np.array([np.nan]),np.ones(16001)]:
            with self.assertRaises(ValueError):
                audio.validate(v,1)
        with self.assertRaises(ValueError):
            audio.decode(b"invalid audio")

    def test_cache_resume_corruption_and_skip(self):
        rows=[dict(id=str(i),audio="sample.wav",audio_hash=digest(self.wav)) for i in range(3)]
        extract(self.c,rows[:1],branches=["audio"])
        first=cache_path(self.c,"audio",rows[0])
        mtime=first.stat().st_mtime_ns
        self.assertEqual(extract(self.c,rows,branches=["audio"])["audio"]["samples"],2)
        self.assertEqual(first.stat().st_mtime_ns,mtime)
        self.assertFalse(extract(self.c,rows,branches=["audio"]))
        first.write_bytes(b"interrupted")
        self.assertEqual(extract(self.c,rows,branches=["audio"])["audio"]["samples"],1)
        self.assertEqual(load_feature(first,"audio").shape,(44,))

    def test_setup_idempotence_and_atomic_rollback(self):
        path=Path(self.temp.name)/"copy.wav"
        h=digest(self.wav)
        self.assertTrue(fetch(path,"never-used",self.wav.stat().st_size,h,self.wav))
        with patch("requests.get",side_effect=AssertionError("network used")):
            self.assertFalse(fetch(path,"offline",self.wav.stat().st_size,h))
        with self.assertRaises(RuntimeError):
            with atomic(path) as tmp:
                tmp.write_bytes(b"incomplete")
                raise RuntimeError()
        self.assertEqual(digest(path),h)

    def test_mono_inputs_and_ablation_shapes(self):
        x={k:torch.randn(3,n) for k,n in DIMS.items()}
        for kind in PRIMARY+ABLATIONS:
            model=Net(kind,width=8)
            inputs={"audio":x["audio"]} if kind=="speech" else ({"text":x["text"]} if kind=="text" else x)
            logits,info=model(inputs)
            self.assertEqual(tuple(logits.shape),(3,))
            torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.,0.])).backward()
            self.assertEqual("gate" in info,model.gated)
            self.assertEqual("conflict" in info,model.incongruity)
        d=conflict(torch.tensor([[0.,0.,1.]]),torch.tensor([[0.,0.,1.,0.]]))
        self.assertAlmostEqual(d[0,-2].item(),1)
        self.assertAlmostEqual(d[0,6].item(),1)

    def test_training_resume_reproduces_weights(self):
        torch.manual_seed(1)
        x={k:torch.randn(30,n) for k,n in DIMS.items()}
        labels=torch.arange(30).remainder(2).float()
        args=("proposed",x,labels,torch.full((30,),-1),list(range(20)),list(range(20,30)))
        p=Path(self.temp.name)
        with self.assertRaises(InterruptedError):
            train(*args,p/"resume",self.c,"unit",interrupt_after=2)
        first=train(*args,p/"resume",self.c,"unit")
        other=train(*args,p/"full",self.c,"unit")
        for a,b in zip(first[0].parameters(),other[0].parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(first[2],other[2])
        self.assertTrue((p/"resume/inference.pt").exists())

    def test_joined_leakage_groups(self):
        rows=[dict(text="Great!",speaker="A",context=["Goodbye."]),dict(text="Great",speaker="B",context=[]),
              dict(text="Goodbye",speaker="C",context=[]),dict(text="Different words",speaker="D",context=[])]
        g=groups(rows,True)
        self.assertEqual(g[0],g[1])
        self.assertEqual(g[0],g[2])
        self.assertNotEqual(g[0],g[3])

    def test_metrics_and_consent(self):
        m=metrics([0,0,1,1],[.1,.8,.9,.7])
        self.assertEqual(m["confusion"],[[1,1],[0,2]])
        self.assertAlmostEqual(m["f1"],.8)
        with self.assertRaises(ValueError):
            collect(self.c,self.y,"Test","p1",0,"quiet",False)
        self.assertFalse((self.c.root/"private").exists())

if __name__=="__main__":
    unittest.main()
