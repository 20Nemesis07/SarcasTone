import contextlib
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def digest(path, git=False):
    path = Path(path)
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

@contextlib.contextmanager
def atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        with tmp.open("rb+") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def save(path, obj):
    with atomic(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def key(value):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", value):
        raise ValueError("Use letters, digits, hyphens and underscores for identifiers.")
    return value

class Config:
    def __init__(self, file=None):
        data = yaml.safe_load(Path(file or os.environ.get("SARCASTONE_CONFIG", PROJECT/"configs/default.yaml")).read_text())
        self.root = Path(os.environ.get("SARCASTONE_HOME", data["storage"])).expanduser().resolve()
        self.train, self.features, self.protocol = data["training"], data["features"], data["protocol"]
        if self.features["rate"] != 16000:
            raise ValueError("Frozen audio models require 16 kHz.")
        self.raw = self.root/"datasets/raw"
        self.archive = self.root/"datasets/archives/mmsd_raw_data.zip"
        self.processed = self.root/"datasets/processed"
        self.models = self.root/"pretrained_models"
        self.cache = self.root/"feature_cache"
        self.runs = self.root/"results"
        self.locks = read(PROJECT/"configs/models.lock.json")
        for var, folder in {"HF_HOME":"environment/hf","TORCH_HOME":"environment/torch","MPLCONFIGDIR":"environment/mpl"}.items():
            os.environ[var] = str(self.root/folder)
        os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",TOKENIZERS_PARALLELISM="false")
    def model(self, branch):
        return self.models/branch/self.locks[branch]["revision"]
