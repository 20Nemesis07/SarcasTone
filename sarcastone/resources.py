import os
import shutil
from pathlib import Path
import requests
from filelock import FileLock
from tqdm import tqdm
from .core import PROJECT, atomic, digest, read

def entries(c):
    for name, row in read(PROJECT/"configs/dataset.lock.json").items():
        dest = c.archive if name.endswith(".zip") else c.raw/name
        yield dest, row["url"], row["size"], row["sha256"]
    for branch, model in c.locks.items():
        for name, row in model["files"].items():
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"
            yield c.model(branch)/name, url, row["size"], row["digest"]

def valid(path, size, expected):
    return path.is_file() and path.stat().st_size == size and digest(path, len(expected)==40) == expected

def fetch(path, url, size, expected, manual=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)+".lock", timeout=30):
        if valid(path, size, expected):
            return False
        if manual:
            src = Path(manual)
            if not valid(src,size,expected):
                raise ValueError("Manual archive differs from the official size or SHA-256.")
            with atomic(path) as tmp:
                shutil.copyfile(src,tmp)
            return True
        part = path.with_suffix(path.suffix+".part")
        offset = part.stat().st_size if part.exists() else 0
        if offset == size and valid(part,size,expected):
            os.replace(part,path)
            return True
        if offset >= size:
            offset = 0
        headers = {"Accept-Encoding":"identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with requests.get(url, headers=headers, stream=True, timeout=(20,90)) as r:
            r.raise_for_status()
            append = r.status_code == 206
            if append and not r.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                raise ValueError("Server returned an inconsistent resume offset.")
            with part.open("ab" if append else "wb") as f, tqdm(total=size,initial=offset if append else 0,unit="B",unit_scale=True,desc=path.name) as bar:
                for block in r.iter_content(1024*512):
                    f.write(block)
                    bar.update(len(block))
        if not valid(part,size,expected):
            raise ValueError(f"Incomplete or mismatched download: {part}. Retry setup.")
        os.replace(part,path)
        return True

def setup(c, manual=None, metadata=False):
    for path,url,size,h in entries(c):
        if metadata and path.parent != c.raw:
            continue
        try:
            fetch(path,url,size,h,manual if path==c.archive else None)
        except requests.RequestException as exc:
            raise RuntimeError(f"Download interrupted. Rerun setup, or import the official archive using --archive. Source: {url}") from exc

def verify(c):
    missing = [str(p) for p,u,s,h in entries(c) if not valid(p,s,h)]
    if missing:
        raise ValueError("Missing or damaged resources:\n"+"\n".join(missing))
