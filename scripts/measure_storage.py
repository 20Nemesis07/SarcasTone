import json
import platform
import shutil
import sys
from pathlib import Path
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sarcastone.core import Config,save

def main():
    c=Config()
    folders={"raw_metadata":c.raw,"archives":c.archive.parent,"processed":c.processed,
             "pretrained_models":c.models,"feature_cache":c.cache,"experiments":c.runs,"environment":c.root/"environment"}
    sizes={k:sum(p.stat().st_size for p in v.rglob("*") if p.is_file()) for k,v in folders.items()}
    report=dict(bytes=sizes,total_bytes=sum(sizes.values()),free_bytes=shutil.disk_usage(c.root).free,
                ram_bytes=psutil.virtual_memory().total,cpu=platform.processor(),logical_cpus=psutil.cpu_count(),
                physical_cpus=psutil.cpu_count(logical=False),python=platform.python_version())
    save(c.root/"reports/storage.json",report)
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    main()
