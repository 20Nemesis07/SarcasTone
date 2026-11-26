import argparse
from .core import Config,read
def main():
    p=argparse.ArgumentParser(description="SarcasTone offline research workbench")
    p.add_argument("stage",choices=["setup","verify","prepare","extract","train","evaluate","pipeline","resume","external","report"])
    p.add_argument("--config")
    p.add_argument("--archive",help="Import the publisher's original ZIP")
    p.add_argument("--metadata-only",action="store_true")
    p.add_argument("--run",default="benchmark")
    p.add_argument("--protocol",choices=["official","friends","speaker"])
    p.add_argument("--models",nargs="+",help="Train selected heads only")
    p.add_argument("--force-recompute",action="store_true",help="Rebuild derived PCM/features")
    args=p.parse_args()
    c=Config(args.config)
    from .resources import setup,verify
    if args.stage=="setup":
        return setup(c,args.archive,args.metadata_only)
    if args.stage=="verify":
        verify(c)
        print("All pinned resources verified locally.")
        return
    from .core import key
    folder=c.runs/key(args.run)
    if args.stage=="external":
        from .external import evaluate_private
        return evaluate_private(c,folder)
    if args.stage=="report":
        from .report import build
        return build(c,folder)
    if args.stage=="evaluate":
        from .evaluation import summarize
        return summarize(folder)
    pipeline=args.stage in ("pipeline","resume")
    if pipeline:
        verify(c)
    if pipeline or args.stage=="prepare":
        from .data import prepare
        rows=prepare(c,args.force_recompute)
    else:
        rows=read(c.processed/"manifest.json")
    if pipeline or args.stage=="extract":
        from .features import extract
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=c.train["threads"]):
            extract(c,rows,args.force_recompute)
    if pipeline or args.stage=="train":
        from .experiments import run
        run(c,rows,args.run,args.models,args.protocol)
    if pipeline:
        from .report import build
        build(c,folder)
if __name__=="__main__":
    main()
