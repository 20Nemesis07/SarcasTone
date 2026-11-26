import copy
import random
import shutil
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from .core import atomic
from .model import Net,model_inputs

def normalize(x,stats):
    return {k: ((v-stats[k][0])/stats[k][1]).clamp(-8,8) if k in stats else v for k,v in x.items()}

def scale_stats(x,tr):
    return {k:(x[k][tr].mean(0),x[k][tr].std(0,unbiased=False).clamp_min(1e-5)) for k in ("text","audio") if k in x}

def save_pt(path,obj):
    with atomic(path) as tmp:
        torch.save(obj,tmp)
        torch.load(tmp,map_location="cpu",weights_only=True)

def restore(folder,signature):
    for name in ("latest.pt","previous.pt","best.pt"):
        path=folder/name
        if path.is_file():
            try:
                d=torch.load(path,map_location="cpu",weights_only=True)
            except (OSError,RuntimeError,EOFError,ValueError,IndexError,__import__("pickle").UnpicklingError):
                continue
            if d["signature"]!=signature:
                raise ValueError("Checkpoint configuration differs. Select a fresh run name.")
            return d
    if any(folder.glob("*.pt")):
        raise ValueError("All checkpoint generations are unreadable. Select a fresh run.")
    return None

def train(kind,x,y,aux,tr,va,folder,config,signature,interrupt_after=None):
    opts=config.train
    seed=opts["seed"]
    torch.set_num_threads(opts["threads"])
    device=opts["device"]
    if device=="auto":
        device="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    inputs=model_inputs(kind,x)
    stats=scale_stats(inputs,tr)
    z={k:v.to(device) for k,v in normalize(inputs,stats).items()}
    y=y.to(device)
    aux=aux.to(device)
    args=dict(kind=kind,width=opts["width"],dropout=opts["dropout"],aux_classes=int(aux[tr].max().item()+1) if kind=="aux_emotion" else 0)
    model=Net(**args).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=opts["learning_rate"],weight_decay=opts["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",patience=3,factor=.5)
    loss_fn=nn.BCEWithLogitsLoss()
    folder.mkdir(parents=True,exist_ok=True)
    def measure(ids):
        with torch.inference_mode():
            logits,_=model({k:v[ids] for k,v in z.items()})
            f=f1_score(y[ids].cpu(),(logits>=0).cpu(),zero_division=0)
            return loss_fn(logits,y[ids]).item(),float(f)
    d=restore(folder,signature)
    start,stale,best_score,history,seconds,best=0,0,-1,[],0.,None
    if d:
        model.load_state_dict(d["weights"])
        optimizer.load_state_dict(d["optimizer"])
        scheduler.load_state_dict(d["scheduler"])
        start,stale,best_score,history,seconds,best=(d[k] for k in ("epoch","stale","best_score","history","seconds","best_weights"))
        torch.set_rng_state(d["rng"])
        if device=="cuda" and d["cuda_rng"]:
            torch.cuda.set_rng_state_all(d["cuda_rng"])
        random.setstate(d["python_rng"])
    for epoch in range(start,opts["epochs"]):
        if stale>=opts["patience"]:
            break
        tick=time.perf_counter()
        model.train()
        for ids in torch.as_tensor(tr)[torch.randperm(len(tr))].split(opts["batch"]):
            optimizer.zero_grad(set_to_none=True)
            logits,info=model({k:v[ids] for k,v in z.items()})
            loss=loss_fn(logits,y[ids])
            if "aux" in info:
                valid=aux[ids]>=0
                if valid.any():
                    loss=loss+.2*nn.functional.cross_entropy(info["aux"][valid],aux[ids][valid])
            loss.backward()
            optimizer.step()
        model.eval()
        tl,tf=measure(tr)
        vl,vf=measure(va)
        scheduler.step(vf)
        history.append(dict(epoch=epoch+1,train_loss=tl,val_loss=vl,train_f1=tf,val_f1=vf))
        improved=vf>best_score
        if improved:
            best_score,stale,best=vf,0,copy.deepcopy(model.state_dict())
        else:
            stale+=1
        seconds+=time.perf_counter()-tick
        d=dict(signature=signature,args=args,epoch=epoch+1,stale=stale,best_score=best_score,best_weights=best,
               weights=model.state_dict(),history=history,seconds=seconds,stats=stats,optimizer=optimizer.state_dict(),
               scheduler=scheduler.state_dict(),rng=torch.get_rng_state(),python_rng=random.getstate(),
               cuda_rng=torch.cuda.get_rng_state_all() if device=="cuda" else [])
        latest=folder/"latest.pt"
        if latest.exists():
            with atomic(folder/"previous.pt") as tmp:
                shutil.copyfile(latest,tmp)
        save_pt(latest,d)
        if improved:
            save_pt(folder/"best.pt",d)
        if epoch+1==interrupt_after:
            raise InterruptedError("Injected unit-test interruption")
    # best_weights is embedded in every generation, including a commit interrupted before best.pt.
    model.load_state_dict(best)
    model.cpu().eval()
    save_pt(folder/"inference.pt",dict(args=args,weights=model.state_dict(),stats=stats,signature=signature))
    return model,stats,history,seconds
