import torch
from torch import nn
import torch.nn.functional as F
from .audio import GROUPS,NAMES
from .features import DIMS

PRIMARY=["text","speech","concat","proposed"]
ABLATIONS=["gate_only","incongruity_only","no_sentiment","no_emotion",*["without_"+k.lower() for k in GROUPS]]
DISAGREE=["negative gap","neutral gap","positive gap","negative overlap","neutral overlap","positive overlap","cosine distance","praise × negative voice","criticism × positive voice"]
def model_inputs(kind, features):
    branches={"speech":("audio",),"text":("text",),"concat":("text","audio"),"gate_only":("text","audio")}
    return {k:features[k] for k in branches.get(kind,tuple(DIMS))}

def conflict(s,e):
    v=torch.stack((e[:,2]+e[:,3],e[:,0],e[:,1]),dim=1)
    return torch.cat(((s-v).abs(),s*v,(1-F.cosine_similarity(s,v)).unsqueeze(1),
                      s[:,2:3]*v[:,0:1],s[:,0:1]*v[:,2:3]),1)

class Net(nn.Module):
    def __init__(self,kind,width=48,dropout=.25,aux_classes=0):
        super().__init__()
        if kind not in PRIMARY+ABLATIONS+["aux_emotion"]:
            raise ValueError(kind)
        self.kind=kind
        def projection(size):
            return nn.Sequential(nn.Linear(size,width),nn.GELU(),nn.Dropout(dropout))
        if kind!="speech":
            self.text=projection(DIMS["text"])
        if kind!="text":
            self.audio=projection(DIMS["audio"])
        self.gated=kind not in ("text","speech","concat","incongruity_only")
        self.incongruity=kind not in ("text","speech","concat","gate_only")
        if self.gated:
            self.gate=nn.Linear(2*width,width)
        fused=2*width if kind in ("concat","incongruity_only") else width
        self.head=nn.Sequential(nn.Linear(fused+9*self.incongruity,width//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(width//2,1))
        if aux_classes:
            self.aux=nn.Linear(width,aux_classes)
        mask=torch.ones(len(NAMES))
        if kind.startswith("without_"):
            group=next(k for k in GROUPS if kind=="without_"+k.lower())
            mask[[NAMES.index(n) for n in GROUPS[group]]]=0
        self.register_buffer("acoustic_mask",mask)

    def forward(self,x,omit_conflict=False):
        info={}
        if self.kind!="speech":
            t=self.text(x["text"])
        if self.kind!="text":
            a=self.audio(x["audio"]*self.acoustic_mask)
        if self.kind=="speech":
            z=a
        elif self.kind=="text":
            z=t
        elif self.gated:
            g=self.gate(torch.cat((t,a),1)).sigmoid()
            z=g*t+(1-g)*a
            info["gate"]=g
        else:
            z=torch.cat((t,a),1)
        if self.incongruity:
            s,e=x["sentiment"],x["emotion"]
            if self.kind=="no_sentiment":
                s=torch.zeros_like(s)
            if self.kind=="no_emotion":
                e=torch.zeros_like(e)
            d=conflict(s,e)
            info["conflict"]=d
            z=torch.cat((z,torch.zeros_like(d) if omit_conflict else d),1)
        if hasattr(self,"aux"):
            info["aux"]=self.aux(a)
        return self.head(z).flatten(),info
