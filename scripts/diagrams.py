from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]/"research"
def render(name,nodes,edges,size):
    fig,ax=plt.subplots(figsize=size)
    fig.patch.set_facecolor("#f6f7f5")
    ax.set(xlim=(-.012,1.012),ylim=(-.018,1.018))
    ax.axis("off")
    compact=name=="data-flow"
    w,h=.173,(.43 if compact else .19)
    for a,b in edges:
        x,y,*_=nodes[a]
        u,v,*_=nodes[b]
        ax.annotate("",(u-w/2,v),(x+w/2,y),arrowprops=dict(arrowstyle="-|>",color="#75948b",lw=1.5,shrinkA=1,shrinkB=2))
    for x,y,title,sub in nodes:
        box=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=.007,rounding_size=.02",facecolor="#ffffff",edgecolor="#bad0c7",lw=1.2)
        ax.add_patch(box)
        ax.text(x,y+(.075 if compact else .025),title,ha="center",va="center",color="#134c44",fontsize=11,weight="bold")
        ax.text(x,y-(.085 if compact else .038),sub,ha="center",va="center",color="#405951",fontsize=8)
    fig.savefig(ROOT/(name+".png"),dpi=165,bbox_inches="tight",pad_inches=.12)
    plt.close(fig)
if __name__=="__main__":
    nodes=[(.10,.8,"Transcript","Original wording"),(.10,.22,"Audio","Mono · 16 kHz"),(.31,.86,"DistilBERT","Masked mean · 768"),
           (.31,.60,"Text sentiment","Three probabilities"),(.31,.35,"Vocal emotion","Four probabilities"),(.31,.10,"Prosody","44 descriptors"),
           (.55,.76,"Modality gate","48 coordinates"),(.55,.27,"Incongruity","9 comparisons"),
           (.76,.53,"Fusion head","GELU · dropout"),(.94,.53,"Sarcasm","Sigmoid score")]
    nodes[-1]=(.908,.53,"Sarcasm","Sigmoid score")
    nodes[-2]=(.714,.53,"Fusion head","GELU · dropout")
    nodes[6]=(.51,.76,"Modality gate","48 coordinates")
    nodes[7]=(.51,.27,"Incongruity","9 comparisons")
    render("architecture",nodes,[(0,2),(0,3),(1,4),(1,5),(2,6),(5,6),(3,7),(4,7),(6,8),(7,8),(8,9)],(13,3.5))
    render("data-flow",[(.1,.5,"Resources","Pinned files + hashes"),(.3,.5,"Validated pairs","Target WAV + metadata"),(.5,.5,"Frozen features","Per-branch disk cache"),(.7,.5,"Train + validate","Split IDs + checkpoints"),(.9,.5,"Test + deploy","Held-out scores + app")],[(0,1),(1,2),(2,3),(3,4)],(13,2))
