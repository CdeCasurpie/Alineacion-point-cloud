"""Deterministic categorical colors, with numeric map labels preserved as IDs."""
import colorsys
from .contracts import digest


def palette(manifests):
    result={};used=set()
    for ref in sorted(manifests,key=lambda r:manifests[r]["block_id"]):
        label=manifests[ref]["block_id"]
        value=int(label) if label.isdecimal() and 0<int(label)<=65535 else int(digest(label)[:8],16)%65535+1
        if value in used:
            raise ValueError("Categorical block ID collision; numeric IDs must be unique")
        used.add(value)
        hue=((value-1)*.618033988749895)%1
        rgb=[round(255*x) for x in colorsys.hsv_to_rgb(hue,.78,.92 if value%2 else .75)]
        result[ref]={"block_id":value,"label":label,"rgb":rgb,"hex":"#"+"".join(f"{c:02x}" for c in rgb)}
    return result


def render_legend(colors,path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ordered=sorted(colors.values(),key=lambda r:r["label"])
    fig,ax=plt.subplots(figsize=(6,max(2,.32*len(ordered))))
    ax.axis("off")
    for i,c in enumerate(ordered):
        ax.add_patch(patches.Rectangle((0,len(ordered)-i-.7),.6,.5,color=c["hex"]))
        ax.text(.8,len(ordered)-i-.5,f"Cuadra {c['label']}  |  block_id={c['block_id']}  |  {c['hex']}",va="center",fontsize=9)
    ax.set_xlim(0,7);ax.set_ylim(0,len(ordered)+.5)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
