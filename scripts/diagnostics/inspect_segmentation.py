#!/usr/bin/env python3
"""CPU selected-frame RGB/mask overlays plus mask-area audit; no GT inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segmentation",required=True,type=Path)
    p.add_argument("--input-manifest",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--indices",nargs="+",type=int)
    p.add_argument("--crop",nargs=4,type=int,help="optional x0 y0 x1 y1, fixed all frames")
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.input_manifest.read_text()); paths=manifest["frame_paths"]
    with np.load(a.segmentation,allow_pickle=False) as d: labels=d["entity_labels"]
    if len(paths)!=len(labels): raise ValueError("frame count differs")
    indices=a.indices or np.linspace(0,len(paths)-1,12).round().astype(int).tolist()
    if any(i<0 or i>=len(paths) for i in indices): raise ValueError("invalid selected index")
    fig,axes=plt.subplots(len(indices)//3+(len(indices)%3>0),3,figsize=(15,4.5*(len(indices)//3+(len(indices)%3>0))),squeeze=False)
    times=manifest.get("timestamp_seconds",manifest.get("frame_times_seconds",manifest.get("timestamps_seconds")))
    for ax,idx in zip(axes.ravel(),indices):
        image=np.asarray(Image.open(paths[idx]).convert("RGB"),dtype=float)/255
        colors=np.zeros_like(image); colors[labels[idx]==1]=[0,0.9,1]; colors[labels[idx]==2]=[1,0.1,0.1]
        alpha=(labels[idx]>0)[...,None]*0.38
        overlay=image*(1-alpha)+colors*alpha
        if a.crop:
            x0,y0,x1,y1=a.crop; image=image[y0:y1,x0:x1];overlay=overlay[y0:y1,x0:x1]
        ax.imshow(np.concatenate([image,overlay],axis=1));ax.axis("off")
        title=f"Frame {idx}: RGB | person cyan, object red"
        if times: title+=f"; {times[idx]:.3f}s"
        ax.set_title(title,fontsize=9)
    for ax in axes.ravel()[len(indices):]:ax.axis("off")
    fig.suptitle("Estimated SAM2 priors — visual QA only, not ground-truth visibility",fontsize=15)
    fig.tight_layout();fig.savefig(a.output/"segmentation_overlay.png",dpi=140);plt.close(fig)
    areas={name:(labels==label).sum(axis=(1,2)) for label,name in [(1,"person"),(2,"object")]}
    fig,ax=plt.subplots(figsize=(10,3))
    for name,values in areas.items():ax.plot(values,label=name)
    ax.set(xlabel="Processed frame index",ylabel="Estimated mask area (pixels)",title="Mask area; zero is not proof of true occlusion")
    ax.legend();fig.tight_layout();fig.savefig(a.output/"segmentation_area.png",dpi=140);plt.close(fig)
    audit={"selected_indices":indices,"crop":a.crop,"pixel_area":{key:val.tolist() for key,val in areas.items()},
           "empty_mask_frames":{key:np.flatnonzero(val==0).tolist() for key,val in areas.items()},
           "note":"Numeric mask audit; visual observations must be recorded separately after inspection"}
    (a.output/"segmentation_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"overlay":str((a.output/"segmentation_overlay.png").resolve()),"empty_mask_frames":audit["empty_mask_frames"]}))


if __name__=="__main__":main()
