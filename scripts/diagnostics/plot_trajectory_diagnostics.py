#!/usr/bin/env python3
"""CPU scientific figures for fixed-query world trajectories and manual RGB phases."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate_trajectories import load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models",nargs="+",required=True,help="name=three_dimensional/report.json")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();reports={};datasets={}
    for item in a.models:
        name,path=item.split("=",1);reports[name]=json.loads(Path(path).read_text());datasets[name]=load_bundle(Path(reports[name]["input"]))
    base=next(iter(datasets.values()));n=base.predicted.shape[1]
    for data in datasets.values():
        if not np.array_equal(data.reference,base.reference,equal_nan=True) or not np.array_equal(data.times,base.times):raise ValueError("figures require identical reference and times")
    a.output.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(16,6.5*((n+2)//3)))
    for qi in range(n):
        ax=fig.add_subplot((n+2)//3,3,qi+1,projection="3d")
        ref=base.reference[:,qi].copy();ref[~base.valid[:,qi]]=np.nan
        ax.plot(ref[:,0],ref[:,1],ref[:,2],"k.-",label="Fitted reference",linewidth=1.6)
        allpoints=[ref]
        for color,(name,data) in zip(["tab:blue","tab:orange","tab:green","tab:red"],datasets.items()):
            xyz=data.predicted[:,qi].copy();xyz[~data.pred_valid[:,qi]]=np.nan
            ax.plot(xyz[:,0],xyz[:,1],xyz[:,2],".-",color=color,label=name,alpha=.85)
            allpoints.append(xyz)
        points=np.concatenate(allpoints);finite=points[np.isfinite(points).all(-1)]
        if len(finite):
            lo,hi=finite.min(0),finite.max(0);centre=(lo+hi)/2;span=max((hi-lo).max(),.1)*.56
            ax.set_xlim(centre[0]-span,centre[0]+span);ax.set_ylim(centre[1]-span,centre[1]+span);ax.set_zlim(centre[2]-span,centre[2]+span)
        ax.set_box_aspect((1,1,1));ax.set_xlabel("World X (m)",fontsize=8);ax.set_ylabel("World Y (m)",fontsize=8);ax.set_zlabel("World Z (m)",fontsize=8)
        ax.set_title(str(base.query_id[qi]) if base.query_id is not None else f"track{qi}",fontsize=10,pad=14)
        if qi==0:ax.legend(fontsize=8)
    fig.suptitle("Fixed-query trajectories in one world frame; gaps=missing predictions; each panel keeps all outliers",fontsize=12,y=.985)
    fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.92,wspace=.14,hspace=.25)
    fig.savefig(a.output/"world_trajectories.png",dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for color,(name,report) in zip(["tab:blue","tab:orange","tab:green","tab:red"],reports.items()):
        for ri,key,label in [(0,"events","Absolute EPE"),(1,"displacement_events","First-frame displacement"),(2,"relative_pair_events","Fixed hand-object vector")]:
            rows=report["results"].get(key,[])
            if ri==2:rows=[r for r in rows if r["pair_id"]=="all_fixed_pairs"]
            if not rows:continue
            positions=np.arange(len(rows));values=[np.nan if r["mean_epe_m"] is None else r["mean_epe_m"]*100 for r in rows]
            coverage=[np.nan if r["coverage"] is None else r["coverage"] for r in rows]
            axes[ri,0].plot(positions,values,"o-",color=color,label=name)
            axes[ri,1].plot(positions,coverage,"o-",color=color,label=name)
            for ci in range(2):axes[ri,ci].set_xticks(positions,[r["phase"] for r in rows])
            axes[ri,0].set_ylabel(f"{label} (cm)");axes[ri,1].set_ylabel("Valid prediction fraction");axes[ri,1].set_ylim(-.02,1.05)
    for ax in axes.ravel():ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Manual RGB event phases; conditional errors require coverage — one sparse reference time per phase",fontsize=12)
    fig.savefig(a.output/"manual_event_errors_and_coverage.png",dpi=150);plt.close(fig)
    print(str(a.output.resolve()))


if __name__=="__main__":main()
