#!/usr/bin/env python3
"""CPU same-reference/full-coverage and common-valid comparison; no model ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_trajectories import load_bundle,statistics,write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models",nargs="+",required=True,help="name=/absolute/three_dimensional/report.json")
    p.add_argument("--reference",type=Path,required=True,help="reference with query_id/prediction_frame_indices")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError("output must be new/empty")
    reports={};datasets={};sources={}
    for item in a.models:
        name,path=item.split("=",1);path=Path(path)
        if name in reports:raise ValueError("model names must be unique")
        reports[name]=json.loads(path.read_text());datasets[name]=load_bundle(Path(reports[name]["input"]))
        sources[name]={"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    base=next(iter(datasets.values()))
    for data in datasets.values():
        if data.frame!=base.frame or not np.array_equal(data.times,base.times) or not np.array_equal(data.valid,base.valid) or not np.array_equal(data.entity,base.entity) or not np.array_equal(data.reference,base.reference,equal_nan=True):
            raise ValueError("model reports must share identical reference, timestamps, entity order and coordinates")
    with np.load(a.reference,allow_pickle=False) as z:ids=z["query_id"].copy();frame_indices=z["prediction_frame_indices"].copy()
    common=base.valid.copy()
    for data in datasets.values():common &= data.pred_valid
    common_displacement=common&common[:1];common_displacement[0]=False
    paired=[];details=[];event_rows=[];full=[]
    errors={};displacements={}
    for name,data in datasets.items():
        errors[name]=np.linalg.norm(data.predicted-data.reference,axis=-1)
        displacements[name]=np.linalg.norm((data.predicted-data.predicted[:1])-(data.reference-data.reference[:1]),axis=-1)
        for group in ["all",*np.unique(data.entity).tolist()]:
            groupmask=np.ones_like(common) if group=="all" else np.broadcast_to(data.entity==group,common.shape)
            for metric,error,expected in [("absolute_epe",errors[name],common),("first_frame_displacement",displacements[name],common_displacement)]:
                paired.append({"method":name,"entity":group,"metric":metric,**statistics(error,expected&groupmask,np.ones_like(common),[.01,.02,.05])})
        full.append({"method":name,"absolute":reports[name]["results"]["summaries"][0],"displacement":reports[name]["results"]["displacement_summaries"][0]})
        for event in reports[name]["events_definition"]:
            start,end=event["occlusion_start"],event["occlusion_end"]
            for phase,lo,hi in [("before",max(0,start-event["pre_frames"]),start),("during",start,end),("after",end,min(len(data.times),end+event["post_frames"]))]:
                temporal=np.zeros_like(common);temporal[lo:hi]=True
                for entity in np.unique(data.entity):
                    mask=temporal&data.valid&(data.entity==entity)
                    disp_mask=mask&data.valid[:1];disp_mask[0]=False
                    for metric,error,expected,present in [("absolute_epe",errors[name],mask,data.pred_valid),("first_frame_displacement",displacements[name],disp_mask,data.pred_valid&data.pred_valid[:1])]:
                        event_rows.append({"method":name,"event_id":event["event_id"],"phase":phase,"entity":entity,"metric":metric,"start_time_s":float(data.times[lo]) if lo<hi else None,"last_time_s":float(data.times[hi-1]) if lo<hi else None,"reference_frame_count":hi-lo,**statistics(error,expected,present,[.01,.02,.05])})
    for ti,qi in np.argwhere(common):
        row={"reference_index":int(ti),"input_frame_index":int(frame_indices[ti]),"time_s":float(base.times[ti]),"query_id":str(ids[qi]),"entity":str(base.entity[qi])}
        for name in datasets:row[f"{name}_absolute_epe_m"]=float(errors[name][ti,qi]);row[f"{name}_displacement_epe_m"]=float(displacements[name][ti,qi]) if common_displacement[ti,qi] else None
        details.append(row)
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/"paired_summary.csv",paired);write_csv(a.output/"paired_samples.csv",details);write_csv(a.output/"per_entity_event_summary.csv",event_rows)
    report={"schema_version":1,"sources":sources,"reference_expected_samples":int(base.valid.sum()),"common_valid_samples":int(common.sum()),"common_fraction":float(common.sum()/base.valid.sum()),"common_displacement_samples_excluding_anchor":int(common_displacement.sum()),"selection_warning":"This common-valid subset is selected by participating methods' valid/visibility outputs; it is not unbiased, not a replacement for all-point coverage, and excludes their shared failures.","full_sample_summaries":full,"paired_summaries":paired,"per_entity_events":event_rows,"GT_visibility":"unknown","events":"Manual input RGB phases only; each event phase here contains a single sparse reference observation; no dense recovery time claim"}
    (a.output/"comparison.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    def cm(value):return "—" if value is None else f"{100*value:.2f}"
    lines=["# 固定查询的配对诊断（尚非标准基准）","","仅比较同一段真实HOI的6个首帧固定表面查询、14个独立拟合参考观测。坐标为BEHAVE k1世界系，单位米；评价没有拟合对齐。手套模糊、人体/物体拟合及名义时间偏差仍限制参考精度。","","## 全部参考样本：均值必须与覆盖率一起读","","| 方法 | 绝对EPE均值/中位（cm） | 绝对覆盖 | 位移EPE均值/中位（cm） | 位移覆盖 |","| --- | --- | --- | --- | --- |"]
    for row in full:
        s,d=row["absolute"],row["displacement"]
        lines.append(f"| {row['method']} | {cm(s['mean_epe_m'])}/{cm(s['median_epe_m'])} | {s['predicted_samples']}/{s['expected_samples']} | {cm(d['mean_epe_m'])}/{cm(d['median_epe_m'])} | {d['predicted_samples']}/{d['expected_samples']} |")
    lines.extend(["","位移误差比较每点相对首帧的运动变化；没有把参考点用作模型对齐。首帧自身位移恒为0，因此排除；锚点或当前预测缺失均计失败。","","## 同时有预测的配对子集","",f"共有{int(common.sum())}/{int(base.valid.sum())}个绝对误差样本、{int(common_displacement.sum())}个位移样本。**该子集由模型有效性/可见性选择，有选择偏差。** 不覆盖被排除的遮挡难例，不能替代上表。","","| 方法 | 配对绝对EPE均值/中位（cm） | 配对位移EPE均值/中位（cm） |","| --- | --- | --- |"])
    for name in datasets:
        s=next(x for x in paired if x['method']==name and x['entity']=='all' and x['metric']=='absolute_epe');d=next(x for x in paired if x['method']==name and x['entity']=='all' and x['metric']=='first_frame_displacement')
        lines.append(f"| {name} | {cm(s['mean_epe_m'])}/{cm(s['median_epe_m'])} | {cm(d['mean_epe_m'])}/{cm(d['median_epe_m'])} |")
    lines.extend(["","## 人工RGB事件内的人/物分别结果","","事件阶段来自此前独立查看输入RGB固定的区间，未使用SAM2或tracker可见性决定边界。每阶段只有一个稀疏参考时刻，因此以下是具体观测，不是遮挡机制的充分证据；逐点GT可见性仍未知。","","| 方法 | 阶段 | 实体 | 绝对EPE均值（cm）/覆盖 | 位移EPE均值（cm）/覆盖 |","| --- | --- | --- | --- | --- |"])
    for name in datasets:
        for phase in ['before','during','after']:
            for entity in np.unique(base.entity):
                same=[r for r in event_rows if r['method']==name and r['phase']==phase and r['entity']==entity]
                if not same:continue
                s=next(r for r in same if r['metric']=='absolute_epe');d=next(r for r in same if r['metric']=='first_frame_displacement')
                lines.append(f"| {name} | {phase} | {entity} | {cm(s['mean_epe_m'])} / {s['predicted_samples']}/{s['expected_samples']} | {cm(d['mean_epe_m'])} / {d['predicted_samples']}/{d['expected_samples']} |")
    lines.extend(["","不能因持续输出轨迹便称三维几何更准确，也不能因可见点条件误差较小便称恢复更好。CoTracker+深度在不可见时不产生3D点；Point4D提供持续坐标，但坐标还依赖与输入侧单目深度的一次全局Sim(3)转换，转换的先验不一致需与重建误差分开分析。","",f"详细机器可读结果：[comparison.json]({(a.output/'comparison.json').resolve()})；逐样本配对：[paired_samples.csv]({(a.output/'paired_samples.csv').resolve()})。"])
    (a.output/"COMPARISON.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"output":str(a.output.resolve()),"paired_count":int(common.sum()),"paired_summaries":[r for r in paired if r['entity']=='all']},ensure_ascii=False))


if __name__=="__main__":main()
