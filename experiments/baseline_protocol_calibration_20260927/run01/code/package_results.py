"""Collect measured costs and a small portable research feedback bundle.

This does not train, render or edit an experiment result. Models/data stay local.
"""
import argparse
import csv
import json
import subprocess
import time
import zipfile
from pathlib import Path
from adapter_4dgs import ROOT,RUN,sha,save_json


def read(p):return json.loads(Path(p).read_text())
def identity(p):
    p=Path(p);return dict(path=str(p.absolute()),sha256=sha(p),bytes=p.stat().st_size)


def costs():
    ledger=read(RUN/'protocol/gpu_cost_ledger.json')
    total=sum(x['wall_seconds'] for x in ledger)
    runs=[]
    for dataset in ['behave_dev1','behave_dev2','hos_backpack']:
        base=RUN/'runs'/f'{dataset}_formal';r=read(base/'run.json')
        record={k:r[k] for k in ['status','nominal_iterations','optimizer_updates','seconds','final_points','peak_points',
                                'peak_allocated_bytes','peak_reserved_bytes','checkpoint','checkpoint_sha256']}
        attempt=read(RUN/'logs'/f'{dataset}_formal'/'attempt.json')
        record.update(dataset=dataset,process_wall_seconds=attempt['wall_seconds'],
                      peak_process_nvidia_MiB=attempt['peak_process_nvidia_MiB'],GPU=r['GPU'],
                      final_resume_checkpoint_bytes=Path(r['checkpoint']).stat().st_size,
                      final_inference_files=[identity(p) for p in sorted((base/'point_cloud/fine_iteration_14000').glob('*')) if p.is_file()])
        runs.append(record)
    prep=[]
    init=RUN/'inputs/hos_backpack/initialization.json'
    if init.exists():
        r=read(init)
        for k in ['wall_seconds','seconds','elapsed_seconds']:
            if k in r:prep.append(dict(name='HOS train-only features and triangulation',seconds=r[k],device='CPU',note='Measured preprocessing; download/installation excluded'));break
    for name in ['behave_dev1_native_cpu_check.json','behave_dev2_native_cpu_check.json','hos_independent_source_review.json']:
        p=RUN/'protocol'/name
        if p.exists():
            r=read(p)
            if 'wall_seconds' in r:prep.append(dict(name=name,seconds=r['wall_seconds'],device='CPU',note='Independent protocol/identity check'))
    vals=dict(status='completed',formal_runs=runs,gpu_job_ledger=ledger,
       totals=[dict(label='正式名义迭代',value=sum(r['nominal_iterations'] for r in runs),unit='轮',note='3×(3000+14000)'),
               dict(label='正式参数更新',value=sum(r['optimizer_updates'] for r in runs),unit='次',note='保留官方最后fine迭代不step的行为'),
               dict(label='临时优化',value=len((RUN/'protocol/temporary_steps.jsonl').read_text().splitlines()),unit='步',note='50+50+100，未进入正式初始化'),
               dict(label='GPU占用任务累计墙钟',value=total,unit='秒',note='持久launcher含加载与导出；早期短检查只计driver块，少量Python导入开销未单独测量'),
               dict(label='GPU任务预算比例',value=total/43200*100,unit='%',note='12 GPU小时上限，未把PyTorch allocated当整卡用量')],
       preprocessing=prep,
       notes=['全部GPU作业为本机物理GPU1 RTX3090。无GPU0、远端或5090训练。',
              'nvidia-smi按子进程PID每5秒采样，可能漏过短峰；PyTorch峰值allocated与reserved另列。',
              '旧SMPL-X/RGB姿态、SAM2、UniDepth和S1等复用成本不记作本轮首次零成本；历史成本见原实验。',
              'H0使用既有官方200000步检查点，不在本轮从零训练；下载、解压及环境适配是工程成本。',
              'CPU统一评价与DOCX排版不消耗GPU预算，但墙钟成本单列。'],
       CPU_evaluation_seconds={
          'BEHAVE':read(RUN/'evaluation/comparison/evaluation_run.json')['seconds'],
          'HOS':read(RUN/'evaluation/hos_comparison/summary.json')['wall_seconds']},
       source=identity(RUN/'protocol/gpu_cost_ledger.json'))
    assert total<43200
    save_json(RUN/'costs.json',vals)
    return vals


def feedback():
    assert read(RUN/'protocol/final_integrity.json')['status']=='passed'
    doc=RUN/'output/V3_baseline_calibration.docx';assert doc.exists()
    # This bundle is a local review artifact, not an upload/publication.
    selected=[]
    for name in ['PROTOCOL.md','REPRODUCE.md','NEXT_DECISION.md','MISSING_ASSETS.md','HOS_ASSETS.md',
                 'LITERATURE_PROTOCOL_AUDIT.md','ERROR_ATTRIBUTION.md','existing_error_budget.csv',
                 'existing_error_summary.json','input_fit_existing.csv','costs.json','model_index.json','asset_sources.json','split_manifest.json']:
        p=RUN/name
        if p.exists():selected.append(p)
    selected += [doc]
    for directory in ['configs','code','protocol','evaluation/comparison','evaluation/hos_comparison']:
        for p in (RUN/directory).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.json','.csv','.yaml','.yml','.md','.txt','.png'} and '__pycache__' not in p.parts:
                selected.append(p)
    for ds in ['behave_dev1','behave_dev2','hos_backpack']:
        for n in ['manifest.json','evaluation_manifest.json','shared_init.json']:
            p=RUN/'inputs'/ds/n
            if p.exists():selected.append(p)
        for n in ['run.json','effective_config.json','projection_check.json','upstream_adaptation.json']:
            p=RUN/'runs'/f'{ds}_formal'/n
            if p.exists():selected.append(p)
    for p in (RUN/'logs').rglob('attempt.json'):selected.append(p)
    selected=sorted(set(selected))
    out=RUN/'output/baseline_calibration_feedback.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in selected:z.write(p,p.relative_to(RUN))
    save_json(RUN/'feedback_index.json',dict(zip=identity(out),files=[identity(p) for p in selected],
             excludes=['datasets','raw input RGB','raw rendered NPZ','third-party source copies','model checkpoints','virtual environments'],
             code_is_source_only=True,created_unix=time.time()))
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['costs','feedback']);a=p.parse_args()
    costs() if a.action=='costs' else feedback()
