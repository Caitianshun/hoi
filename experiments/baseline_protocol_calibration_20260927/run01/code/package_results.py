"""Collect measured costs and a small portable research feedback bundle.

This does not train, render or edit an experiment result. Models/data stay local.
"""
import argparse
import csv
import json
import posixpath
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from adapter_4dgs import ROOT,RUN,sha,save_json


def read(p):return json.loads(Path(p).read_text())
def identity(p):
    p=Path(p);return dict(path=str(p.absolute()),sha256=sha(p),bytes=p.stat().st_size)


def costs():
    integrity=read(RUN/'protocol/final_integrity.json')
    assert integrity['status']=='passed','Numerical verification must precede cost delivery'
    assert sha(integrity['model_index']['path'])==integrity['model_index']['sha256']
    ledger=read(RUN/'protocol/gpu_cost_ledger.json')
    total=sum(x['wall_seconds'] for x in ledger)
    assert total==integrity['budget']['gpu_job_wall_seconds'],'GPU ledger changed after verification'
    runs=[]
    for dataset in ['behave_dev1','behave_dev2','hos_backpack']:
        base=RUN/'runs'/f'{dataset}_formal';r=read(base/'run.json')
        assert r['status']=='completed' and r['nominal_iterations']==17000 and r['optimizer_updates']==16999
        record={k:r[k] for k in ['status','nominal_iterations','optimizer_updates','seconds','final_points','peak_points',
                                'peak_allocated_bytes','peak_reserved_bytes','checkpoint','checkpoint_sha256']}
        attempt=read(RUN/'logs'/f'{dataset}_formal'/'attempt.json')
        assert attempt['status']=='completed'
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
            for key in ['wall_seconds','seconds','elapsed_seconds']:
                if key in r:
                    prep.append(dict(name=name,seconds=r[key],device='CPU',note='Independent protocol/identity check',source=identity(p)));break
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
              'CPU统一评价不消耗GPU预算，墙钟成本单列；DOCX构建及视觉验收见独立报告记录。',
              '工程失败与修复未从反馈包删除；失败状态、日志摘要及补丁与HANDOFF/阶段决定一同保留，未测工程耗时不填零。'],
       CPU_evaluation_seconds={
          'BEHAVE':read(RUN/'evaluation/comparison/evaluation_run.json')['seconds'],
          'HOS':read(RUN/'evaluation/hos_comparison/summary.json')['wall_seconds']},
       source=identity(RUN/'protocol/gpu_cost_ledger.json'),numerical_verification=identity(RUN/'protocol/final_integrity.json'))
    assert total<43200
    save_json(RUN/'costs.json',vals)
    return vals


def check_docx_structure(doc):
    """Check embedded image relationships, not visual layout or cross-host UI."""
    with zipfile.ZipFile(doc) as z:
        assert z.testzip() is None,'Corrupt DOCX ZIP member'
        names=set(z.namelist());media=[n for n in names if n.startswith('word/media/')]
        assert media,'Report must contain embedded evidence images'
        rels=ET.fromstring(z.read('word/_rels/document.xml.rels'))
        image_ids={}
        for rel in rels:
            if rel.attrib.get('Type','').endswith('/image'):
                assert rel.attrib.get('TargetMode')!='External','Report image has external dependency'
                target=posixpath.normpath(posixpath.join('word',rel.attrib['Target']))
                assert target in names,(rel.attrib['Id'],target)
                image_ids[rel.attrib['Id']]=target
        body=ET.fromstring(z.read('word/document.xml'))
        refs=[]
        for node in body.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
            relid=node.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            assert relid in image_ids,'Body image must reference embedded package resource'
            refs.append(relid)
        assert refs
    return dict(status='embedded_resource_structure_passed',embedded_media=len(media),
                body_image_references=len(refs),external_image_dependencies=0,
                visual_layout_verified_by_this_function=False,cross_host_application_test=False)


def incident_materials():
    """Preserve small failure states and bounded engineering-log excerpts."""
    files=[];logs=[]
    # Includes suffixes such as .json.failure_01, which suffix-only filters miss.
    for folder in [RUN,RUN/'protocol',RUN/'logs',RUN/'runs']:
        iterator=folder.glob('*') if folder==RUN else folder.rglob('*')
        for p in iterator:
            if p.is_file() and any(word in p.name.lower() for word in ['failure','failed']):
                if p.stat().st_size<=1<<20:
                    try:p.read_text()
                    except (UnicodeDecodeError,OSError):continue
                    files.append(p)
    candidates=[RUN/'logs'/name for name in [
        'rasterizer_build.log','rasterizer_build_fixed.log',
        'hos_environment.log','hos_environment_metrics.log',
        'hos_checkpoint_load.log','hos_checkpoint_load_02.log']]
    candidates += [p for p in (RUN/'logs').rglob('*') if p.is_file() and 'failure' in p.name.lower()]
    for p in (RUN/'logs').rglob('attempt.json'):
        if read(p).get('status')=='failed':
            files.append(p);candidates.append(p.with_name('stdout.log'))
    for p in sorted(set(candidates)):
        if not p.is_file():continue
        content=p.read_text(errors='replace');lines=content.splitlines()
        # Preserve the first and last lines and explicit error context. The full
        # log stays local under its identity; this is a review-scale excerpt.
        chosen=set(range(min(25,len(lines))))|set(range(max(0,len(lines)-60),len(lines)))
        for i,line in enumerate(lines):
            if any(word in line.lower() for word in ['error','traceback','failed','exception']):
                chosen.update(range(max(0,i-2),min(len(lines),i+4)))
        excerpt=[dict(line=i+1,text=lines[i][:3000]) for i in sorted(chosen)[:400]]
        logs.append(dict(source=identity(p),total_lines=len(lines),excerpt=excerpt,
                         excerpt_limit='first25/last60/error contexts, max400 lines, max3000 chars per line; not full log'))
    summary=dict(status='engineering_records_collected',failure_state_files=[identity(p) for p in sorted(set(files))],
        log_excerpts=logs,interpretation='Engineering failures and fixes are evidence, not additional successful formal runs. HANDOFF/NEXT_DECISION provide the root-authored explanation; missing elapsed time remains unmeasured.')
    path=RUN/'output/engineering_incidents.json';save_json(path,summary)
    return sorted(set(files))+[path]


def feedback():
    assert read(RUN/'protocol/final_integrity.json')['status']=='passed'
    doc=RUN/'output/V3_baseline_calibration.docx';assert doc.exists()
    doc_structure=check_docx_structure(doc)
    build_audit=RUN/'output/report_build_audit.json'
    assert build_audit.is_file(),'Report build identity must accompany the document'
    assert read(build_audit)['docx']['sha256']==sha(doc),'DOCX changed since build audit'
    # This bundle is a local review artifact, not an upload/publication.
    selected=[]
    for name in ['PROTOCOL.md','REPRODUCE.md','NEXT_DECISION.md','MISSING_ASSETS.md','HOS_ASSETS.md','HANDOFF.md','report_content.json',
                 'LITERATURE_PROTOCOL_AUDIT.md','ERROR_ATTRIBUTION.md','existing_error_budget.csv',
                 'existing_error_summary.json','input_fit_existing.csv','costs.json','model_index.json','asset_sources.json','split_manifest.json']:
        p=RUN/name
        if p.exists():selected.append(p)
    selected += [doc]
    selected += incident_materials()
    for directory in ['configs','code','protocol','evaluation/comparison','evaluation/hos_comparison']:
        for p in (RUN/directory).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.json','.csv','.yaml','.yml','.md','.txt','.png'} and '__pycache__' not in p.parts:
                selected.append(p)
    for ds in ['behave_dev1','behave_dev2','hos_backpack']:
        for n in ['manifest.json','evaluation_manifest.json','shared_init.json','initialization.json','sampling_and_support.json']:
            p=RUN/'inputs'/ds/n
            if p.exists():selected.append(p)
        for n in ['run.json','effective_config.json','projection_check.json','upstream_adaptation.json']:
            p=RUN/'runs'/f'{ds}_formal'/n
            if p.exists():selected.append(p)
    for p in (RUN/'logs').rglob('attempt.json'):selected.append(p)
    for p in (RUN/'patches').rglob('*.patch'):selected.append(p)
    for p in (RUN/'output/report_figures').rglob('*.png'):selected.append(p)
    for p in (RUN/'output').rglob('*'):
        name=str(p.relative_to(RUN/'output')).lower()
        if p.is_file() and p.suffix in {'.json','.md','.txt'} and ('audit' in name or 'qa' in name):
            selected.append(p)
    selected=sorted(set(selected))
    out=RUN/'output/baseline_calibration_feedback.zip'
    temp=out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in selected:
            assert p.suffix.lower() not in {'.pt','.pth','.ckpt','.npz','.npy','.ply','.mp4'}
            z.write(p,p.relative_to(RUN))
        z.writestr('FEEDBACK_README.txt','先阅读 output/V3_baseline_calibration.docx。图像已内嵌；output/report_figures为报告组合图，evaluation目录保留全部预定对照图。清单绝对路径只说明原始来源，原始数据、浮点渲染和权重未打包。工程失败与修复见HANDOFF、NEXT_DECISION、output/engineering_incidents.json和patches；嵌入资源结构检查不替代视觉排版验收。\n')
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(selected)+1
        assert 'protocol/finals.json' in z.namelist() and 'protocol/hos_finals.json' in z.namelist()
    temp.replace(out)
    save_json(RUN/'feedback_index.json',dict(zip=identity(out),files=[identity(p) for p in selected],
             excludes=['datasets','raw input RGB','raw rendered NPZ','third-party source copies','model checkpoints','virtual environments'],
             code_is_source_only=True,docx_structure=doc_structure,engineering_failures_preserved=True,
             report_visual_QA_status='See included report QA records; not inferred from embedded-image or ZIP checks',created_unix=time.time()))
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['costs','feedback']);a=p.parse_args()
    costs() if a.action=='costs' else feedback()
