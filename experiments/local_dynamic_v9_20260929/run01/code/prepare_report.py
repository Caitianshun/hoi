"""Collect private measured costs and report inputs; never invent a decision."""
from common import *
import numpy as np


def training_jobs(jobs, scene, mode):
    selected=[]
    for job in jobs:
        command=job['command']
        if not any(Path(x).name=='train_scene.py' for x in command):continue
        if '--scene' not in command or '--mode' not in command:continue
        if command[command.index('--scene')+1]==scene and command[command.index('--mode')+1]==mode:
            selected.append(job)
    assert selected, (scene, mode)
    return selected


def run():
    assert read(RUN/'protocol/independent_verification.json')['status']=='passed'
    jobs=[json.loads(x) for x in (RUN/'protocol/GPU_jobs.jsonl').read_text().splitlines()]
    cpu=[json.loads(x) for x in (RUN/'protocol/CPU_jobs.jsonl').read_text().splitlines()]
    cpu=[x for x in cpu if x['label'] in ['all_terminal_evaluation','independent_verification']]
    branches=[];initializations={}
    for scene in config()['scenes']:
        sd=scene_dir(scene);initializations[scene]=read(sd/'protocol/initialization.json')
        bg_seconds=sum(j['wall_seconds'] for j in training_jobs(jobs,scene,'BG'))
        for mode in ['BG','Q0','S','SL']:
            out=sd/'runs'/mode;r=read(out/'run.json')
            density=[json.loads(x) for x in (out/'density_events.jsonl').read_text().splitlines()] if (out/'density_events.jsonl').exists() else []
            attempts=training_jobs(jobs,scene,mode)
            seconds=sum(j['wall_seconds'] for j in attempts)
            complete_memory_measurement=len(attempts)==1 and attempts[0]['returncode']==0
            metric_rows=[json.loads(x) for x in (out/'training_metrics.jsonl').read_text().splitlines()]
            branches.append(dict(scene=scene,mode=mode,**{k:r[k] for k in ['completed_updates','final_points','variable_points']},
                peak_points=max([r['peak_points']]+[x['points'] for x in metric_rows]),
                seconds=seconds,final_process_engine_seconds=r['seconds'],process_attempts=attempts,
                peak_allocated_bytes=r['peak_allocated_bytes'] if complete_memory_measurement else None,
                peak_reserved_bytes=r['peak_reserved_bytes'] if complete_memory_measurement else None,
                observed_final_process_peak_allocated_bytes=r['peak_allocated_bytes'],
                observed_final_process_peak_reserved_bytes=r['peak_reserved_bytes'],
                peak_memory_scope='whole job' if complete_memory_measurement else 'whole-job peak unavailable; final process only is a lower bound',
                logical_path_seconds=seconds+(bg_seconds if mode!='BG' else 0),
                added_points=sum(max(0,x['points_after']-x['points_before']) for x in density),
                removed_points=sum(max(0,x['points_before']-x['points_after']) for x in density),checkpoint=r['checkpoint']))
    costs=dict(GPU_task_seconds=sum(x['wall_seconds'] for x in jobs),GPU_task_hours=sum(x['wall_seconds'] for x in jobs)/3600,
        CPU_job_seconds=sum(x['wall_seconds'] for x in cpu),CPU_scope='Evaluation and independent verification only; report preparation, authoring and interactive QA excluded',GPU_jobs=jobs,CPU_jobs=cpu,branches=branches,
        counting='GPU process wall time including load/save, not CUDA kernel time. Each shared BG parent counted once in actual total, included per logical path separately.',
        verification=read(RUN/'protocol/independent_verification.json'))
    save_json(RUN/'costs.json',costs)
    ev=read(RUN/'evaluation_summary.json')
    pareto={}
    for scene,s in ev['scenes'].items():
        pareto[scene]={}
        for reg in ['full','foreground','background']:
            vectors={a:np.array([s['summaries'][a]['retained'][reg][k] for k in ['psnr_db','ssim','lpips_spatial_mean']])*[1,1,-1] for a in ['Q0','S','SL']}
            pareto[scene][reg]={a:[b for b in vectors if a!=b and np.all(vectors[b]>=vectors[a]) and np.any(vectors[b]>vectors[a])] for a in vectors}
    save_json(RUN/'protocol/report_inputs.json',dict(status='ready_for_complete_vector_and_visual_assessment',
        pareto_dominated_by=pareto,all_terminals=identity(RUN/'protocol/finals.json'),
        metrics=identity(RUN/'evaluation_summary.json'),costs=identity(RUN/'costs.json'),
        no_automatic_scalar_score=True,development_scenes=list(config()['scenes']),
        pending=['Review all fixed full/FG/train images','State actual metric tradeoffs and retained configuration','Create embedded DOCX with bundled runtime and inspect all rendered pages','Package and verify feedback ZIP','Sync source and close execution log']))


if __name__=='__main__':run()
