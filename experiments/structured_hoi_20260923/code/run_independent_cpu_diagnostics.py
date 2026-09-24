"""Event-driven CPU queue for frozen exports; never dispatches GPU work."""
from pathlib import Path
import ctypes,json,os,subprocess,time,traceback,sys
ROOT=Path('/home/cai_tianshun/Project/HOI');E=ROOT/'experiments/structured_hoi_20260923';status=E/'independent_cpu_diagnostics.json'
def save(rec):
    p=status.with_suffix('.tmp');p.write_text(json.dumps(rec,indent=2)+'\n');p.replace(status)
def wait_export(path):
    lib=ctypes.CDLL(None,use_errno=True);fd=lib.inotify_init1(0);watched=set()
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        while True:
            for parent in [E,path.parent]:
                while not parent.exists():parent=parent.parent
                if parent not in watched:
                    if lib.inotify_add_watch(fd,str(parent).encode(),0x8|0x80|0x100)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
                    watched.add(parent)
            if path.exists():
                x=json.loads(path.read_text())
                if x.get('status')=='completed' and x.get('evaluation_status')=='completed':return
                if x.get('status')=='failed':raise RuntimeError(f'Export failed: {path}')
            recovery=E/'recovery_and_dev2_pipeline.json'
            if recovery.exists() and json.loads(recovery.read_text()).get('status')=='failed':raise RuntimeError('Root GPU pipeline failed')
            os.read(fd,65536)
    finally:os.close(fd)
def main():
    record={'status':'running','pid':os.getpid(),'GPU_used':False,'event_driven_wait':'inotify','stages':{}};save(record);start=time.perf_counter()
    try:
        for dev in ['dev1','dev2']:
            seg=ROOT/'experiments/mosca_baseline_20260922/segmentation/segmentation.npz' if dev=='dev1' else E/'data/dev2/segmentation/segmentation.npz'
            if dev=='dev2':
                import torch
                seg=Path(torch.load(E/'dev2_initialization/initialization.pt',map_location='cpu',weights_only=False)['segmentation'])
            for branch in ['S0','S1']:
                export=E/f'{dev}_{branch}_v1'/('evaluation_v2' if dev=='dev1' else 'evaluation')
                record['current_stage']=f'waiting_{dev}_{branch}';save(record);wait_export(export/'export_manifest.json')
                jobs=[('support',[sys.executable,E/'code/diagnose_structured_support_cpu.py','--export',export,'--segmentation',seg,'--output',export/'support_diagnostic'])]
                if dev=='dev2':jobs.insert(0,('surface',[sys.executable,E/'code/score_structured_surface_cpu.py','--export',export,'--reference',E/'data_audit/dev2_evaluation/reference_meshes_world.npz','--output',export/'surface_reference']))
                for kind,cmd in jobs:
                    name=f'{dev}_{branch}_{kind}';record['current_stage']=name;save(record);t=time.perf_counter()
                    with (E/f'{name}.log').open('w') as f:r=subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,check=True)
                    record['stages'][name]={'status':'completed','wall_seconds':time.perf_counter()-t,'command':[str(x) for x in cmd]};save(record)
        record['status']='completed'
    except BaseException as exc:record.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:record['wall_seconds']=time.perf_counter()-start;save(record)
if __name__=='__main__':main()
