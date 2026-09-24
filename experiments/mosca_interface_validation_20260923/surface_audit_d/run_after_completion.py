#!/usr/bin/env python3
"""Wait on inotify, then audit the completed D branch using immutable CPU tools."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import ctypes
import hashlib
import json
import select
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
ROOT = EXP.parents[1]
BRANCH = EXP/'d_exact_geometry'
PIPELINE = BRANCH/'pipeline.json'
AUDIT = ROOT/'experiments/mosca_hand_trace_20260923/surface_comparison/audit_surface_branch.py'
A_RESULT = ROOT/'experiments/mosca_hand_trace_20260923/surface_audit/surface_audit.json'
PYTHON = ROOT/'envs/mosca/bin/python'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def dump(p, data):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(p)


def wait_completed():
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
    if fd < 0: raise OSError(ctypes.get_errno(), 'inotify_init1')
    mask = 0x00000008|0x00000080|0x00000100  # CLOSE_WRITE, MOVED_TO, CREATE
    watches = set()
    try:
        while True:
            for folder in [EXP, BRANCH]:
                if folder.exists() and folder not in watches:
                    if libc.inotify_add_watch(fd, os.fsencode(folder), mask)<0:
                        raise OSError(ctypes.get_errno(), str(folder))
                    watches.add(folder)
            if PIPELINE.exists():
                state = load(PIPELINE)
                if state['status'] == 'completed': return state
                if state['status'] == 'failed': raise RuntimeError('D pipeline failed; no final-model diagnosis run')
            # Read state only following filesystem events, never timed GPU polling.
            while True:
                ready, _, _ = select.select([fd], [], [], 60)
                if ready:
                    os.read(fd, 65536)
                    break
    finally:
        os.close(fd)


def compare_results():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a, d = load(A_RESULT), load(OUT/'surface_audit.json')
    def key(row):return (row['query_id'],row['input_frame_index'],row['sample_location'])
    aa={key(r):r for r in a['rows']};dd={key(r):r for r in d['rows']}
    assert len(aa)==len(dd)==56 and aa.keys()==dd.keys()
    protocol_fields=['time_seconds','reference_index','reference_uv','reference_world_m','reference_camera_z_m','nominal_reference_time_seconds','reference_time_offset_seconds','tracker_uv','tracker_claims_visible','tracker_reference_pixel_distance','uv']
    for k in aa:
        for field in protocol_fields:
            assert np.array_equal(np.asarray(aa[k][field]),np.asarray(dd[k][field])),(k,field)
    rows=[]
    for query in ['hand_glove_centre','hand_glove_upper']:
        for location in ['training_tracker','reference_projection']:
            sa=a['summary'][query][location];sd=d['summary'][query][location]
            row={'query_id':query,'sample_location':location,'count':14,'tracker_visible_count':5}
            for metric in ['mean_fixed_source_3d_epe_m','mean_abs_depth_difference_m','mean_ray_backprojection_difference_m','alpha_min','dynamic_fraction_mean','abs_depth_within_10cm_count','abs_depth_within_30cm_count']:
                row[metric]={'A':sa[metric],'D':sd[metric],'D_minus_A':sd[metric]-sa[metric]}
            rows.append(row)
    dump(OUT/'comparison_ad.json',{'status':'completed','conditions':56,'protocol_fields_equal':protocol_fields,'A_result_sha256':sha(A_RESULT),'D_result_sha256':sha(OUT/'surface_audit.json'),'rows':rows,'evaluation_only':True,'reference_projection_uses_reference_2d_location':True})
    names=['hand_glove_centre','hand_glove_upper']
    fig,axs=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for i,name in enumerate(names):
        for label, data, style in [('A',aa,'--'),('D',dd,'-')]:
            refrows=sorted([r for r in data.values() if r['query_id']==name and r['sample_location']=='reference_projection'],key=lambda r:r['input_frame_index'])
            tt=[r['time_seconds'] for r in refrows]
            axs[i,0].plot(tt,[100*r['fixed_source_3d_epe_m'] for r in refrows],style,label=label)
            axs[i,1].plot(tt,[100*abs(r['depth_signed_error_to_fitted_point_m']) for r in refrows],style,label=label)
            axs[i,2].plot(tt,[r['dynamic_fraction'] for r in refrows],style,label=label)
        for j,ax in enumerate(axs[i]):
            ax.grid(alpha=.2);ax.legend();ax.set_title(('Centre' if i==0 else 'Upper')+' | '+['fixed source EPE','fitted-UV depth difference','fitted-UV dynamic fraction'][j])
            if i==1:ax.set_xlabel('Actual observation time (s)')
            ax.set_ylabel('cm' if j<2 else 'Fraction')
    fig.suptitle('A vs D: fixed identity and fitted-UV surface probes stay separate',fontsize=14)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.text(.05,.015,'Fitted-UV probes use reference-supplied 2D locations. They are evaluation diagnostics, not independent tracking predictions.',fontsize=10)
    fig.savefig(OUT/'comparison_ad.png',dpi=180);plt.close(fig)
    report=['# A / D：固定源与当前表面分开评价','',
      '**D 已完整训练、导出和评价结束后，执行同一56组条件的CPU表面诊断。** 两个固定手套点、14个已固定参考时刻、两类查询位置；没有用中间检查点替代最终结果。','',
      'A/D 的参考坐标、时间、拟合投影像素、训练tracker像素、可见性输出与查询位置逐元素相同。D 新相机标记与本次源码身份已核对；CPU从D最终模型重新计算各时刻高斯，与D自己的GPU关键帧及首帧源查询交叉检查，不复用A几何缓存。','',
      '| 查询 | 固定源3D EPE A→D | 拟合UV处深度绝对差 A→D | Tracker UV处深度绝对差 A→D |',
      '| --- | ---: | ---: | ---: |']
    for name in names:
        ar=a['summary'][name]['reference_projection'];dr=d['summary'][name]['reference_projection'];at=a['summary'][name]['training_tracker'];dt=d['summary'][name]['training_tracker']
        report.append(f'| {"手套中心" if name==names[0] else "手套上部"} | {100*ar["mean_fixed_source_3d_epe_m"]:.2f}→{100*dr["mean_fixed_source_3d_epe_m"]:.2f}cm | {100*ar["mean_abs_depth_difference_m"]:.2f}→{100*dr["mean_abs_depth_difference_m"]:.2f}cm | {100*at["mean_abs_depth_difference_m"]:.2f}→{100*dt["mean_abs_depth_difference_m"]:.2f}cm |')
    report += ['', '固定源沿首帧固定高斯混合传播，反映这条源关联的误差，但不保证对应同一物理材料点。拟合UV处重新查询的是各时刻当前渲染层，贡献高斯会变化，而且二维位置由评价参考提供；其较低误差不能当作无需参考的新跟踪方法收益。Tracker UV处则同时受tracker漂移和表面解释影响，两手均只有5/14时刻被tracker声称可见，表中没有据此筛掉其余时刻。','',
      '| 查询 / 拟合UV处 | 最小alpha A→D | 平均动态贡献 A→D | 深度差≤10cm时刻数 A→D |','| --- | ---: | ---: | ---: |']
    for name in names:
        ar=a['summary'][name]['reference_projection'];dr=d['summary'][name]['reference_projection']
        report.append(f'| {"手套中心" if name==names[0] else "手套上部"} | {ar["alpha_min"]:.5f}→{dr["alpha_min"]:.5f} | {100*ar["dynamic_fraction_mean"]:.2f}%→{100*dr["dynamic_fraction_mean"]:.2f}% | {ar["abs_depth_within_10cm_count"]}/14→{dr["abs_depth_within_10cm_count"]}/14 |')
    report += ['', '高alpha不等于表面正确，期望深度可能混合前后层；拟合手套表面、遮挡和对应存在不确定性。这56组局部诊断不能代表整个人体连接性或保留视角结构，需要结合根代理的dense表面与保留视角结果。','',
      f'![A/D按相同时刻的固定源误差、拟合UV深度差与动态贡献]({OUT/"comparison_ad.png"})','',
      f'[完整D结果]({OUT/"surface_audit.json"}) · [逐条件CSV]({OUT/"surface_samples.csv"}) · [A/D比较JSON]({OUT/"comparison_ad.json"}) · [完成状态/版本预检]({OUT/"completion_preflight.json"})','',
      '所有操作仅CPU评价；未改模型、未训练、未调整参考或查询。图的视觉检查另记，不能仅凭脚本成功声称已目检。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')


def main():
    start=time.perf_counter()
    state={'status':'waiting_for_completed_pipeline','pipeline':str(PIPELINE),'pid':os.getpid(),'mechanism':'Linux inotify; no timed GPU/state polling','start_epoch':time.time()}
    dump(OUT/'observer.json',state)
    try:
        final=wait_completed()
        import torch
        torch.set_num_threads(4)
        model=BRANCH/'model';run=load(model/'run.json');export=load(BRANCH/'diagnostics/export_manifest.json')
        assert run['status']=='completed' and run['geometry_interface_version']==2
        assert all(s['status']=='completed' for s in final['stages'])
        cam_path=model/'photometric_cam.pth'
        camera=torch.load(cam_path,map_location='cpu',weights_only=False)
        assert int(camera['geometry_interface_version'])==2
        assert sha(cam_path)==export['checkpoint_sha256']['camera']
        source_camera=EXP/'code/MoSca/lib_moca/camera.py'
        assert sha(source_camera)==run['source_identity']['lib_moca/camera.py']
        assert sha(EXP/'code/MoSca/lib_moca/pixel_geometry.py')==run['source_identity']['lib_moca/pixel_geometry.py']
        audit_hash=sha(AUDIT);a_hash=sha(A_RESULT)
        dump(OUT/'completion_preflight.json',{'status':'passed','pipeline_sha256':sha(PIPELINE),'model_run_sha256':sha(model/'run.json'),'export_sha256':sha(BRANCH/'diagnostics/export_manifest.json'),'geometry_interface_version':2,'camera_checkpoint_hash_matches_export':True,'camera_and_mapping_source_hashes_match_training':True,'immutable_audit_script_sha256':audit_hash,'A_result_sha256':a_hash,'checkpoint_directory':str(model),'CPU_reconstruction_uses_branch_models_and_calibrated_input_K':True})
        state['status']='running_cpu_surface_audit';dump(OUT/'observer.json',state)
        with (OUT/'run.log').open('w') as log:
            subprocess.run([str(PYTHON),str(AUDIT),'--branch-dir',str(BRANCH),'--output',str(OUT),'--label','D'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert sha(AUDIT)==audit_hash and sha(A_RESULT)==a_hash
        compare_results()
        state.update(status='completed',completed_epoch=time.time(),total_wall_seconds_including_event_wait=time.perf_counter()-start)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        dump(OUT/'observer.json',state)
    print(json.dumps({'status':state['status'],'report':str(OUT/'REPORT.md')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
