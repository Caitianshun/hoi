"""Aggregate only completed S0/S1 runs; no filtering of scenes or frames."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import cv2
E=Path('/home/cai_tianshun/Project/HOI/experiments/structured_hoi_20260923')

def read(path):return json.loads(path.read_text())
def csvrow(path,**match):
    rows=list(csv.DictReader(path.open()))
    return next(r for r in rows if all(r[k]==v for k,v in match.items()))

def main():
    summary={'status':'completed','protocol':'structured_hoi_known_shape_v1','single_seed':12345,'independent_sequence_count':2,'rows':[],'warnings':['All input image metrics are training fit.','Fitted meshes are approximate references, not sensor ground truth.','S0/S1 equal updates but unequal time; prefix cost included for each equivalent standalone run.','Single seed pilot, not statistical significance or official benchmark.']}
    for dev in ['dev1','dev2']:
        prefix=read(E/f'{dev}_prefix01/run.json');assert prefix['status']=='completed'
        pair=[]
        for b in ['S0','S1']:
            p=E/f'{dev}_{b}_v1';r=read(p/'run.json');assert r['status']=='completed' and r['final_step']==8000
            ev=p/('evaluation_v2' if dev=='dev1' else 'evaluation');x=read(ev/'export_manifest.json');assert x['evaluation_status']=='completed'
            inp=read(ev/'input_image_metrics.json');held=read(ev/'heldout_metrics.json')
            row={'dev':dev,'branch':b,'run':str(p),'evaluation':str(ev),'equivalent_full_train_seconds':prefix['wall_seconds']+r['wall_seconds'],'actual_tail_seconds':r['wall_seconds'],'prefix_shared_seconds':prefix['wall_seconds'],'peak_train_allocated_gib':max(prefix['peak_allocated_bytes'],r['peak_allocated_bytes'])/2**30,'final_points':r['points'],'input_rgb':inp['summary'],'heldout_rgb_full':held['summary'],'source_query_entity_fractions':x['query_source_entity_fractions'],'query_alpha':x['query_source_alpha'],'sparse_cuda_check':x['sparse_cuda_check'],'evaluation_seconds':x['wall_seconds']}
            surface=ev/('surface_evaluation' if dev=='dev1' else 'surface_reference')/'metrics.json'
            s=read(surface);assert s['status']=='completed';row['surface_proxy']=s['summary'];row['surface_evaluation_seconds']=s['wall_seconds']
            if dev=='dev1':
                d=ev/'reference_evaluation/three_dimensional';row['fixed_queries_cm']={}
                for name in ['hand','object']:
                    v=csvrow(d/'summaries.csv',group='entity:'+name,visibility='all');row['fixed_queries_cm'][name]={'mean':100*float(v['mean_epe_m']),'coverage':float(v['coverage'])}
                    v=csvrow(d/'displacement_summaries.csv',group='entity:'+name,visibility='all');row['fixed_queries_cm'][name]['displacement_from_first_mean']=100*float(v['mean_epe_m'])
                v=csvrow(d/'relative_pair_summaries.csv',pair_id='all_fixed_pairs');row['fixed_queries_cm']['hand_object_relative']={'mean':100*float(v['mean_epe_m']),'coverage':float(v['coverage'])}
            summary['rows'].append(row);pair.append(r)
        for key in ['initialization_sha256','resume','data_identity','training_code_identity','frame_schedule_sha256']:
            assert pair[0][key]==pair[1][key],(dev,key)
    out=E/'summary';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['| 序列 | 分支 | 手套查询 cm | 箱体查询 cm | 手物相对 cm | 人体表面代理 cm | 物体表面代理 cm | 全程训练 s |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary['rows']:
        q=r.get('fixed_queries_cm',{});s=r.get('surface_proxy',{})
        value=lambda x:'—' if x is None else f'{x:.3f}'
        vals=[q.get('hand',{}).get('mean'),q.get('object',{}).get('mean'),q.get('hand_object_relative',{}).get('mean'),s.get('human',{}).get('symmetric_proxy_mean_cm'),s.get('object',{}).get('symmetric_proxy_mean_cm'),r['equivalent_full_train_seconds']]
        lines.append('| '+r['dev']+' | '+r['branch']+' | '+' | '.join(value(v) for v in vals)+' |')
    (out/'table.md').write_text('\n'.join(lines)+'\n')
    # Input view comparisons at five uniformly selected indices; no result-driven selection.
    for dev in ['dev1','dev2']:
        rows=[r for r in summary['rows'] if r['dev']==dev];ex=[Path(r['evaluation']) for r in rows];m=read(Path(read(ex[0]/'export_manifest.json')['input_manifest']));n=len(m['frame_paths']);panels=[]
        for t in np.linspace(0,n-1,5).astype(int):
            pics=[cv2.imread(m['frame_paths'][t])]+[cv2.imread(str(e/'rgb'/f'{t:05d}.png')) for e in ex]
            for pic,label in zip(pics,['Input','S0','S1']):cv2.putText(pic,f'{label} | f{t} | {m["timestamp_seconds"][t]:.3f}s',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
            panels.append(np.concatenate(pics,1))
        cv2.imwrite(str(out/f'{dev}_input_S0_S1.jpg'),np.concatenate(panels,0))
    print('\n'.join(lines))

if __name__=='__main__':main()
