"""Read-only CPU interpretation of the locked backend's saved buffer layout."""
from common import *
import argparse,torch,numpy as np
def summarize(a):
    a=np.asarray(a);f=np.isfinite(a)
    return dict(shape=list(a.shape),finite=int(f.sum()),nan=int(np.isnan(a).sum()),posinf=int(np.isposinf(a).sum()),neginf=int(np.isneginf(a).sum()),finite_quantiles=quantiles(a[f]))
def analyze(folder):
    folder=Path(folder);args=torch.load(folder/'raster_backward_last_inputs.pt',weights_only=False,map_location='cpu');bad=torch.load(folder/'first_bad_values.pt',weights_only=False,map_location='cpu')
    n=len(args[1]);buf=args[17].numpy();offset=0;fields={}
    for name,count,dtype,shape in [('depth',n,'<f4',(n,)),('clamped',n*3,'?',(n,3)),('radii',n,'<i4',(n,)),('means2D',n*2,'<f4',(n,2)),('cov3D',n*6,'<f4',(n,6)),('conic_opacity',n*4,'<f4',(n,4)),('rgb',n*3,'<f4',(n,3)),('tiles_touched',n,'<u4',(n,))]:
        offset=(offset+127)//128*128;size=np.dtype(dtype).itemsize*count
        fields[name]=np.frombuffer(buf[offset:offset+size],dtype=dtype).reshape(shape);offset+=size
    assert np.array_equal(fields['radii'],args[2].numpy()),'Validate geometry layout against separately returned radii'
    names=['screen','precomputed_color','opacity','xyz','covariance','SH','scale','rotation'];bad_rows={name:torch.nonzero(~torch.isfinite(t),as_tuple=False)[:,0].unique().tolist() for name,t in zip(names,bad)}
    ids=sorted(set(i for rows in bad_rows.values() for i in rows));visible=fields['radii']>0
    def safe(a):
        if np.isscalar(a):return float(a) if np.isfinite(a) else str(a)
        return [safe(x) for x in a]
    result=dict(buffer_schema='Locked GeometryState::fromChunk; 128-byte alignment; radii equality independently checked',points=n,candidate_visible=int(visible.sum()),bad_rows=bad_rows,all_bad_attribute_row_intersection=sorted(set.intersection(*[set(x) for x in bad_rows.values() if x])),upstream_RGB=summarize(args[12].numpy()),upstream_depth=summarize(args[13].numpy()),upstream_depth_all_zero=bool(torch.count_nonzero(args[13])==0),visible_geometry={name:summarize(v[visible]) for name,v in fields.items() if v.dtype.kind=='f'},bad_points={str(i):{name:safe(value[i]) for name,value in fields.items()} for i in ids})
    result['bad_input_geometry']={str(i):dict(xyz=safe(args[1][i].numpy()),scale=safe(args[4][i].numpy()),rotation=safe(args[5][i].numpy())) for i in ids}
    save_json(folder/'kernel_case_analysis.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');analyze(p.parse_args().folder)
