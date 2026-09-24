"""Serial six-frame/four-config UniDepth input-only frozen predictions."""
from pathlib import Path
import hashlib,json,os,sys,time,traceback
import cv2
import numpy as np
import torch
ROOT=Path('/home/cai_tianshun/Project/HOI');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'third_party/UniDepth'))
from unidepth.models import UniDepthV2
from unidepth.models.unidepthv2.unidepthv2 import get_paddings,get_resize_factor
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quantile(a):return dict(zip(['min','p10','median','p90','max'],np.quantile(a,[0,.1,.5,.9,1]).tolist()))
def main():
 begin=time.perf_counter();plan=json.loads((OUT/'plan.json').read_text());assert len(plan['configs'])<=4
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Root allocated physical GPU0 only'
 run=dict(status='running',pid=os.getpid(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],plan_sha256=sha(OUT/'plan.json'),script_sha256=sha(__file__),configs=[],reference_used=False)
 record=OUT/'prediction_run.json'
 def save():record.write_text(json.dumps(run,indent=2)+'\n')
 save()
 try:
  assert sha(plan['checkpoint'])==plan['checkpoint_sha256']
  torch.manual_seed(plan['seed']);np.random.seed(plan['seed']);torch.cuda.set_device(0)
  run['gpu']=torch.cuda.get_device_name(0);run['torch_version']=torch.__version__
  cp=ROOT/'third_party/UniDepth/configs/config_v2_vitl14.json';cfg=json.loads(cp.read_text());run['model_config_sha256']=sha(cp)
  load=time.perf_counter();model=UniDepthV2(cfg);model.load_state_dict(torch.load(plan['checkpoint'],map_location='cpu',weights_only=True),strict=True);model.cuda().eval();torch.cuda.synchronize()
  run['model_load_seconds']=time.perf_counter()-load
  network_shapes=[]
  hook=model.pixel_encoder.register_forward_pre_hook(lambda module,args:network_shapes.append(list(args[0].shape)))
  for c in plan['configs']:
   stage=time.perf_counter();m=json.loads(Path(c['input_manifest']).read_text());assert sha(c['input_manifest'])==c['input_manifest_sha256']
   assert m['role']=='input_only' and len(m['frames'])==6
   out=OUT/c['name'];out.mkdir(exist_ok=True)
   assert not (out/'frozen_prediction_manifest.json').exists(),'Do not overwrite frozen predictions'
   model.resolution_level=c['resolution_level'];K=torch.tensor(m['K'],dtype=torch.float32,device='cuda')[None];Koriginal=K.clone()
   torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();rows=[];total_infer=0
   for f in m['frames']:
    assert sha(f['path'])==f['sha256'];rgb=cv2.cvtColor(cv2.imread(f['path']),cv2.COLOR_BGR2RGB)
    x=torch.from_numpy(rgb).permute(2,0,1).cuda();inputK=K.clone()
    torch.cuda.synchronize();tic=time.perf_counter()
    with torch.inference_mode():pred=model.infer(x,camera=inputK)
    torch.cuda.synchronize();seconds=time.perf_counter()-tic;total_infer+=seconds
    dep=pred['depth'][0,0].float().cpu().numpy();confidence=pred['confidence'][0,0].float().cpu().numpy()
    assert dep.shape==rgb.shape[:2] and confidence.shape==dep.shape
    assert np.isfinite(dep).all() and (dep>0).all() and np.isfinite(confidence).all()
    assert torch.equal(K,Koriginal),'Shared K mutated despite per-frame clone'
    dst=out/f'{f["source_baseline_index"]:05d}.npz'
    np.savez_compressed(dst,dep=dep,confidence_raw=confidence,K_input=K[0].cpu().numpy(),K_returned=pred['intrinsics'][0].float().cpu().numpy(),
      source_baseline_index=f['source_baseline_index'],timestamp_seconds=f['timestamp_seconds'],
      role=np.array('frozen_input_only_prediction_no_reference_scale_alignment'),confidence_semantics=np.array('error_related_raw_value_not_probability_or_cross_image_calibrated'))
    row=dict(source_baseline_index=f['source_baseline_index'],timestamp_seconds=f['timestamp_seconds'],input_path=f['path'],input_sha256=f['sha256'],
      prediction_path=str(dst),prediction_sha256=sha(dst),inference_seconds=seconds,output_shape=list(dep.shape),encoder_input_shape=network_shapes[-1],
      depth_quantiles_m=quantile(dep),confidence_raw_quantiles=quantile(confidence),
      supplied_K_clone_max_mutation=float((inputK-Koriginal).abs().max()),shared_K_max_change=float((K-Koriginal).abs().max()))
    rows.append(row);print(c['name'],f['source_baseline_index'],'shape',network_shapes[-1],'seconds',round(seconds,3),flush=True)
    del pred,x,inputK
   item=dict(**c,status='completed',frames=rows,inference_seconds=total_infer,wall_seconds=time.perf_counter()-stage,
     peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),reference_read=False)
   freeze=out/'frozen_prediction_manifest.json';freeze.write_text(json.dumps(item,indent=2)+'\n')
   run['configs'].append(dict(name=c['name'],manifest=str(freeze),sha256=sha(freeze),**{k:item[k]for k in ['inference_seconds','wall_seconds','peak_allocated_bytes','peak_reserved_bytes']}));save()
  hook.remove();run['status']='completed';run['all_predictions_frozen_before_evaluation']=True
 except BaseException as exc:
  run.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
 finally:
  run['wall_seconds']=time.perf_counter()-begin;save()
if __name__=='__main__':main()
