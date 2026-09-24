import ast,json,hashlib
from pathlib import Path
import numpy as np,torch,smplx
from scipy.spatial.transform import Rotation
root=Path('/home/cai_tianshun/Project/HOI'); out=root/'experiments/gvhmr_validation_20260923/coordinate_checks_cpu.json'
p=root/'experiments/mosca_interface_validation_20260923/human_prior/prepared_input/input_manifest.json';d=json.loads(p.read_text()); K=np.asarray(d['K']); T=np.asarray(d['c2w']);Rc=T[:3,:3];tc=T[:3,3]
source=root/'third_party/GVHMR/hmr4d/utils/geo/hmr_cam.py';tree=ast.parse(source.read_text()); funcs=['compute_transl_full_cam','get_a_pred_cam','compute_bbox_info_bedlam','perspective_projection'];ns={'torch':torch}
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in funcs],type_ignores=[]),str(source),'exec'),ns)
bbx=torch.tensor([[282.,333.,216.],[258.,300.5,289.2]])
kt=torch.from_numpy(K).float().repeat(2,1,1);transl=torch.tensor([[0.1,0.3,2.4],[-.2,.1,2.8]])
pc=ns['get_a_pred_cam'](transl,bbx,kt);back=ns['compute_transl_full_cam'](pc,bbx,kt)
bbox_info=ns['compute_bbox_info_bedlam'](bbx,kt);assert bbox_info.shape==(2,3) and torch.isfinite(bbox_info).all()
points=torch.tensor([[[.1,.2,2.4]],[[.3,-.1,2.8]]]);uv=ns['perspective_projection'](points,kt);manual=points.numpy()[...,:2]/points.numpy()[...,[2]]*np.array([K[0,0],K[1,1]])+np.array([K[0,2],K[1,2]])
model=smplx.create('/home/cai_tianshun/Project/mml/smpl_model',model_type='smplx',gender='neutral',num_betas=10,num_pca_comps=12,flat_hand_mean=False,batch_size=2).eval()
bs=torch.tensor([[.2,-.1,0,0,0,0,0,0,0,0],[-.1,.15,0,0,0,0,0,0,0,0]],dtype=torch.float32);pose=torch.zeros((2,63));pose[0,54]=.2;pose[1,57]=-.3;ori=torch.tensor([[.1,.2,-.1],[-.2,.1,.15]],dtype=torch.float32)
with torch.no_grad():
 oc=model(betas=bs,body_pose=pose,global_orient=ori,transl=transl)
 vshape=model.v_template[None]+torch.einsum('bl,vcl->bvc',bs,model.shapedirs);J=torch.einsum('v,bvc->bc',model.J_regressor[0],vshape).numpy()
 ow=Rotation.from_matrix(Rc[None]@Rotation.from_rotvec(ori.numpy()).as_matrix()).as_rotvec(); tw=transl.numpy()@Rc.T+tc+J@(Rc-np.eye(3)).T
 owout=model(betas=bs,body_pose=pose,global_orient=torch.from_numpy(ow).float(),transl=torch.from_numpy(tw).float())
 expected=oc.vertices.numpy()@Rc.T+tc; naive=model(betas=bs,body_pose=pose,global_orient=torch.from_numpy(ow).float(),transl=torch.from_numpy(transl.numpy()@Rc.T+tc).float()).vertices.numpy()
report={'scope':'CPU synthetic interface acceptance only; no video inference, no reference meshes or GT used','input_manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'camera_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'actual_source_functions_executed':funcs,'K':K.tolist(),'focal_relative_fx_over_fy_minus_1':float(K[0,0]/K[1,1]-1),'same_3d_point_projection_only_square_focal_vs_K_max_vertical_full_height_px':float(abs(K[1,1]/K[0,0]-1)*max(K[1,2],479-K[1,2])),'translation_roundtrip_max_m':float((back-transl).abs().max()),'projection_manual_max_px':float(abs(uv.numpy()-manual).max()),'c2w_orthogonality_max':float(abs(Rc.T@Rc-np.eye(3)).max()),'c2w_det':float(np.linalg.det(Rc)),'synthetic_world_parameter_conversion_vertex_max_m':float(abs(expected-owout.vertices.numpy()).max()),'synthetic_naive_translation_only_vertex_max_m':float(abs(expected-naive).max()),'timestamp_gap_quantiles_s':np.quantile(np.diff(d['timestamp_seconds']),[0,.25,.5,.75,1]).tolist(),'smplx_output_joints':list(oc.joints.shape),'smplx_output_vertices':list(oc.vertices.shape)}
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
