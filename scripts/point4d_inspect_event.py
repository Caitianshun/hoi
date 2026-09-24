#!/usr/bin/env python3
"""CPU-only integrity and projection diagnostics; does not load evaluation GT."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", required=True, type=Path)
parser.add_argument("--times", nargs=9, type=float, default=[18.03,19.5,20.3,20.63,20.9,21.06,21.3,25.0,30.99])
args = parser.parse_args()
p = args.run
d = json.loads((p / "run.json").read_text())
z = np.load(p / "predictions.npz", allow_pickle=False)
uv = z["projected_uv_original"]
err = np.linalg.norm(uv[0] - z["query_uv_original"], axis=-1)
R, t, K = z["geometry_cam_R"], z["geometry_cam_t"], z["geometry_intrinsics"]
angles = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
times = z["timestamps_seconds"]
indices = [int(np.argmin(abs(times - value))) for value in args.times]
summary = {
    "finite": d["all_finite"], "query_ids": z["query_ids"].tolist(),
    "confidence_quantiles": np.quantile(z["confidence"], [0,.1,.5,.9,1]).tolist(),
    "first_frame_reprojection_error_px_quantiles": np.quantile(err, [0,.5,.9,1]).tolist(),
    "first_six_first_frame_reprojection_error_px": err[:6].tolist(),
    "predicted_camera_translation_norm_max_model_units": float(np.linalg.norm(t,axis=-1).max()),
    "predicted_camera_rotation_angle_from_initial_max_deg": float(angles.max()),
    "predicted_focal_x_range": [float(K[:,0,0].min()),float(K[:,0,0].max())],
    "figure_frame_indices": indices,
    "interpretation": "Internal consistency diagnostics only; no withheld GT loaded, confidence is not visibility, model units are not certified metres."
}
(p / "integrity_diagnostics.json").write_text(json.dumps(summary, indent=2))
colors = [(0,0,255),(0,150,255),(0,230,230),(0,200,0),(255,170,0),(255,0,255)]
tiles = []
for i in indices:
    image = cv2.imread(d["frames"][i]["path"])
    if image is None:
        raise ValueError("Inspection expects explicit RGB frame inputs, not an unextracted video")
    for q, color in enumerate(colors):
        x, y = uv[i,q]
        if np.isfinite([x,y]).all() and 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            xy = (int(round(x)),int(round(y)))
            cv2.circle(image,xy,5,color,2)
            cv2.putText(image,str(q),(xy[0]+7,xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    cv2.rectangle(image,(0,0),(image.shape[1],45),(0,0,0),-1)
    cv2.putText(image,f"frame {i}, {times[i]:.3f}s | Point4D projection only",(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image,"Not visibility or ground truth; first six fixed query IDs",(8,36),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    tiles.append(image)
sheet = np.concatenate([np.concatenate(tiles[j:j+3],axis=1) for j in range(0,9,3)],axis=0)
cv2.imwrite(str(p / "first_six_projection_diagnostic.png"),sheet)
print(json.dumps({"status":"inspected", "run":str(p),"figure_frame_indices":indices}))
