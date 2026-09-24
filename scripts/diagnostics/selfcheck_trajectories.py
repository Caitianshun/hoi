#!/usr/bin/env python3
"""Small CPU correctness checks with known answers, not benchmark results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from evaluate_trajectories import (Trajectories, automatic_events, evaluate, load_bundle,
                                   main as evaluate_main, validate_protocol)
from mesh_query_reference import intersect_mesh, camera_rays, project_world
from evaluate_cotracker_reference import sample_depth, lift_visible_to_world


def fixture(pred, ref, valid=None, visibility=None, entity=None, times=None):
    pred, ref = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    t, n, _ = pred.shape
    return Trajectories(pred, ref, np.ones((t,n), bool) if valid is None else np.asarray(valid, bool),
                        np.isfinite(pred).all(axis=-1), np.full((t,n), -1, np.int8) if visibility is None else np.asarray(visibility),
                        np.arange(t, dtype=float) if times is None else np.asarray(times, float),
                        np.array(["object"]*n) if entity is None else np.asarray(entity), "fixture_world", "m")


class KnownAnswerChecks(unittest.TestCase):
    def test_translation_is_not_aligned_away(self):
        ref = np.zeros((4,2,3)); pred = ref + [0.03,0.04,0.0]
        result, _ = evaluate(fixture(pred,ref), [], [0.02,0.06],0.02,2)
        self.assertAlmostEqual(result["summaries"][0]["mean_epe_m"],0.05)
        self.assertEqual(result["summaries"][0]["success_at_0.02m"],0)
        self.assertAlmostEqual(result["displacement_summaries"][0]["mean_epe_m"],0.)
        self.assertEqual(result["displacement_summaries"][0]["expected_samples"],6)

    def test_scale_error_and_irregular_time_remain(self):
        ref = np.zeros((3,1,3)); ref[:,0,0] = [0,1,3]
        result,_ = evaluate(fixture(2*ref,ref,times=[0,0.5,1.5]),[],[0.02],0.02,1)
        self.assertAlmostEqual(result["summaries"][0]["mean_epe_m"],4/3)
        self.assertAlmostEqual(result["velocity"]["mean_velocity_error_m_per_s"],2.0)

    def test_missing_prediction_counts_as_failure(self):
        ref = np.zeros((3,1,3)); ref[2]=np.nan
        pred = np.ones((3,1,3))*[0.1,0,0]; pred[1]=np.nan
        result,_ = evaluate(fixture(pred,ref,valid=[[1],[1],[0]]),[],[0.2],0.02,1)
        s=result["summaries"][0]
        self.assertEqual(s["expected_samples"],2)
        self.assertEqual(s["missing_predictions"],1)
        self.assertAlmostEqual(s["mean_epe_m"],0.1)
        self.assertEqual(s["success_at_0.2m"],0.5)

    def test_visibility_partition_event_and_recovery(self):
        ref=np.zeros((8,1,3)); pred=ref.copy();pred[1:5,0,0]=0.1
        data=fixture(pred,ref,visibility=np.array([1,1,0,0,1,1,1,1])[:,None],times=np.arange(8)*0.1)
        events=automatic_events(data,2,4)
        result,_=evaluate(data,events,[0.02],0.02,2)
        self.assertEqual(events[0]["occlusion_start"],2)
        self.assertEqual(events[0]["occlusion_end"],4)
        self.assertEqual(result["recovery"][0]["recovery_frame"],5)
        self.assertAlmostEqual(result["recovery"][0]["recovery_time_s"],0.1)
        self.assertEqual([r["expected_samples"] for r in result["events"]],[2,2,4])

    def test_unknown_visibility_is_not_visible(self):
        data=fixture(np.zeros((3,1,3)),np.zeros((3,1,3)))
        self.assertEqual(automatic_events(data,1,2),[])
        result,_=evaluate(data,[],[0.02],0.02,1)
        self.assertEqual(result["summaries"][1]["expected_samples"],0)
        self.assertEqual(result["summaries"][3]["expected_samples"],3)
        self.assertEqual(result["identity"]["status"],"unavailable")

    def test_identity_and_relative_metrics(self):
        ref=np.zeros((3,2,3));ref[:,1,0]=1
        pred=ref.copy();pred[:,1,0]+=0.1
        data=fixture(pred,ref,entity=["hand","object"])
        data.pred_identity=np.array([["hand","cup"],["hand","hand"],["hand",""]])
        data.reference_identity=np.array([["hand","cup"]]*3)
        result,_=evaluate(data,[],[0.02],0.02,1)
        self.assertEqual(result["identity"]["switches_across_adjacent_known_frames"],1)
        self.assertEqual(result["identity"]["accuracy_missing_as_failure"],4/6)
        self.assertAlmostEqual(result["hand_object_relative"]["mean_centroid_relative_position_error_m"],0.1)

    def test_units_and_coordinate_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"data.npz"
            base=dict(predicted=np.ones((2,1,3))*[10,0,0],reference=np.zeros((2,1,3)),
                      valid_mask=np.ones((2,1),bool),frame_times=np.array([0.,0.1]),
                      entity=np.array(["object"]),units=np.array("mm"),coordinate_frame=np.array("world"))
            np.savez(p,**base);data=load_bundle(p)
            self.assertAlmostEqual(data.predicted[0,0,0],0.01)
            with self.assertRaisesRegex(ValueError,"conflicts"):
                load_bundle(p,"m")
            np.savez(p,**base,predicted_coordinate_frame=np.array("camera"))
            with self.assertRaisesRegex(ValueError,"differs"):
                load_bundle(p)
            base["valid_mask"]=np.array([[1],[2]])
            np.savez(p,**base)
            with self.assertRaisesRegex(ValueError,"boolean"):
                load_bundle(p)

    def test_declared_leakage_is_rejected(self):
        for protocol in [dict(reference_used_for_training=True),dict(alignment="per_frame_icp"),
                         dict(training_assets=["fit"],reference_assets=["fit"]),
                         dict(alignment="predeclared_global_transform",global_transform_provenance="test points")]:
            with self.assertRaises(ValueError):validate_protocol(protocol)
        validate_protocol(dict(reference_used_for_training=False,reference_role="registered_reference"))

    def test_mesh_ray_chooses_front_surface_and_preserves_material_weights(self):
        front=np.array([[-1,-1,2],[1,-1,2],[0,1,2]],float)
        vertices=np.concatenate([front,front+[0,0,2]])
        faces=np.array([[3,4,5],[0,1,2]])
        face,weights,distance=intersect_mesh(np.zeros(3),np.array([0,0,1]),vertices,faces)
        self.assertEqual(face,1)
        self.assertAlmostEqual(distance,2.)
        np.testing.assert_allclose(weights,[0.25,0.25,0.5])
        moved=vertices.copy();moved[:3]+=[0.1,0.2,0.3]
        np.testing.assert_allclose(weights @ moved[faces[face]], [0.1,0.2,2.3])
        self.assertIsNone(intersect_mesh(np.zeros(3),np.array([0,0,-1]),vertices,faces))

    def test_camera_world_direction_and_inverse_projection(self):
        k=np.array([[100,0,50],[0,100,40],[0,0,1.]])
        r=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
        origin=np.array([3,2,1.]); uv=np.array([[50.,40.],[60.,45.]])
        rays=camera_rays(uv,k,np.zeros(8),r)
        np.testing.assert_allclose(rays[0],[1,0,0],atol=1e-12)
        reprojected,depth=project_world(origin+2*rays,k,np.zeros(8),r,origin)
        np.testing.assert_allclose(reprojected,uv,atol=1e-10)
        self.assertTrue((depth>0).all())

    def test_depth_bilinear_and_invisible_lift(self):
        depth=np.array([[1.,2.,3.],[3.,4.,5.],[5.,6.,7.]])
        uv=np.array([[.5,.25],[1.,1.]])
        self.assertAlmostEqual(sample_depth(depth,uv)[0],2.)
        c=np.eye(4);c[:3,3]=[3,2,1]
        xyz=lift_visible_to_world(depth,uv,np.array([True,False]),np.eye(3),c)
        np.testing.assert_allclose(xyz[0],[4,2.5,3])
        self.assertTrue(np.isnan(xyz[1]).all())
        self.assertTrue(np.isnan(sample_depth(depth,np.array([[-1.,0.]]))[0]))

    def test_displacement_missing_anchor_counts_failure(self):
        ref=np.zeros((3,2,3));pred=ref.copy();pred[0,1]=np.nan
        result,_=evaluate(fixture(pred,ref),[],[.02],.02,1)
        summary=result["displacement_summaries"][0]
        self.assertEqual(summary["expected_samples"],4)
        self.assertEqual(summary["predicted_samples"],2)
        self.assertEqual(summary["success_at_0.02m"],.5)

    def test_relative_vector_pairs_preserve_identity_and_missing(self):
        ref=np.zeros((3,3,3));ref[:,1,0]=1;ref[:,2,1]=1
        pred=ref+[1,2,3];pred[:,1,0]+=.1;pred[1,2]=np.nan
        data=fixture(pred,ref,entity=["hand","object","object"])
        data.query_id=np.array(["glove","box_a","box_b"])
        result,_=evaluate(data,[],[.02],.02,1)
        rows=result["relative_pair_summaries"]
        self.assertEqual(rows[0]["expected_samples"],6)
        self.assertEqual(rows[0]["predicted_samples"],5)
        self.assertAlmostEqual(rows[0]["mean_epe_m"],.06)
        self.assertEqual(rows[1]["pair_id"],"glove__minus__box_a")
        self.assertAlmostEqual(rows[2]["mean_epe_m"],0.)


def write_demo(output):
    output.mkdir(parents=True,exist_ok=True)
    t=50;times=np.arange(t)/10
    ref=np.zeros((t,4,3))
    ref[:,:,0]=times[:,None]*0.03+np.array([0,0.02,0.15,0.18])[None,:]
    ref[:,:,1]=[0,0,0.03,0.03]
    pred=ref.copy();pred[15:32,2:,0]+=np.linspace(0.005,0.08,17)[:,None]
    pred[32:38,2:,0]+=np.linspace(0.06,0.005,6)[:,None]
    pred[19:21,3,:]=np.nan
    visibility=np.ones((t,4),np.int8);visibility[15:32,2:]=0
    path=output/"synthetic_fixture.npz"
    np.savez(path,predicted=pred,reference=ref,valid_mask=np.ones((t,4),bool),visibility=visibility,
             frame_times=times,entity=np.array(["hand","hand","object","object"]),
             units=np.array("m"),coordinate_frame=np.array("synthetic_world"))
    protocol=output/"synthetic_protocol.json"
    protocol.write_text(json.dumps(dict(reference_role="synthetic_fixture",reference_used_for_training=False,
                                       visibility_source="synthetic_fixture",alignment="none",
                                       note="Known-answer CPU fixture, not a model or research result"),indent=2))
    evaluate_main(["--input",str(path),"--output",str(output/"rendered_check"),"--protocol",str(protocol),"--overwrite"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifacts",type=Path)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(KnownAnswerChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.artifacts:
        args.artifacts.mkdir(parents=True,exist_ok=True)
        (args.artifacts/"selfcheck.json").write_text(json.dumps({"tests":result.testsRun,"failures":len(result.failures),
            "errors":len(result.errors),"successful":result.wasSuccessful(),"kind":"CPU synthetic correctness checks"},indent=2)+"\n")
        if result.wasSuccessful():write_demo(args.artifacts)
    raise SystemExit(0 if result.wasSuccessful() else 1)
