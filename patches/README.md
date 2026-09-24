# Local changes to upstream code

The project keeps upstream repositories outside the GitHub code mirror. These
patches preserve tracked local code changes without uploading complete copies
of third-party repositories or experiment data. Apply each patch to the matching
upstream commit with `git apply`.

| Patch | Upstream checkout | Base commit |
| --- | --- | --- |
| `mosca_runtime.patch` | `third_party/MoSca` | `02404d2841b4e0dc2d9074860950bce5da579656` |
| `mosca_validation.patch` | `experiments/mosca_validation_20260923/code/MoSca` | `02404d2841b4e0dc2d9074860950bce5da579656` |
| `mosca_interface.patch` | `experiments/mosca_interface_validation_20260923/code/MoSca` | `02404d2841b4e0dc2d9074860950bce5da579656` |
| `gvhmr_validation.patch` | `experiments/gvhmr_validation_20260923/code/GVHMR` | `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` |
| `megapose_pose.patch` | `experiments/object_pose_refinement_20260924/run01/megapose/repo` | `f3b8e1247f133f3d098833a251b8f2d744c03e1f` |

The interface patch also includes the new `lib_moca/pixel_geometry.py`. Deleted
media and pruned upstream demo files are deliberately absent from the patches.
