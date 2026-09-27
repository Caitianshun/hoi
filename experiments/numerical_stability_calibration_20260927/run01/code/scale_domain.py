"""One declared numerical-domain intervention on final rendered log-scales."""
import math
def bound_log_scale(log_scale,scene_extent):
    assert scene_extent>0 and math.isfinite(scene_extent)
    return log_scale.clamp(max=math.log(scene_extent))
