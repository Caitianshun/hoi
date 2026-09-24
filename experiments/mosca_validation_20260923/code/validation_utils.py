"""Helpers for reversible input-only coordinate normalization."""
import hashlib
import torch

def tensor_state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        h.update(key.encode())
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
            h.update(value.numpy().tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()

def expose_metric_gaussians(static, dynamic, scale):
    """Convert final Gaussian tuples, leaving the learned normalized model intact.

    Cameras passed to the renderer remain the original supplied metric cameras.
    Each Gaussian standard deviation is divided by s (covariance thus by s²).
    Rotations, appearance and opacity do not change. No reference is read.
    """
    def convert(params):
        mean, frame, size, opacity, sh = params
        return mean / scale, frame, size / scale, opacity, sh
    static_forward = static.forward
    dynamic_forward = dynamic.forward
    static.forward = lambda *a, **kw: convert(static_forward(*a, **kw))
    dynamic.forward = lambda *a, **kw: convert(dynamic_forward(*a, **kw))
