"""V10 Q/Qr image objective with the original full-image SSIM11 formula.

The Gaussian window, zero padding and population covariance reproduce
4DGaussians/utils/loss_utils.py. Region averaging is applied to that full RGB
map; RGB is never blackened before SSIM. This is a training objective, separate
from the project's SSIM7 evaluation metric.
"""
from math import exp

import torch
import torch.nn.functional as F


def ssim11_map(pred, target):
    """Return [B,C,H,W], or [C,H,W] for one image, without scalar reduction."""
    if pred.shape != target.shape or pred.ndim not in (3, 4):
        raise ValueError('SSIM inputs must have identical CHW/BCHW shapes')
    single = pred.ndim == 3
    x, y = (pred.unsqueeze(0), target.unsqueeze(0)) if single else (pred, target)
    # Construct in float32 just like the original helper, then type_as(img).
    g = torch.tensor([exp(-(i - 5) ** 2 / (2 * 1.5 ** 2)) for i in range(11)])
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).float()[None, None]
    window = window.expand(x.shape[1], 1, 11, 11).contiguous().to(x)
    conv = lambda v: F.conv2d(v, window, padding=5, groups=x.shape[1])
    mx, my = conv(x), conv(y)
    vx, vy, vxy = conv(x * x) - mx.square(), conv(y * y) - my.square(), conv(x * y) - mx * my
    score = ((2 * mx * my + 0.01 ** 2) * (2 * vxy + 0.03 ** 2)) / (
        (mx.square() + my.square() + 0.01 ** 2) * (vx + vy + 0.03 ** 2))
    return score[0] if single else score


def _foreground(mask, batch_shape, device):
    if mask is None:
        raise ValueError('Qr requires the original training foreground mask')
    m = torch.as_tensor(mask, device=device)
    if m.ndim == 2:
        m = m[None, None]
    elif m.ndim == 3:
        m = m[:, None]
    if m.ndim != 4 or m.shape[1] != 1 or (m.shape[0], *m.shape[-2:]) != (batch_shape[0], *batch_shape[-2:]):
        raise ValueError('Foreground mask must be HW, BHW or B1HW with matching images')
    # Already-binary masks are the thresholded cache. Raw masks use >=128.
    if m.dtype == torch.bool:
        return m
    if torch.all((m == 0) | (m == 1)):
        return m > 0
    return m >= 128


def reconstruction_loss(pred, target, mask=None, regional=False, return_terms=False):
    """Mean of per-image Q or Qr, including Qfull fallback for empty masks.

    Raw RGB is used (no clip). Bool/0-1 masks are accepted as an already
    thresholded cache; uint8/raw mask values use the original >=128 threshold.
    """
    if pred.shape != target.shape or pred.ndim not in (3, 4) or pred.shape[-3] != 3:
        raise ValueError('Q expects matching RGB CHW/BCHW inputs')
    x, y = (pred[None], target[None]) if pred.ndim == 3 else (pred, target)
    l1 = (x - y).abs()
    dissimilarity = 1 - ssim11_map(x, y)
    l1_full, dssim_full = l1.mean((1, 2, 3)), dissimilarity.mean((1, 2, 3))
    q_full = 0.8 * l1_full + 0.2 * dssim_full
    q_fg = q_full
    counts = x.new_zeros((len(x),))
    if regional:
        fg = _foreground(mask, x.shape, x.device).to(x.dtype)
        counts = fg.sum((1, 2, 3))
        denominator = (counts * 3).clamp_min(1)
        l1_fg = (l1 * fg).sum((1, 2, 3)) / denominator
        dssim_fg = (dissimilarity * fg).sum((1, 2, 3)) / denominator
        q_fg = torch.where(counts > 0, 0.8 * l1_fg + 0.2 * dssim_fg, q_full)
    q = (0.9 * q_full + 0.1 * q_fg) if regional else q_full
    value = q.mean()
    if return_terms:
        return value, {'q_full': q_full.detach(), 'q_fg': q_fg.detach(),
                       'l1_full': l1_full.detach(), 'dssim11_full': dssim_full.detach(),
                       'foreground_pixels': counts.detach()}
    return value
