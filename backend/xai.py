import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import ATTENTION_BUCKETS, GRADCAM_LAYER


def gradcam(net: nn.Module, x: torch.Tensor) -> tuple[list[float], int, int]:
    """Grad-CAM of the spoof logit at GRADCAM_LAYER, resized to the [N_MELS, T] input grid and scaled to [0, 1]."""
    layer = dict(net.named_modules())[GRADCAM_LAYER]
    captured = {}

    def keep(_module, _inputs, output):
        captured["act"] = output
        output.register_hook(lambda g: captured.update(grad=g))

    handle = layer.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            net.zero_grad(set_to_none=True)
            net(x.detach()).sum().backward()
    finally:
        handle.remove()
        net.zero_grad(set_to_none=True)
    weights = captured["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * captured["act"]).sum(dim=1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    lo, hi = cam.min(), cam.max()
    cam = (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)
    return cam.astype(np.float32).flatten().tolist(), cam.shape[0], cam.shape[1]


def attention_buckets(attn: torch.Tensor, buckets: int = ATTENTION_BUCKETS) -> tuple[list[float], int]:
    """Frame attention resampled to a fixed number of buckets and scaled to [0, 1]."""
    w = attn.float().numpy()
    n = w.shape[0]
    b = np.interp(np.linspace(0, n - 1, buckets), np.arange(n), w)
    lo, hi = b.min(), b.max()
    b = (b - lo) / (hi - lo) if hi > lo else np.full(buckets, 0.5)
    return b.tolist(), n
