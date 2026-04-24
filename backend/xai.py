#xai.py
'''
GradCAM: ResNet18 layer4 activation x gradient, resized to [N_MELS, T]
Attention → Wav2Vec2 soft-attention weights, bucketed to ATTENTION_BUCKETS'''

import logging
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE, GRADCAM_TARGET_LAYER, ATTENTION_BUCKETS

logger = logging.getLogger(__name__)

#gradcam
class GradCAM:

    def __init__(self, model: nn.Module, target_layer_name: str = GRADCAM_TARGET_LAYER):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:Optional[torch.Tensor] = None
        self._hooks: List = []

        #locating target sub-module by name
        target = dict(model.named_modules()).get(target_layer_name)
        if target is None:
            raise ValueError(
                f"Layer '{target_layer_name}' not found in model. "
                f"Available: {list(dict(model.named_modules()).keys())}"
            )
        self._hooks.append(
            target.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            target.register_full_backward_hook(self._save_gradients)
        )
    #hook callback
    def _save_activations(self, _module, _input, output: torch.Tensor) -> None:
        self._activations = output.detach()
    def _save_gradients(self, _module, _grad_input, grad_output: Tuple) -> None:
        self._gradients = grad_output[0].detach()

    #main api
    def compute(
        self,
        input_tensor: torch.Tensor,   
        target_h: int = 128,          
        target_w: Optional[int] = None,
    ) -> Tuple[List[float], int, int]:
        self.model.eval()

        x = input_tensor.to(DEVICE)
        x.requires_grad_(False)

        #gradients for backward pass
        with torch.enable_grad():
            x_grad = x.clone().detach().requires_grad_(True)
            self.model.zero_grad()
            logit  = self.model(x_grad)          # [1, 1]
            score  = torch.sigmoid(logit)
            logit.backward()

        if self._activations is None or self._gradients is None:
            logger.warning("GradCAM hooks did not fire — returning empty heatmap")
            return [], 0, 0

        # GAP over spatial dims->channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  
        cam= (weights * self._activations).sum(dim=1, keepdim=True) 
        cam= F.relu(cam)
        #output shape
        if target_w is None:
            #time-width from the mel spectrogram input
            _, _, orig_h, orig_w = x.shape
            target_w = orig_w

        #resize to mel spectrogram resolution
        cam = F.interpolate(cam, size=(target_h, target_w), mode="bilinear", align_corners=False)

        #normalise to [0, 1]
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        rows, cols = cam_np.shape
        values = cam_np.flatten().tolist()
        return values, rows, cols

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def compute_gradcam(
    model: nn.Module,
    features: torch.Tensor,   # [3, 128, T]
) -> Tuple[List[float], int, int]:
    #convenience wrapper creates a GradCAM instance, computes the heatmap, removes hooks, and returns (values, rows, cols).
    cam_engine = GradCAM(model, GRADCAM_TARGET_LAYER)
    try:
        x = features.unsqueeze(0).to(DEVICE)   # [1, 3, 128, T]
        target_h = features.shape[1]           # N_MELS = 128
        target_w = features.shape[2]           # time frames
        values, rows, cols = cam_engine.compute(x, target_h=target_h, target_w=target_w)
    except Exception:
        logger.exception("GradCAM computation failed")
        values, rows, cols = [], 0, 0
    finally:
        cam_engine.remove_hooks()

    return values, rows, cols

#attention weight extraction
def compute_attention_weights(
    attn: torch.Tensor,          #[T, 1] raw softmax attention from W2VModel
    num_buckets: int = ATTENTION_BUCKETS,
) -> Tuple[List[float], int]:

    weights = attn.squeeze(-1).cpu()   # [T]
    num_frames = weights.shape[0]

    if num_frames == 0:
        return [0.0] * num_buckets, 0

    #average-pool into buckets
    w_np = weights.numpy().astype(np.float32)

    #resize using linear interpolation
    indices = np.linspace(0, num_frames - 1, num_buckets)
    bucketed = np.interp(indices, np.arange(num_frames), w_np)

    #normalise to [0, 1]
    b_min, b_max = bucketed.min(), bucketed.max()
    if b_max > b_min:
        bucketed = (bucketed - b_min) / (b_max - b_min)
    else:
        bucketed = np.ones(num_buckets) * 0.5

    return bucketed.tolist(), num_frames