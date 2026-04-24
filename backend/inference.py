#inference.py: forward-pass inference for both ResNet18 and Wav2Vec2.
#architecture definitions are exact copies of the training notebooks so that load_state_dict() works without remapping keys.
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import Wav2Vec2Model

from config import (
    DEVICE,
    RESNET_CKPT,
    W2V_CKPT,
    WAV2VEC2_BASE,
    RESNET_DEFAULT_THRESHOLD,
    W2V_DEFAULT_THRESHOLD,
)
logger = logging.getLogger(__name__)

#RESNET18 architecture
def _build_resnet18() -> nn.Module:
    #Frozen:  stem, layer1. Trained: layer2, layer3, layer4, fc
    #FC head: BN(512) -> Dropout(0.3) -> Linear(512, 1)
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in name:
            param.requires_grad = False

    in_feat = model.fc.in_features   # 512
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_feat),
        nn.Dropout(0.3),
        nn.Linear(in_feat, 1),
    )
    return model

#W2V architecture
class W2VModel(nn.Module):
    #Wav2Vec2 with all layers frozen except encoder.layers.10 and encoder.layers.11, soft attention pooling over the time dimension, binary classifier head
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(WAV2VEC2_BASE)

        for name, param in self.encoder.named_parameters():
            if "encoder.layers.10" not in name and "encoder.layers.11" not in name:
                param.requires_grad = False

        self.attention = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        #per-sample z-score normalisation
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (
            wav.std(dim=-1, keepdim=True) + 1e-7
        )
        hidden = self.encoder(wav).last_hidden_state          # [B, T, 768]
        attn   = torch.softmax(self.attention(hidden), dim=1) # [B, T, 1]
        pooled = torch.sum(hidden * attn, dim=1)              # [B, 768]
        return self.classifier(pooled)                        # [B, 1]

    def forward_with_attention(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (
            wav.std(dim=-1, keepdim=True) + 1e-7
        )
        hidden = self.encoder(wav).last_hidden_state
        attn   = torch.softmax(self.attention(hidden), dim=1)
        pooled = torch.sum(hidden * attn, dim=1)
        logits = self.classifier(pooled)
        return logits, attn  # [B,1], [B,T,1]

#checkpoint loading helpers
@dataclass
class ResNetCheckpoint:
    model: nn.Module
    threshold: float
    mel_mean: float
    mel_std: float
    delta_mean: float
    delta_std: float
    delta2_mean: float
    delta2_std: float

@dataclass
class W2VCheckpoint:
    model: W2VModel
    threshold: float

def load_resnet(ckpt_path: str = RESNET_CKPT) -> ResNetCheckpoint:
    logger.info("Loading ResNet18 checkpoint from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = _build_resnet18()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    threshold = float(ckpt.get("recall_threshold", RESNET_DEFAULT_THRESHOLD))
    logger.info(
        "ResNet18 loaded — epoch=%s, dev EER=%.4f, threshold=%.4f",
        ckpt.get("epoch"),
        ckpt.get("eer", float("nan")),
        threshold,
    )

    return ResNetCheckpoint(
        model=model,
        threshold=threshold,
        mel_mean=float(ckpt["mel_mean"]),
        mel_std=float(ckpt["mel_std"]),
        delta_mean=float(ckpt["delta_mean"]),
        delta_std=float(ckpt["delta_std"]),
        delta2_mean=float(ckpt["delta2_mean"]),
        delta2_std=float(ckpt["delta2_std"]),
    )


def load_w2v(ckpt_path: str = W2V_CKPT) -> W2VCheckpoint:
    logger.info("Loading Wav2Vec2 checkpoint from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = W2VModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    threshold = float(ckpt.get("eer_threshold", W2V_DEFAULT_THRESHOLD))
    logger.info(
        "Wav2Vec2 loaded — epoch=%s, dev EER=%.4f, threshold=%.4f",
        ckpt.get("epoch"),
        ckpt.get("eer", float("nan")),
        threshold,
    )

    return W2VCheckpoint(model=model, threshold=threshold)

#inference helpers
def predict_resnet(
    model: nn.Module,
    features: torch.Tensor,   
    threshold: float,
) -> Dict:
    x = features.unsqueeze(0).to(DEVICE)   
    with torch.no_grad():
        logit = model(x)                 
        score = torch.sigmoid(logit).item()

    prediction = "AI" if score > threshold else "Human"
    return {"prediction": prediction, "score": round(score, 4), "threshold": threshold}


def predict_w2v(
    model: W2VModel,
    waveform: torch.Tensor,   
    threshold: float,
) -> Tuple[Dict, torch.Tensor]:
    x = waveform.unsqueeze(0).to(DEVICE) 
    with torch.no_grad():
        logits, attn = model.forward_with_attention(x)
        score = torch.sigmoid(logits).item()

    prediction = "AI" if score > threshold else "Human"
    result = {"prediction": prediction, "score": round(score, 4), "threshold": threshold}
    return result, attn.squeeze(0) 