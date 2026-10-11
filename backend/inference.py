import json
import math
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import resnet18
from transformers import Wav2Vec2Config, Wav2Vec2Model

from config import DEVICE, FUSION_INPUTS

# isort: split
import audio_utils as U

REQUIRED_KEYS = {
    "resnet18": ("model_state_dict", "threshold_logit", "threshold_source", "feature_stats", "dev_eer", "epoch"),
    "wav2vec2": ("model_state_dict", "threshold_logit", "threshold_source", "encoder_config", "dev_eer", "epoch"),
}


class FP32Head(nn.Sequential):
    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return super().forward(x.float())


class W2VClassifier(nn.Module):
    def __init__(self, encoder: Wav2Vec2Model):
        super().__init__()
        self.encoder = encoder
        dim = encoder.config.hidden_size
        self.layer_logits = nn.Parameter(torch.zeros(encoder.config.num_hidden_layers + 1))
        self.attention = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(
            nn.Linear(dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )

    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(wave, output_hidden_states=True).hidden_states
        with torch.autocast(wave.device.type, enabled=False):
            stacked = torch.stack([h.float() for h in hidden], dim=0)
            mixed = (torch.softmax(self.layer_logits, dim=0)[:, None, None, None] * stacked).sum(dim=0)
            attn = torch.softmax(self.attention(mixed), dim=1)
            pooled = (attn * mixed).sum(dim=1)
            return self.classifier(pooled).squeeze(1), attn.squeeze(-1)


@dataclass
class LoadedModel:
    name: str
    net: nn.Module
    threshold_logit: float
    dev_eer: float
    epoch: int
    sha256: str
    features: U.LogMelDelta | None = None


def _read_checkpoint(path: str, name: str, expected_sha256: str | None) -> tuple[dict, str]:
    sha = U.file_sha256(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"{path}: sha256 {sha} does not match the evaluated checkpoint {expected_sha256}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("model") != name:
        raise ValueError(f"{path}: expected a {name} checkpoint, found {ckpt.get('model')!r}")
    missing = [k for k in REQUIRED_KEYS[name] if k not in ckpt]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    if ckpt["threshold_source"] != "dev_eer":
        raise ValueError(f"{path}: threshold does not come from dev")
    return ckpt, sha


def load_resnet(path: str, expected_sha256: str | None = None) -> LoadedModel:
    ckpt, sha = _read_checkpoint(path, "resnet18", expected_sha256)
    net = resnet18(weights=None)
    net.fc = FP32Head(nn.BatchNorm1d(512), nn.Dropout(0.3), nn.Linear(512, 1))
    net.load_state_dict(ckpt["model_state_dict"], strict=True)
    features = U.LogMelDelta(U.FeatureStats.from_dict(ckpt["feature_stats"]))
    return LoadedModel(
        "resnet18",
        net.to(DEVICE).eval(),
        float(ckpt["threshold_logit"]),
        float(ckpt["dev_eer"]),
        int(ckpt["epoch"]),
        sha,
        features.to(DEVICE).eval(),
    )


def load_w2v(path: str, expected_sha256: str | None = None) -> LoadedModel:
    ckpt, sha = _read_checkpoint(path, "wav2vec2", expected_sha256)
    cfg = {k: v for k, v in ckpt["encoder_config"].items() if k != "transformers_version"}
    net = W2VClassifier(Wav2Vec2Model(Wav2Vec2Config(**cfg)))
    net.load_state_dict(ckpt["model_state_dict"], strict=True)
    return LoadedModel(
        "wav2vec2",
        net.to(DEVICE).eval(),
        float(ckpt["threshold_logit"]),
        float(ckpt["dev_eer"]),
        int(ckpt["epoch"]),
        sha,
    )


def resnet_input(model: LoadedModel, raw: torch.Tensor) -> torch.Tensor:
    return model.features(raw.unsqueeze(0).to(DEVICE))


@torch.no_grad()
def score_resnet(model: LoadedModel, raw: torch.Tensor) -> float:
    return float(model.net(resnet_input(model, raw)).squeeze())


@torch.no_grad()
def score_w2v(model: LoadedModel, wave: torch.Tensor) -> tuple[float, torch.Tensor]:
    logit, attn = model.net(wave.unsqueeze(0).to(DEVICE))
    return float(logit.squeeze()), attn.squeeze(0).cpu()


def calibrated_score(logit: float, threshold_logit: float) -> float:
    """Logistic of the margin to the dev-EER threshold, so 0.5 is the decision boundary."""
    z = logit - threshold_logit
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def label(logit: float, threshold_logit: float) -> str:
    return "AI" if logit >= threshold_logit else "Human"


@dataclass
class Fusion:
    mean: tuple[float, float]
    scale: tuple[float, float]
    coef: tuple[float, float]
    intercept: float
    threshold_logit: float

    @classmethod
    def from_json(cls, path: str) -> "Fusion":
        with open(path) as fh:
            p = json.load(fh)
        if p.get("inputs") != FUSION_INPUTS or p.get("fitted_on") != "dev":
            raise ValueError(f"{path}: expected dev-fitted fusion of {FUSION_INPUTS}, found {p.get('inputs')}")
        return cls(
            tuple(p["scaler_mean"]),
            tuple(p["scaler_scale"]),
            tuple(p["coef"]),
            float(p["intercept"]),
            float(p["threshold"]),
        )

    def logit(self, resnet_logit: float, w2v_logit: float) -> float:
        z = [(v - m) / s for v, m, s in zip((resnet_logit, w2v_logit), self.mean, self.scale)]
        return float(self.coef[0] * z[0] + self.coef[1] * z[1] + self.intercept)
