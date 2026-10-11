"""
Shared loading, features, metrics and score I/O for AudioInsight.

Every notebook and the backend import this module; none of this logic is
re-implemented elsewhere. Labels: bona fide = 0, spoof = 1. Models output a
spoof logit; metric code flips it once, in spoof_logit_to_cm_score, because
the official ASVspoof scorer expects higher = more bona fide.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

UTILS_VERSION = "1.0.1"

TARGET_SR = 16_000
MAX_SAMPLES = TARGET_SR * 4
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
N_FRAMES = MAX_SAMPLES // HOP_LENGTH + 1
NORM_EPS = 1e-6

LABEL_BONAFIDE = 0
LABEL_SPOOF = 1
CHANNELS = ("mel", "delta", "delta2")
SPLITS = ("train", "dev", "eval")

EXPECTED_COUNTS = {
    "train": {"bonafide": 2_580, "spoof": 22_800},
    "dev": {"bonafide": 2_548, "spoof": 22_296},
    "eval": {"bonafide": 7_355, "spoof": 63_882},
}
EXPECTED_ATTACKS = {
    "train": tuple(f"A{i:02d}" for i in range(1, 7)),
    "dev": tuple(f"A{i:02d}" for i in range(1, 7)),
    "eval": tuple(f"A{i:02d}" for i in range(7, 20)),
}
PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
AUDIO_DIRS = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}

TDCF_PSPOOF = 0.05
TDCF_COST_MODEL = {
    "Pspoof": TDCF_PSPOOF,
    "Ptar": (1 - TDCF_PSPOOF) * 0.99,
    "Pnon": (1 - TDCF_PSPOOF) * 0.01,
    "Cmiss_asv": 1,
    "Cfa_asv": 10,
    "Cmiss_cm": 1,
    "Cfa_cm": 10,
}


def utils_sha256() -> str:
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_cm_protocol(path: str, split: str) -> pd.DataFrame:
    """Lines: SPEAKER UTT_ID - SYSTEM_ID KEY. Raises unless the file matches the official release."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{path}:{line_no}: expected 5 fields, got {len(parts)}")
            speaker, utt_id, _, system_id, key = parts
            if key not in ("bonafide", "spoof"):
                raise ValueError(f"{path}:{line_no}: unknown key {key!r}")
            if (key == "bonafide") != (system_id == "-"):
                raise ValueError(f"{path}:{line_no}: key {key!r} with system {system_id!r}")
            rows.append({
                "utt_id": utt_id,
                "speaker": speaker,
                "attack": "bonafide" if key == "bonafide" else system_id,
                "key": key,
                "label": LABEL_BONAFIDE if key == "bonafide" else LABEL_SPOOF,
                "split": split,
            })
    df = pd.DataFrame(rows)
    validate_protocol(df, split)
    return df


def validate_protocol(df: pd.DataFrame, split: str) -> None:
    if df["utt_id"].duplicated().any():
        raise AssertionError(f"[{split}] duplicated utterance ids")
    counts = df["key"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS[split]:
        raise AssertionError(f"[{split}] counts {counts} != {EXPECTED_COUNTS[split]}")
    attacks = tuple(sorted(a for a in df["attack"].unique() if a != "bonafide"))
    if attacks != EXPECTED_ATTACKS[split]:
        raise AssertionError(f"[{split}] attacks {attacks} != {EXPECTED_ATTACKS[split]}")


def assert_disjoint_splits(frames: Dict[str, pd.DataFrame]) -> None:
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for col in ("utt_id", "speaker"):
                shared = set(frames[a][col]) & set(frames[b][col])
                if shared:
                    raise AssertionError(f"{len(shared)} {col} values shared by {a} and {b}")


def parse_asv_scores(path: str) -> Dict[str, np.ndarray]:
    """Organisers' ASV score file. Lines: SOURCE KEY SCORE."""
    data = np.genfromtxt(path, dtype=str)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns, got shape {data.shape}")
    keys = data[:, 1]
    unknown = set(np.unique(keys)) - {"target", "nontarget", "spoof"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    scores = data[:, 2].astype(np.float64)
    out = {k: scores[keys == k] for k in ("target", "nontarget", "spoof")}
    for k, v in out.items():
        if v.size == 0 or not np.isfinite(v).all():
            raise ValueError(f"{path}: missing or non-finite {k} scores")
    return out


def read_audio_int16(path: str) -> np.ndarray:
    info = sf.info(path)
    if info.samplerate != TARGET_SR or info.channels != 1 or info.subtype != "PCM_16":
        raise ValueError(f"{path}: {info.samplerate} Hz, {info.channels} ch, {info.subtype}; need 16 kHz mono PCM_16")
    x, _ = sf.read(path, dtype="int16", always_2d=False)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{path}: unexpected shape {x.shape}")
    return x


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    """Same scaling as torchaudio.load for 16-bit PCM."""
    if x.dtype != np.int16:
        raise TypeError(f"expected int16, got {x.dtype}")
    return x.astype(np.float32) / 32768.0


def fit_head(x: torch.Tensor, n: int = MAX_SAMPLES) -> torch.Tensor:
    """Keep the first n samples, zero-pad on the right."""
    t = x.shape[-1]
    return x[..., :n] if t >= n else torch.nn.functional.pad(x, (0, n - t))


def edge_silence(x: np.ndarray, frame: int = 400, hop: int = 160, rel_db: float = -40.0) -> Tuple[int, int]:
    """Leading/trailing samples below rel_db of the loudest frame. Analysis only."""
    xf = x.astype(np.float64)
    if xf.size < frame:
        return 0, 0
    n_frames = 1 + (xf.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    energy = (xf[idx] ** 2).mean(axis=1)
    if energy.max() <= 0:
        return int(xf.size), 0
    voiced = np.flatnonzero(10.0 * np.log10(np.maximum(energy, 1e-12) / energy.max()) > rel_db)
    return int(voiced[0] * hop), int(max(0, xf.size - (voiced[-1] * hop + frame)))


@dataclass
class FeatureStats:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    n_values: int
    source: str = "train"

    def to_dict(self) -> Dict:
        return {
            "mel_mean": self.mean[0], "mel_std": self.std[0],
            "delta_mean": self.mean[1], "delta_std": self.std[1],
            "delta2_mean": self.mean[2], "delta2_std": self.std[2],
            "n_values": self.n_values, "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureStats":
        if d.get("source") != "train":
            raise ValueError(f"feature stats must come from train, got {d.get('source')!r}")
        return cls(
            mean=(float(d["mel_mean"]), float(d["delta_mean"]), float(d["delta2_mean"])),
            std=(float(d["mel_std"]), float(d["delta_std"]), float(d["delta2_std"])),
            n_values=int(d["n_values"]),
        )


class LogMelDelta(nn.Module):
    """[B, MAX_SAMPLES] waveform -> [B, 3, N_MELS, N_FRAMES]; normalised when stats are given."""

    def __init__(self, stats: Optional[FeatureStats] = None) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        mean = torch.zeros(3) if stats is None else torch.tensor(stats.mean, dtype=torch.float32)
        std = torch.ones(3) if stats is None else torch.tensor(stats.std, dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))
        self.normalise = stats is not None

    @torch.no_grad()
    def raw(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() != 2 or wave.shape[-1] != MAX_SAMPLES:
            raise ValueError(f"expected [B, {MAX_SAMPLES}], got {tuple(wave.shape)}")
        mel = self.to_db(self.mel(wave.float()))
        delta = torchaudio.functional.compute_deltas(mel)
        delta2 = torchaudio.functional.compute_deltas(delta)
        return torch.stack([mel, delta, delta2], dim=1)

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        feats = self.raw(wave)
        return (feats - self.mean) / (self.std + NORM_EPS) if self.normalise else feats


class StatsAccumulator:
    """Exact per-channel mean and unbiased std over all values, in float64."""

    def __init__(self) -> None:
        self.s = torch.zeros(3, dtype=torch.float64)
        self.ss = torch.zeros(3, dtype=torch.float64)
        self.n = 0

    def update(self, raw_feats: torch.Tensor) -> None:
        if raw_feats.dim() != 4 or raw_feats.shape[1] != 3:
            raise ValueError(f"expected [B, 3, F, T], got {tuple(raw_feats.shape)}")
        x = raw_feats.detach().to("cpu", torch.float64)
        self.s += x.sum(dim=(0, 2, 3))
        self.ss += (x * x).sum(dim=(0, 2, 3))
        self.n += x.shape[0] * x.shape[2] * x.shape[3]

    def result(self) -> FeatureStats:
        if self.n < 2:
            raise ValueError("not enough values accumulated")
        mean = self.s / self.n
        var = (self.ss / self.n - mean * mean) * self.n / (self.n - 1)
        if (var <= 0).any():
            raise ValueError(f"non-positive variance {var.tolist()}")
        return FeatureStats(
            mean=tuple(float(v) for v in mean),
            std=tuple(float(v) for v in var.sqrt()),
            n_values=int(self.n),
        )


def check_feature_batch(feats: torch.Tensor, tol_mean: float = 0.2, tol_std: float = 0.2) -> None:
    """Fails if a normalised batch (>= 32 clips) is off zero mean / unit std."""
    x = feats.detach().float()
    mean, std = x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))
    for c, name in enumerate(CHANNELS):
        if abs(mean[c].item()) > tol_mean or abs(std[c].item() - 1.0) > tol_std:
            raise AssertionError(f"{name}: mean {mean[c].item():+.3f}, std {std[c].item():.3f}")


# Official ASVspoof 2019 eval_metrics.py. Numerics unchanged; printing removed, sys.exit -> exceptions.
def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model):
    if cost_model["Ptar"] < 0 or cost_model["Pnon"] < 0 or cost_model["Pspoof"] < 0 or \
            np.abs(cost_model["Ptar"] + cost_model["Pnon"] + cost_model["Pspoof"] - 1) > 1e-10:
        raise ValueError("priors must be positive and sum to one")
    if Pmiss_spoof_asv is None:
        raise ValueError("ASV miss rate on spoof trials is required")
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        raise ValueError("CM scores contain nan or inf")
    if np.unique(combined_scores).size < 3:
        raise ValueError("CM scores must be soft scores, not decisions")
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)
    C1 = cost_model["Ptar"] * (cost_model["Cmiss_cm"] - cost_model["Cmiss_asv"] * Pmiss_asv) - \
        cost_model["Pnon"] * cost_model["Cfa_asv"] * Pfa_asv
    C2 = cost_model["Cfa_cm"] * cost_model["Pspoof"] * (1 - Pmiss_spoof_asv)
    if C1 < 0 or C2 < 0:
        raise ValueError("negative t-DCF weights; check ASV error rates")
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_norm = tDCF / np.minimum(C1, C2)
    return tDCF_norm, CM_thresholds


def spoof_logit_to_cm_score(spoof_logit: np.ndarray) -> np.ndarray:
    return -np.asarray(spoof_logit, dtype=np.float64)


def _split_by_label(labels: np.ndarray, spoof_logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    spoof_logits = np.asarray(spoof_logits, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != spoof_logits.shape:
        raise ValueError(f"shape mismatch: {labels.shape} vs {spoof_logits.shape}")
    if not set(np.unique(labels)) <= {LABEL_BONAFIDE, LABEL_SPOOF}:
        raise ValueError("labels must be 0 or 1")
    if not np.isfinite(spoof_logits).all():
        raise ValueError("non-finite logits")
    cm = spoof_logit_to_cm_score(spoof_logits)
    bona, spoof = cm[labels == LABEL_BONAFIDE], cm[labels == LABEL_SPOOF]
    if bona.size == 0 or spoof.size == 0:
        raise ValueError("both classes are required")
    return bona, spoof


def eer(labels: np.ndarray, spoof_logits: np.ndarray) -> Tuple[float, float]:
    """Returns (eer, threshold_logit); predict spoof when logit >= threshold_logit."""
    bona, spoof = _split_by_label(labels, spoof_logits)
    value, thr_cm = compute_eer(bona, spoof)
    return float(value), float(-thr_cm)


def eer_independent(labels: np.ndarray, spoof_logits: np.ndarray) -> float:
    """sklearn-based EER, used only to cross-check eer()."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(np.asarray(labels), np.asarray(spoof_logits, dtype=np.float64),
                            pos_label=LABEL_SPOOF, drop_intermediate=False)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def asv_operating_point(asv: Dict[str, np.ndarray]) -> Dict[str, float]:
    eer_asv, asv_threshold = compute_eer(asv["target"], asv["nontarget"])
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv = obtain_asv_error_rates(
        asv["target"], asv["nontarget"], asv["spoof"], asv_threshold
    )
    return {
        "eer_asv": float(eer_asv), "asv_threshold": float(asv_threshold),
        "Pfa_asv": float(Pfa_asv), "Pmiss_asv": float(Pmiss_asv), "Pmiss_spoof_asv": float(Pmiss_spoof_asv),
    }


def min_tdcf(labels: np.ndarray, spoof_logits: np.ndarray, asv: Dict[str, np.ndarray]) -> float:
    bona, spoof = _split_by_label(labels, spoof_logits)
    op = asv_operating_point(asv)
    curve, _ = compute_tDCF(bona, spoof, op["Pfa_asv"], op["Pmiss_asv"], op["Pmiss_spoof_asv"], TDCF_COST_MODEL)
    return float(np.min(curve))


def rates_at_threshold(labels: np.ndarray, spoof_logits: np.ndarray, threshold_logit: float) -> Dict[str, float]:
    """threshold_logit must come from dev, never from the split being scored."""
    labels = np.asarray(labels)
    pred = (np.asarray(spoof_logits, dtype=np.float64) >= threshold_logit).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return {
        "threshold_logit": float(threshold_logit), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "spoof_recall": tp / max(tp + fn, 1),
        "spoof_precision": tp / max(tp + fp, 1),
        "bonafide_rejection_rate": fp / max(fp + tn, 1),
        "spoof_miss_rate": fn / max(fn + tp, 1),
        "accuracy": (tp + tn) / max(len(labels), 1),
    }


def bonafide_average_precision(labels: np.ndarray, spoof_logits: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    labels = np.asarray(labels)
    return float(average_precision_score(labels == LABEL_BONAFIDE, spoof_logit_to_cm_score(spoof_logits)))


def per_attack_eer(df: pd.DataFrame, logit_col: str = "logit") -> pd.DataFrame:
    """Each attack scored against all bona fide trials of the same split."""
    bona = df[df["label"] == LABEL_BONAFIDE]
    rows = []
    for attack in sorted(a for a in df["attack"].unique() if a != "bonafide"):
        sub = pd.concat([bona, df[df["attack"] == attack]])
        value, _ = eer(sub["label"].to_numpy(), sub[logit_col].to_numpy())
        rows.append({"attack": attack, "n_spoof": int((df["attack"] == attack).sum()), "eer": value})
    return pd.DataFrame(rows)


SCORE_COLUMNS = ("utt_id", "split", "label", "attack", "logit")


def write_scores(df: pd.DataFrame, path: str, model_tag: str, checkpoint_sha: str) -> None:
    missing = [c for c in SCORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing score columns {missing}")
    if df["utt_id"].duplicated().any():
        raise ValueError("duplicated utterance ids")
    if not np.isfinite(df["logit"].to_numpy(dtype=np.float64)).all():
        raise ValueError("non-finite logits")
    out = df.loc[:, list(SCORE_COLUMNS)].copy()
    out["model"] = model_tag
    out["checkpoint_sha"] = checkpoint_sha
    out["utils_sha"] = utils_sha256()
    out.to_csv(path, index=False, float_format="%.8f")


def read_scores(path: str, manifest: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in SCORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if manifest is not None:
        m = manifest.set_index("utt_id")
        expected = set(m.index[m["split"].isin(df["split"].unique())])
        if set(df["utt_id"]) != expected:
            raise AssertionError(f"{path}: utterance set differs from manifest")
        if (m.loc[df["utt_id"], "label"].to_numpy() != df["label"].to_numpy()).any():
            raise AssertionError(f"{path}: labels differ from manifest")
    return df


def _self_test() -> None:
    rng = np.random.default_rng(0)

    for trial in range(300):
        n_b, n_s = rng.integers(20, 3000, size=2)
        labels = np.r_[np.zeros(n_b, int), np.ones(n_s, int)]
        logits = np.r_[rng.normal(0, 1, n_b), rng.normal(rng.uniform(0, 6), rng.uniform(0.2, 3), n_s)]
        if trial % 2:
            logits = logits.astype(np.float16).astype(np.float64)
        assert abs(eer(labels, logits)[0] - eer_independent(labels, logits)) <= 1.5 / min(n_b, n_s)

    for n_b, n_s, sep in [(500, 4500, 1.5), (2548, 22296, 3.0), (300, 300, 0.0)]:
        labels = np.r_[np.zeros(n_b, int), np.ones(n_s, int)]
        logits = np.r_[rng.normal(-sep / 2, 1, n_b), rng.normal(sep / 2, 1, n_s)]
        e1, thr = eer(labels, logits)
        r = rates_at_threshold(labels, logits, thr)
        assert abs(r["bonafide_rejection_rate"] - e1) < 2.0 / min(n_b, n_s)
        assert abs(r["spoof_miss_rate"] - e1) < 2.0 / min(n_b, n_s)

    labels = np.r_[np.zeros(100, int), np.ones(900, int)]
    logits = np.r_[rng.uniform(-5, -1, 100), rng.uniform(1, 5, 900)]
    assert eer(labels, logits)[0] == 0.0
    assert eer(labels, -logits)[0] == 1.0

    asv = {"target": rng.normal(3, 1, 2000), "nontarget": rng.normal(-3, 1, 20000), "spoof": rng.normal(2, 1, 20000)}
    assert min_tdcf(labels, logits, asv) == 0.0
    labels2 = np.r_[np.zeros(1000, int), np.ones(9000, int)]
    logits2 = np.r_[rng.normal(-1, 1, 1000), rng.normal(1, 1, 9000)]
    assert 0.0 < min_tdcf(labels2, logits2, asv) <= 1.0 + 1e-12
    assert min_tdcf(labels2, rng.normal(0, 1, 10000), asv) > 0.9

    torch.manual_seed(0)
    wave = torch.randn(6, MAX_SAMPLES) * 0.1
    wave[3, 20000:] = 0.0
    fx = LogMelDelta()
    raw = fx(wave)
    assert raw.shape == (6, 3, N_MELS, N_FRAMES)
    assert torch.allclose(raw, torch.cat([fx(wave[i:i + 1]) for i in range(6)]), atol=1e-5)
    acc = StatsAccumulator()
    acc.update(raw[:3])
    acc.update(raw[3:])
    st = acc.result()
    assert np.allclose(st.mean, raw.double().mean(dim=(0, 2, 3)).numpy(), atol=1e-6)
    assert np.allclose(st.std, raw.double().transpose(0, 1).reshape(3, -1).std(dim=1).numpy(), rtol=1e-6)
    check_feature_batch(LogMelDelta(st)(wave))
    assert FeatureStats.from_dict(st.to_dict()) == st

    assert fit_head(torch.ones(1, 10), 4).shape == (1, 4)
    assert fit_head(torch.ones(1, 3), 5).tolist() == [[1.0, 1.0, 1.0, 0.0, 0.0]]

    print(f"audio_utils {UTILS_VERSION} self-test passed (sha {utils_sha256()[:12]})")


if __name__ == "__main__":
    _self_test()
