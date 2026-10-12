import io
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import torchaudio

from config import MIN_SECONDS

# isort: split
import audio_utils as U


@dataclass
class Clip:
    raw: torch.Tensor
    wave: torch.Tensor
    original_sr: int
    seconds: float


def _decode_ffmpeg(data: bytes):
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in"), os.path.join(tmp, "out.wav")
        with open(src, "wb") as fh:
            fh.write(data)
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", src, "-vn", "-c:a", "pcm_f32le", dst],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except FileNotFoundError as exc:
            raise ValueError("format not supported without ffmpeg; upload WAV, FLAC, OGG or MP3") from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError("unsupported or corrupted audio file") from exc
        return sf.read(dst, dtype="float32", always_2d=True)


def decode(data: bytes):
    """Any common audio format -> float32 mono at its native sample rate."""
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        x, sr = _decode_ffmpeg(data)
    if x.size == 0:
        raise ValueError("no audio samples found")
    return x.mean(axis=1), int(sr)


def to_training_grid(x: np.ndarray, sr: int) -> np.ndarray:
    """16 kHz on the int16 grid, the representation every training clip had."""
    if sr != U.TARGET_SR:
        x = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(x)), sr, U.TARGET_SR).numpy()
    q = np.clip(np.round(x.astype(np.float64) * 32768.0), -32768, 32767).astype(np.int16)
    return U.int16_to_float32(q)


def normalise_wave(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / np.sqrt(x.var() + 1e-7)


def prepare(data: bytes) -> Clip:
    x, sr = decode(data)
    x = to_training_grid(x, sr)
    if x.size < MIN_SECONDS * U.TARGET_SR:
        raise ValueError(f"clip shorter than {MIN_SECONDS} s")
    head = x[: U.MAX_SAMPLES]
    return Clip(
        raw=U.fit_head(torch.from_numpy(head)),
        wave=U.fit_head(torch.from_numpy(normalise_wave(head))),
        original_sr=sr,
        seconds=x.size / U.TARGET_SR,
    )
