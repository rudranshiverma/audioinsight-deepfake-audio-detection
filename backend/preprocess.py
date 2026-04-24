#preprocess.py: audio loading and feature extraction, supports WAV / MP3 / FLAC / M4A reliably on Windows.
'''Pipelines:
1. resnet_features(...)-> 3-channel tensor [3, 128, T]
   Mel + Delta + Delta2
2. w2v_features(...)
   -> raw waveform [64000]'''

import io
from typing import Tuple, Union
import librosa
import torch
import torchaudio
import torchaudio.transforms as T

from config import (
    TARGET_SR,
    MAX_SAMPLES,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
)

#global transforms
mel_tf = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
a2db = T.AmplitudeToDB()

#audio loader
def _load_waveform(
    source: Union[str, bytes, io.BytesIO]
) -> Tuple[torch.Tensor, int]:
    #prioritize torchaudio and librosa is fallback
    #returns waveform [1, N], sample_rate

    if isinstance(source, bytes):
        source = io.BytesIO(source)

    #torchaudio
    try:
        waveform, sr = torchaudio.load(source)
        # stereo -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sr
    except Exception as e:
        torchaudio_error = str(e)

    #fallback librosa
    try:
        if isinstance(source, io.BytesIO):
            source.seek(0)
        y, sr = librosa.load(source, sr=None, mono=True)

        waveform = torch.tensor(
            y,dtype=torch.float32).unsqueeze(0)
        return waveform, sr

    except Exception as e:
        raise RuntimeError(f"Could not decode audio file: {str(e)}")

#resample
def _resample(
    waveform: torch.Tensor,
    src_sr: int
) -> torch.Tensor:

    if src_sr != TARGET_SR:
        waveform = T.Resample(src_sr, TARGET_SR)(waveform)

    return waveform

#length=4sec
def _fit_length(
    waveform: torch.Tensor
) -> torch.Tensor:
    #pad/trim to max_samples
    n = waveform.shape[1]
    if n > MAX_SAMPLES:
        waveform = waveform[:, :MAX_SAMPLES]
    elif n < MAX_SAMPLES:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, MAX_SAMPLES - n)
        )

    return waveform

#loader
def load_and_prepare(
    source: Union[str, bytes, io.BytesIO]
) -> Tuple[torch.Tensor, int]:
    #returns:waveform [1, 64000], original_sr

    waveform, original_sr = _load_waveform(source)
    waveform = _resample(waveform, original_sr)
    waveform = _fit_length(waveform)
    return waveform, original_sr

#resnet features
def resnet_features(
    waveform: torch.Tensor,
    mel_mean: float,
    mel_std: float,
    delta_mean: float,
    delta_std: float,
    delta2_mean: float,
    delta2_std: float,
) -> torch.Tensor:

    mel = mel_tf(waveform)
    mel = a2db(mel)

    delta = torchaudio.functional.compute_deltas(mel)
    delta2 = torchaudio.functional.compute_deltas(delta)

    mel = (mel - mel_mean) / (mel_std + 1e-6)
    delta = (delta - delta_mean) / (delta_std + 1e-6)
    delta2 = (delta2 - delta2_mean) / (delta2_std + 1e-6)

    features = torch.cat(
        [mel, delta, delta2],
        dim=0
    )  # [3,128,T]

    return features.float()

#w2v features
def w2v_features(
    waveform: torch.Tensor
) -> torch.Tensor:
    return waveform.squeeze(0).float()