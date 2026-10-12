"""
Separates pipeline errors from model behaviour on real-world files.

Takes ASVspoof 2019 LA eval clips whose correct answer is known, re-encodes each one the way real-world files
arrive (44.1 kHz WAV, MP3 at 16 kHz and 44.1 kHz), and scores every version through the API code path.
A lossless 44.1 kHz copy must score like the original FLAC (tests decoding and resampling); MP3 copies show
how much compression alone moves each model.

    python tests/codec_probe.py --la-root <folder containing ASVspoof2019_LA_eval> --models-dir models
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHECKPOINTS, REPO_ROOT
from main import analyze_audio, load_models


def encode(x, sr, fmt, subtype=None):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


def variants(path):
    x, sr = sf.read(path, dtype="float32")
    up = torchaudio.functional.resample(torch.from_numpy(x), sr, 44100).numpy().clip(-1, 1)
    return {
        "flac_16k": open(path, "rb").read(),
        "wav_44k": encode(up, 44100, "WAV", "PCM_16"),
        "mp3_16k": encode(x, sr, "MP3"),
        "mp3_44k": encode(up, 44100, "MP3"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--la-root", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--per-class", type=int, default=15)
    args = ap.parse_args()

    models = load_models({k: os.path.join(args.models_dir, c["file"]) for k, c in CHECKPOINTS.items()})
    ref = pd.read_csv(os.path.join(REPO_ROOT, "results", "scores", "wav2vec2_seed0_eval.csv"))
    clips = ref.groupby("label").sample(n=args.per_class, random_state=1)
    flac_dir = os.path.join(args.la_root, "ASVspoof2019_LA_eval", "flac")

    rows = []
    for i, r in enumerate(clips.itertuples(), start=1):
        for name, data in variants(os.path.join(flac_dir, f"{r.utt_id}.flac")).items():
            out = analyze_audio(models, data, explain=False)
            rows.append(
                {
                    "utt": r.utt_id,
                    "truth": "AI" if r.label == 1 else "Human",
                    "variant": name,
                    "resnet": out.resnet.logit,
                    "resnet_pred": out.resnet.prediction,
                    "w2v": out.wav2vec2.logit,
                    "w2v_pred": out.wav2vec2.prediction,
                    "fusion_pred": out.fusion.prediction,
                }
            )
        print(f"  {i}/{len(clips)} clips", flush=True)
    df = pd.DataFrame(rows)

    base = df[df["variant"] == "flac_16k"].set_index("utt")
    print("\nAccuracy by version (bona fide / spoof, same clips):")
    for model in ("resnet", "w2v", "fusion"):
        acc = df.assign(ok=df[f"{model}_pred"] == df["truth"]).pivot_table(
            index="variant", columns="truth", values="ok", aggfunc="mean"
        )
        print(f"\n{model}\n{acc.round(2).to_string()}")
    print("\nMedian |logit change| vs the original FLAC:")
    for v in ("wav_44k", "mp3_16k", "mp3_44k"):
        cur = df[df["variant"] == v].set_index("utt")
        print(
            f"  {v:8s} resnet {np.median(np.abs(cur['resnet'] - base['resnet'])):.3f}   "
            f"wav2vec2 {np.median(np.abs(cur['w2v'] - base['w2v'])):.3f}"
        )


if __name__ == "__main__":
    main()
