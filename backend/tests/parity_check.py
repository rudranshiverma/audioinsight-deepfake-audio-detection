"""
Checks that the backend reproduces notebook 03's eval scores.

Scores a stratified sample of ASVspoof 2019 LA eval clips (every attack plus bona fide) through the same
code path as POST /analyze and compares the logits with results/scores/*_eval.csv. Notebook 03 ran in fp16 on
GPU, so small logit differences are expected; a decision may only differ where the reference logit sits
within MARGIN of the threshold.

    python tests/parity_check.py --la-root <folder containing ASVspoof2019_LA_eval> --models-dir models
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHECKPOINTS, REPO_ROOT
from main import analyze_audio, load_models

MARGIN = 0.25
MEDIAN_TOL = 0.05
P99_TOL = 0.25


def reference_scores() -> pd.DataFrame:
    frames = []
    for tag, col in (("resnet18_seed0", "ref_resnet"), ("wav2vec2_seed0", "ref_w2v")):
        df = pd.read_csv(os.path.join(REPO_ROOT, "results", "scores", f"{tag}_eval.csv"))
        frames.append(df.set_index("utt_id")[["label", "attack", "logit"]].rename(columns={"logit": col}))
    return frames[0].join(frames[1][["ref_w2v"]], how="inner")


def check(name, ref, got, thr):
    diff = np.abs(got - ref)
    flips = (ref >= thr) != (got >= thr)
    unexplained = flips & (np.abs(ref - thr) >= MARGIN)
    ok = np.median(diff) < MEDIAN_TOL and np.percentile(diff, 99) < P99_TOL and not unexplained.any()
    print(
        f"{name:9s} median |diff| {np.median(diff):.4f}  p99 {np.percentile(diff, 99):.4f}  max {diff.max():.4f}  "
        f"decision flips {flips.sum()} (outside margin {unexplained.sum()})  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--la-root", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = load_models({k: os.path.join(args.models_dir, c["file"]) for k, c in CHECKPOINTS.items()})
    print("checkpoints match the evaluated sha256; models loaded", flush=True)
    ref = reference_scores()
    n = min(args.per_group, ref["attack"].value_counts().min())
    sample = ref.groupby("attack").sample(n=n, random_state=args.seed)
    flac_dir = os.path.join(args.la_root, "ASVspoof2019_LA_eval", "flac")
    if not os.path.isdir(flac_dir):
        sys.exit(f"{flac_dir} not found; --la-root must contain ASVspoof2019_LA_eval")

    rows, t0 = [], time.time()
    log_every = max(1, len(sample) // 10)
    for i, utt in enumerate(sample.index, start=1):
        with open(os.path.join(flac_dir, f"{utt}.flac"), "rb") as fh:
            r = analyze_audio(models, fh.read(), explain=False)
        rows.append((r.resnet.logit, r.wav2vec2.logit, r.fusion.logit))
        if i % log_every == 0:
            print(f"  {i}/{len(sample)} clips, {time.time() - t0:.0f} s", flush=True)
    got = np.array(rows)

    fusion = models.fusion
    ref_fused = np.array([fusion.logit(a, b) for a, b in zip(sample["ref_resnet"], sample["ref_w2v"])])
    print(f"\n{len(sample)} eval clips across {sample['attack'].nunique()} groups")
    results = [
        check("ResNet18", sample["ref_resnet"].to_numpy(), got[:, 0], models.resnet.threshold_logit),
        check("wav2vec2", sample["ref_w2v"].to_numpy(), got[:, 1], models.w2v.threshold_logit),
        check("Fusion", ref_fused, got[:, 2], fusion.threshold_logit),
    ]
    print("\nPARITY PASSED" if all(results) else "\nPARITY FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
