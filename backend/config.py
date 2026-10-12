import os
import sys

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = os.path.join(BASE_DIR, "models")
FUSION_PATH = os.path.join(REPO_ROOT, "results", "fusion.json")
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend", "dist")

HF_MODEL_REPO = "rudranshiverma/audio-deepfake-detection-model"
CHECKPOINTS = {
    "resnet18": {
        "file": "resnet18_seed0.pt",
        "sha256": "89beeb29853d571f0e5ab973b914661f8bc8deb7aaa2c481a7f95ae8897c8ffe",
    },
    "wav2vec2": {
        "file": "wav2vec2_seed0.pt",
        "sha256": "58dd55996a0c759c8eb97b64bd0f6ef35b2e41090d97fafdcca9521374576212",
    },
}
FUSION_INPUTS = ["resnet18_seed0", "wav2vec2_seed0"]

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MIN_SECONDS = 0.1
GRADCAM_LAYER = "layer4"
ATTENTION_BUCKETS = 34
