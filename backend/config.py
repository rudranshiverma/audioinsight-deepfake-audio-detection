#config.py: numbers derived directly from training
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SR: int = 16_000           
MAX_SAMPLES: int = TARGET_SR * 4  
#mel-spectrogram constants
N_FFT: int = 1024
HOP_LENGTH: int = 256
N_MELS: int = 128

#model checkpoint paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESNET_CKPT = os.path.join(MODELS_DIR, "resnet18_finetuned.pth")
W2V_CKPT= os.path.join(MODELS_DIR, "w2v_finetuned.pth")

WAV2VEC2_BASE = "facebook/wav2vec2-base"

#inference thresholds for fallback
RESNET_DEFAULT_THRESHOLD: float = 0.5
W2V_DEFAULT_THRESHOLD: float= 0.5

#gradcam target layer
GRADCAM_TARGET_LAYER = "layer4"

ATTENTION_BUCKETS: int = 34