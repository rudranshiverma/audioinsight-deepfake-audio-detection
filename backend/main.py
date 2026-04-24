import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from huggingface_hub import hf_hub_download

from config import DEVICE, MODELS_DIR
from inference import (
    ResNetCheckpoint,
    W2VCheckpoint,
    load_resnet,
    load_w2v,
    predict_resnet,
    predict_w2v,
)
from preprocess import load_and_prepare, resnet_features, w2v_features
from schemas import AnalyzeResponse, AttentionWeights, GradCAMHeatmap
from xai import compute_attention_weights, compute_gradcam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class AppState:
    resnet: Optional[ResNetCheckpoint] = None
    w2v: Optional[W2VCheckpoint] = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up on device: %s", DEVICE)

    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        resnet_local = hf_hub_download(
            repo_id="rudranshiverma/audio-deepfake-detection-model",
            filename="resnet18_finetuned.pth",
            local_dir=MODELS_DIR,
        )

        w2v_local = hf_hub_download(
            repo_id="rudranshiverma/audio-deepfake-detection-model",
            filename="w2v_finetuned.pth",
            local_dir=MODELS_DIR,
        )

        app_state.resnet = load_resnet(resnet_local)
        app_state.w2v = load_w2v(w2v_local)

    except Exception as e:
        logger.exception("Model download/load failed: %s", e)

    logger.info(
        "Models ready — ResNet18=%s  Wav2Vec2=%s",
        "✓" if app_state.resnet else "✗",
        "✓" if app_state.w2v else "✗",
    )

    yield

    logger.info("Shutting down — releasing model resources")
    app_state.resnet = None
    app_state.w2v = None


app = FastAPI(
    title="Deepfake Audio Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
    allow_credentials=True,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "resnet_loaded": app_state.resnet is not None,
        "w2v_loaded": app_state.w2v is not None,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if app_state.resnet is None or app_state.w2v is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check Hugging Face repo/files and server logs.",
        )

    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(
        "Received file '%s' (%d bytes)",
        file.filename or "unknown",
        len(audio_bytes),
    )

    t0 = time.perf_counter()

    try:
        waveform, original_sr = load_and_prepare(io.BytesIO(audio_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode audio file: {exc}",
        ) from exc

    rn_ckpt = app_state.resnet

    mel_tensor = resnet_features(
        waveform,
        mel_mean=rn_ckpt.mel_mean,
        mel_std=rn_ckpt.mel_std,
        delta_mean=rn_ckpt.delta_mean,
        delta_std=rn_ckpt.delta_std,
        delta2_mean=rn_ckpt.delta2_mean,
        delta2_std=rn_ckpt.delta2_std,
    )

    resnet_result = predict_resnet(
        rn_ckpt.model,
        mel_tensor,
        rn_ckpt.threshold,
    )

    try:
        cam_values, cam_rows, cam_cols = compute_gradcam(
            rn_ckpt.model,
            mel_tensor,
        )

        gradcam_out = GradCAMHeatmap(
            values=cam_values,
            rows=cam_rows,
            cols=cam_cols,
        )
    except Exception:
        gradcam_out = None

    w2v_ckpt = app_state.w2v

    wav_tensor = w2v_features(waveform)

    w2v_result, attn_weights = predict_w2v(
        w2v_ckpt.model,
        wav_tensor,
        w2v_ckpt.threshold,
    )

    try:
        attn_values, num_frames = compute_attention_weights(attn_weights)

        attention_out = AttentionWeights(
            values=attn_values,
            num_frames=num_frames,
        )
    except Exception:
        attention_out = None

    elapsed = time.perf_counter() - t0

    logger.info("Inference complete in %.2f s", elapsed)

    return AnalyzeResponse(
        resnet_prediction=resnet_result["prediction"],
        resnet_score=resnet_result["score"],
        w2v_prediction=w2v_result["prediction"],
        w2v_score=w2v_result["score"],
        gradcam=gradcam_out,
        attention=attention_out,
        interpretation=None,
        duration_seconds=waveform.shape[1] / 16000,
        original_sr=original_sr,
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend", "dist")
ASSETS_DIR = os.path.join(FRONTEND_DIR, "assets")

if os.path.exists(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))