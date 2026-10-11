import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import CHECKPOINTS, DEVICE, FRONTEND_DIR, FUSION_PATH, HF_MODEL_REPO, MAX_UPLOAD_BYTES, MODELS_DIR

# isort: split
import audio_utils as U

from inference import (
    Fusion,
    LoadedModel,
    calibrated_score,
    label,
    load_resnet,
    load_w2v,
    resnet_input,
    score_resnet,
    score_w2v,
)
from preprocess import prepare
from schemas import AnalyzeResponse, AttentionWeights, GradCAMHeatmap, ModelResult
from xai import attention_buckets, gradcam

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("audioinsight")


@dataclass
class Models:
    resnet: LoadedModel
    w2v: LoadedModel
    fusion: Fusion
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class AppState:
    models: Models | None = None
    error: str | None = None


state = AppState()


def load_models(paths: dict[str, str], verify_sha256: bool = True) -> Models:
    sha = {k: CHECKPOINTS[k]["sha256"] if verify_sha256 else None for k in CHECKPOINTS}
    return Models(
        resnet=load_resnet(paths["resnet18"], sha["resnet18"]),
        w2v=load_w2v(paths["wav2vec2"], sha["wav2vec2"]),
        fusion=Fusion.from_json(FUSION_PATH),
    )


def download_checkpoints() -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    os.makedirs(MODELS_DIR, exist_ok=True)
    return {
        name: hf_hub_download(repo_id=HF_MODEL_REPO, filename=c["file"], local_dir=MODELS_DIR)
        for name, c in CHECKPOINTS.items()
    }


def result(logit: float, threshold_logit: float) -> ModelResult:
    return ModelResult(
        prediction=label(logit, threshold_logit),
        score=calibrated_score(logit, threshold_logit),
        logit=logit,
        threshold_logit=threshold_logit,
    )


def analyze_audio(models: Models, data: bytes, explain: bool = True) -> AnalyzeResponse:
    clip = prepare(data)
    with models.lock:
        rn_logit = score_resnet(models.resnet, clip.raw)
        w2v_logit, attn = score_w2v(models.w2v, clip.wave)
        cam = gradcam(models.resnet.net, resnet_input(models.resnet, clip.raw)) if explain else None
    rn = result(rn_logit, models.resnet.threshold_logit)
    w2v = result(w2v_logit, models.w2v.threshold_logit)
    fused = result(models.fusion.logit(rn_logit, w2v_logit), models.fusion.threshold_logit)
    attention = None
    if explain:
        values, n_frames = attention_buckets(attn)
        attention = AttentionWeights(values=values, num_frames=n_frames)
    return AnalyzeResponse(
        resnet_prediction=rn.prediction,
        resnet_score=rn.score,
        w2v_prediction=w2v.prediction,
        w2v_score=w2v.score,
        resnet=rn,
        wav2vec2=w2v,
        fusion=fused,
        gradcam=GradCAMHeatmap(values=cam[0], rows=cam[1], cols=cam[2]) if cam else None,
        attention=attention,
        duration_seconds=U.MAX_SAMPLES / U.TARGET_SR,
        audio_seconds=round(clip.seconds, 3),
        original_sr=clip.original_sr,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("device: %s", DEVICE)
    try:
        state.models = load_models(download_checkpoints())
        for m in (state.models.resnet, state.models.w2v):
            logger.info(
                "%s: epoch %d, dev EER %.4f%%, threshold %.4f, sha256 %s",
                m.name,
                m.epoch,
                100 * m.dev_eer,
                m.threshold_logit,
                m.sha256[:12],
            )
    except Exception as exc:
        state.error = f"{type(exc).__name__}: {exc}"
        logger.exception("model loading failed")
    yield
    state.models = None


app = FastAPI(title="AudioInsight API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    m = state.models
    return {
        "status": "ok" if m else "unavailable",
        "device": str(DEVICE),
        "resnet_loaded": m is not None,
        "w2v_loaded": m is not None,
        "models": {
            x.name: {"epoch": x.epoch, "dev_eer": x.dev_eer, "threshold_logit": x.threshold_logit, "sha256": x.sha256}
            for x in (m.resnet, m.w2v)
        }
        if m
        else None,
        "error": state.error,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: Annotated[UploadFile, File()]):
    if state.models is None:
        raise HTTPException(status_code=503, detail=f"Models not loaded: {state.error or 'starting up'}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        response = await run_in_threadpool(analyze_audio, state.models, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}") from exc
    logger.info(
        "%s: resnet %s %.3f, wav2vec2 %s %.3f, fusion %s",
        file.filename,
        response.resnet.prediction,
        response.resnet.logit,
        response.wav2vec2.prediction,
        response.wav2vec2.logit,
        response.fusion.prediction,
    )
    return response


if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
