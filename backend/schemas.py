from typing import Literal

from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    prediction: Literal["AI", "Human"]
    score: float = Field(
        ..., ge=0.0, le=1.0, description="sigmoid(logit - threshold_logit); 0.5 is the decision boundary"
    )
    logit: float = Field(..., description="Spoof logit")
    threshold_logit: float = Field(..., description="Dev-EER threshold on the logit")


class GradCAMHeatmap(BaseModel):
    values: list[float]
    rows: int
    cols: int


class AttentionWeights(BaseModel):
    values: list[float]
    num_frames: int


class AnalyzeResponse(BaseModel):
    resnet_prediction: Literal["AI", "Human"]
    resnet_score: float = Field(..., ge=0.0, le=1.0)
    w2v_prediction: Literal["AI", "Human"]
    w2v_score: float = Field(..., ge=0.0, le=1.0)
    resnet: ModelResult
    wav2vec2: ModelResult
    fusion: ModelResult
    gradcam: GradCAMHeatmap | None = None
    attention: AttentionWeights | None = None
    interpretation: str | None = None
    duration_seconds: float = Field(..., description="Length of the window the models analyse")
    audio_seconds: float = Field(..., description="Length of the uploaded audio")
    original_sr: int
