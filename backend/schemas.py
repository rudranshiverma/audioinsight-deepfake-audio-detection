#schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

class ModelResult(BaseModel):
    prediction: str = Field(..., description="'AI' or 'Human'")
    score: float= Field(..., ge=0.0, le=1.0, description="P(AI) after sigmoid")
    threshold: float = Field(..., description="Decision threshold used")

class GradCAMHeatmap(BaseModel):
    #flattened 2-D GradCAM activation map returned as a flat list of floats in row-major order plus the original shape so the frontend can reconstruct or resize as needed.
    values: List[float]
    rows: int   #frequency axis
    cols: int   #time axis


class AttentionWeights(BaseModel):
    values: List[float]   
    num_frames: int      

class AnalyzeResponse(BaseModel):
    #predictions
    resnet_prediction: str  = Field(..., description="'AI' or 'Human'")
    resnet_score: float     = Field(..., ge=0.0, le=1.0)
    w2v_prediction: str     = Field(..., description="'AI' or 'Human'")
    w2v_score: float        = Field(..., ge=0.0, le=1.0)

    #xai outputs
    gradcam: Optional[GradCAMHeatmap]       = None
    attention: Optional[AttentionWeights]   = None

    #interpretation
    interpretation: Optional[str] = None

    #audio metadata
    duration_seconds: float = Field(..., description="Clipped/padded duration fed to models")
    original_sr: int        = Field(..., description="Sample rate of the uploaded file")


class ErrorResponse(BaseModel):
    detail: str