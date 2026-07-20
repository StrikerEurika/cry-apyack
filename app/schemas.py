from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str
    loaded_models: list[str]


class ModelsResponse(BaseModel):
    models: list[str]


class CrackDetectionDetail(BaseModel):
    id: int
    bbox: list[int] = Field(..., description="[xmin, ymin, xmax, ymax]")
    area_pixels: int
    confidence: float


class PredictResponse(BaseModel):
    filename: str
    width: int
    height: int
    model_name: str
    crack_count: int
    crack_coverage_percentage: float
    total_crack_area_pixels: int
    confidence_threshold: float
    patch_size: int
    overlap_ratio: float
    use_tta: bool
    use_clahe: bool
    bounding_boxes: list[list[int]]
    detections: list[CrackDetectionDetail]
    contours: list[list[list[int]]] | None = None


class BatchPredictResponse(BaseModel):
    total_images: int
    successful: int
    failed: int
    results: list[PredictResponse]


class ModelDetailResponse(BaseModel):
    model_id: str
    name: str
    version: str
    description: str
    backend: str
    architecture: dict
    artifacts: dict


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
