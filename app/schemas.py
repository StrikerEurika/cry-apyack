from enum import Enum
from pydantic import BaseModel, Field


def get_registered_model_enum() -> type[Enum]:
    """Generates an Enum of available models registered in findcrack."""
    from app.services.inference import model_manager

    models = model_manager.available_models()
    return Enum("RegisteredModel", {m: m for m in models}, type=str)


def __getattr__(name: str):
    if name == "RegisteredModel":
        return get_registered_model_enum()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    image_base64: str | None = Field(
        None,
        description="Base64 encoded data URI (data:image/png;base64,...) of the rendered prediction image if include_image=True",
    )
    mask_base64: str | None = Field(
        None,
        description="Base64 encoded binary mask (PNG) for remote stitching.",
    )


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

