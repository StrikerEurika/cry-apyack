from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    device: str


class ModelsResponse(BaseModel):
    models: list[str]


class PredictResponse(BaseModel):
    filename: str
    width: int
    height: int
    crack_count: int
    confidence_threshold: float
    bounding_boxes: list[list[int]]


class ErrorResponse(BaseModel):
    error: str
