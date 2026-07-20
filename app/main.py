import logging
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.schemas import ErrorResponse, HealthResponse, ModelsResponse, PredictResponse
from app.services.inference import model_manager
from app.utils import bytes_to_ndarray, ndarray_to_bytes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    model_manager.load_default()
    yield
    logger.info("Shutting down.")


app = FastAPI(title="cry-apyack", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", device=settings.device)


@app.get("/models", response_model=ModelsResponse)
async def models():
    return ModelsResponse(models=model_manager.available_models())


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str | None = Query(None),
    confidence_threshold: float | None = Query(None),
):
    if confidence_threshold is None:
        confidence_threshold = settings.default_confidence_threshold

    raw = await file.read()
    image = bytes_to_ndarray(raw)

    result = model_manager.predict(image, confidence_threshold=confidence_threshold)

    return PredictResponse(
        filename=file.filename or "unknown",
        width=result["original_image"].shape[1],
        height=result["original_image"].shape[0],
        crack_count=len(result["bounding_boxes"]),
        confidence_threshold=confidence_threshold,
        bounding_boxes=result["bounding_boxes"],
    )


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    render_type: str = Query("overlay", pattern="^(overlay|visualization|mask)$"),
):
    raw = await file.read()
    image = bytes_to_ndarray(raw)

    result = model_manager.predict(image)

    render_map = {
        "overlay": "overlay",
        "visualization": "visualization",
        "mask": "binary_mask",
    }
    arr: np.ndarray = result[render_map[render_type]]

    if render_type == "mask":
        arr = np.stack([arr, arr, arr], axis=-1)

    content = ndarray_to_bytes(arr)
    return Response(content=content, media_type="image/png")
