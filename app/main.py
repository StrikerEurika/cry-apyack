import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.config import settings
from app.schemas import (
    BatchPredictResponse,
    CrackDetectionDetail,
    ErrorResponse,
    HealthResponse,
    ModelDetailResponse,
    ModelsResponse,
    PredictResponse,
)
from app.services.inference import model_manager
from app.utils import bytes_to_ndarray, ndarray_to_bytes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing cry-apyack service with findcrack...")
    model_manager.load_default()
    yield
    logger.info("Shutting down cry-apyack service.")


app = FastAPI(
    title="cry-apyack (Crack Detection API)",
    description="FastAPI Service utilizing findcrack deep learning engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        device=settings.device,
        loaded_models=list(model_manager._loaded_pipelines.keys()),
    )


@app.get("/models", response_model=ModelsResponse)
async def models():
    return ModelsResponse(models=model_manager.available_models())


@app.get("/models/{model_name}", response_model=ModelDetailResponse)
async def model_detail(model_name: str):
    try:
        info = model_manager.get_model_info(model_name)
        raw = info.get("raw_config", {})
        return ModelDetailResponse(
            model_id=model_name,
            name=raw.get("name", model_name),
            version=raw.get("version", "1.0.0"),
            description=raw.get("description", ""),
            backend=info.get("backend", "unknown"),
            architecture=raw.get("architecture", {}),
            artifacts=raw.get("artifacts", {}),
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str | None = Query(None, description="Target model variant"),
    confidence_threshold: float | None = Query(None, ge=0.0, le=1.0),
    patch_size: int | None = Query(None, ge=128, le=2048),
    overlap_ratio: float | None = Query(None, ge=0.0, le=0.5),
    use_tta: bool | None = Query(None),
    use_clahe: bool | None = Query(None),
    include_contours: bool = Query(False),
):
    raw = await file.read()
    try:
        image = bytes_to_ndarray(raw)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image file: {str(e)}",
        )

    res = model_manager.predict(
        image=image,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        patch_size=patch_size,
        overlap_ratio=overlap_ratio,
        use_tta=use_tta,
        use_clahe=use_clahe,
    )

    boxes = res["bounding_boxes"]
    h, w, _ = res["original_image"].shape
    binary_mask = res["binary_mask"]
    crack_pixels = int(np.count_nonzero(binary_mask))
    total_pixels = h * w
    coverage_pct = round((crack_pixels / total_pixels) * 100.0, 4)

    detections = [
        CrackDetectionDetail(
            id=idx,
            bbox=box,
            area_pixels=int((box[2] - box[0]) * (box[3] - box[1])),
            confidence=1.0,
        )
        for idx, box in enumerate(boxes)
    ]

    contours_data = None
    if include_contours and "contours" in res:
        contours_data = [c.reshape(-1, 2).tolist() for c in res["contours"]]

    return PredictResponse(
        filename=file.filename or "unknown",
        width=w,
        height=h,
        model_name=res.get("active_model", settings.default_model),
        crack_count=len(boxes),
        crack_coverage_percentage=coverage_pct,
        total_crack_area_pixels=crack_pixels,
        confidence_threshold=confidence_threshold or settings.default_confidence_threshold,
        patch_size=patch_size or settings.default_patch_size,
        overlap_ratio=overlap_ratio or settings.default_overlap_ratio,
        use_tta=use_tta if use_tta is not None else settings.use_tta,
        use_clahe=use_clahe if use_clahe is not None else settings.use_clahe,
        bounding_boxes=boxes,
        detections=detections,
        contours=contours_data,
    )


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str | None = Query(None),
    render_type: str = Query("overlay", pattern="^(overlay|visualization|mask)$"),
    confidence_threshold: float | None = Query(None, ge=0.0, le=1.0),
    use_tta: bool | None = Query(None),
):
    raw = await file.read()
    image = bytes_to_ndarray(raw)

    result = model_manager.predict(
        image=image,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        use_tta=use_tta,
    )

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
