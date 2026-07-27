import base64
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
    RegisteredModel,
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


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        200: {
            "description": "Successful Prediction. Returns PredictResponse JSON or image/png depending on response_format.",
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/PredictResponse"}},
                "image/png": {"description": "Rendered output image (overlay, visualization, or mask)"},
            },
        }
    },
    summary="Analyze image for crack detections",
    description="Perform crack detection analysis on an uploaded image file. Set response_format='image' to display the rendered prediction image directly inside the API docs interface.",
)
async def predict(
    file: UploadFile = File(..., description="Image file to analyze for cracks (e.g. PNG, JPEG format)."),
    model_name: RegisteredModel = Query(
        RegisteredModel(settings.default_model),
        description="Target registered model variant to run inference. Displays available models registered in the package.",
    ),
    response_format: str = Query(
        "json",
        pattern="^(json|image)$",
        description="Response format: 'json' for full analysis payload or 'image' to display the rendered prediction image directly in API docs interface. Default: 'json'.",
    ),
    render_type: str = Query(
        "overlay",
        pattern="^(overlay|visualization|mask)$",
        description="Type of rendered image output when response_format='image' or include_image=true: 'overlay', 'visualization', or 'mask'. Default: 'overlay'.",
    ),
    confidence_threshold: float = Query(
        settings.default_confidence_threshold,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for crack detection filtering (range: 0.0 to 1.0). Default: 0.5.",
    ),
    patch_size: int = Query(
        settings.default_patch_size,
        ge=128,
        le=2048,
        description="Patch size in pixels for sliding-window inference (range: 128 to 2048). Default: 512.",
    ),
    overlap_ratio: float = Query(
        settings.default_overlap_ratio,
        ge=0.0,
        le=0.5,
        description="Overlap ratio between adjacent patches during sliding-window inference (range: 0.0 to 0.5). Default: 0.2.",
    ),
    use_tta: bool = Query(
        settings.use_tta,
        description="Enable Test-Time Augmentation (TTA) for enhanced detection accuracy. Default: False.",
    ),
    use_clahe: bool = Query(
        settings.use_clahe,
        description="Enable CLAHE image contrast preprocessing before inference. Default: True.",
    ),
    include_contours: bool = Query(
        False,
        description="Whether to include detailed pixel contour coordinates in the response payload. Default: False.",
    ),
    include_image: bool = Query(
        False,
        description="Whether to include base64-encoded rendered image in JSON response payload. Default: False.",
    ),
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

    render_map = {
        "overlay": "overlay",
        "visualization": "visualization",
        "mask": "binary_mask",
    }
    arr: np.ndarray = res[render_map[render_type]]
    if render_type == "mask":
        arr = np.stack([arr, arr, arr], axis=-1)

    if response_format == "image":
        content = ndarray_to_bytes(arr)
        return Response(content=content, media_type="image/png")

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

    image_b64 = None
    if include_image:
        png_bytes = ndarray_to_bytes(arr)
        b64_str = base64.b64encode(png_bytes).decode("utf-8")
        image_b64 = f"data:image/png;base64,{b64_str}"

    target_model_name = model_name.value if hasattr(model_name, "value") else str(model_name)

    return PredictResponse(
        filename=file.filename or "unknown",
        width=w,
        height=h,
        model_name=res.get("active_model", target_model_name),
        crack_count=len(boxes),
        crack_coverage_percentage=coverage_pct,
        total_crack_area_pixels=crack_pixels,
        confidence_threshold=confidence_threshold,
        patch_size=patch_size,
        overlap_ratio=overlap_ratio,
        use_tta=use_tta,
        use_clahe=use_clahe,
        bounding_boxes=boxes,
        detections=detections,
        contours=contours_data,
        image_base64=image_b64,
    )


@app.post(
    "/predict/image",
    responses={
        200: {
            "description": "Rendered crack detection output image",
            "content": {
                "image/png": {"description": "Rendered image (overlay, visualization, or mask)"},
            },
        }
    },
    summary="Analyze image and render visual output image",
    description="Perform crack detection analysis on an uploaded image file and return the rendered visual result (overlay, visualization, or mask) as PNG bytes for display in the API docs interface.",
)
async def predict_image(
    file: UploadFile = File(..., description="Image file to analyze for cracks (e.g. PNG, JPEG format)."),
    model_name: RegisteredModel = Query(
        RegisteredModel(settings.default_model),
        description="Target registered model variant to run inference. Displays available models registered in the package.",
    ),
    render_type: str = Query(
        "overlay",
        pattern="^(overlay|visualization|mask)$",
        description="Render mode for output image: 'overlay', 'visualization', or 'mask'. Default: 'overlay'.",
    ),
    confidence_threshold: float = Query(
        settings.default_confidence_threshold,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for crack detection (range: 0.0 to 1.0). Default: 0.5.",
    ),
    use_tta: bool = Query(
        settings.use_tta,
        description="Enable Test-Time Augmentation (TTA). Default: False.",
    ),
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

