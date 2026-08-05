import copy
import logging
import threading
from typing import Any

import numpy as np
from findcrack import CrackInferencePipeline, load_model, list_models
from findcrack.models.registry import MODEL_REGISTRY

from app.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager for loading, caching, and running findcrack pipelines."""

    def __init__(self) -> None:
        self._loaded_pipelines: dict[str, CrackInferencePipeline] = {}
        self.default_model_name: str = settings.default_model
        self._load_lock = threading.Lock()

    def get_pipeline(
        self,
        model_name: Any = None,
        patch_size: int | None = None,
        overlap_ratio: float | None = None,
        confidence_threshold: float | None = None,
        use_tta: bool | None = None,
        use_clahe: bool | None = None,
        blend_mode: str | None = None,
        force_single_forward: bool = False,
    ) -> CrackInferencePipeline:
        target_model = (
            model_name.value if hasattr(model_name, "value") else model_name
        ) or self.default_model_name

        if target_model not in self._loaded_pipelines:
            with self._load_lock:
                if target_model not in self._loaded_pipelines:
                    self.load_model(target_model)

        pipeline = self._loaded_pipelines[target_model]

        try:
            per_request = copy.copy(pipeline)
        except Exception:
            per_request = pipeline

        if force_single_forward:
            if hasattr(per_request, "patch_size") and per_request.patch_size > 512:
                raise ValueError(
                    f"force_single_forward requires image <= patch_size, but patch_size={per_request.patch_size} > 512"
                )
            per_request.use_tta = False
            per_request.overlap_ratio = 0.0
            per_request.patch_size = 512

        if patch_size is not None:
            per_request.patch_size = patch_size
        if overlap_ratio is not None:
            per_request.overlap_ratio = overlap_ratio
        if confidence_threshold is not None:
            per_request.confidence_threshold = confidence_threshold
        if use_tta is not None:
            per_request.use_tta = use_tta
        if use_clahe is not None and hasattr(per_request, "preprocessor"):
            per_request.preprocessor.use_clahe = use_clahe
        if blend_mode is not None:
            per_request.blend_mode = blend_mode

        return per_request

    def load_model(self, model_name: str) -> None:
        logger.info("Loading findcrack model '%s' on device='%s'...", model_name, settings.device)
        model = load_model(model_name, device=settings.device, local_checkpoint=True)
        pipeline = CrackInferencePipeline(
            model=model,
            device=settings.device,
            patch_size=settings.default_patch_size,
            overlap_ratio=settings.default_overlap_ratio,
            confidence_threshold=settings.default_confidence_threshold,
            use_tta=settings.use_tta,
            use_clahe=settings.use_clahe,
            blend_mode=settings.default_blend_mode,
        )
        self._loaded_pipelines[model_name] = pipeline
        logger.info("Model '%s' loaded successfully.", model_name)

    def load_default(self) -> None:
        self.load_model(self.default_model_name)

    def available_models(self) -> list[str]:
        return list_models()

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Model '{model_name}' not found in registry.")
        return MODEL_REGISTRY[model_name]

    def predict(
        self,
        image: np.ndarray,
        model_name: Any = None,
        confidence_threshold: float | None = None,
        patch_size: int | None = None,
        overlap_ratio: float | None = None,
        use_tta: bool | None = None,
        use_clahe: bool | None = None,
        blend_mode: str | None = None,
        force_single_forward: bool = False,
    ) -> dict[str, Any]:
        target_model_name = (
            model_name.value if hasattr(model_name, "value") else model_name
        ) or self.default_model_name

        pipeline = self.get_pipeline(
            model_name=target_model_name,
            patch_size=patch_size,
            overlap_ratio=overlap_ratio,
            confidence_threshold=confidence_threshold,
            use_tta=use_tta,
            use_clahe=use_clahe,
            blend_mode=blend_mode,
            force_single_forward=force_single_forward,
        )

        raw_result = pipeline.predict(image)
        raw_result["active_model"] = target_model_name
        return raw_result


model_manager = ModelManager()
