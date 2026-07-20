import logging
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

    def get_pipeline(
        self,
        model_name: str | None = None,
        patch_size: int | None = None,
        overlap_ratio: float | None = None,
        confidence_threshold: float | None = None,
        use_tta: bool | None = None,
        use_clahe: bool | None = None,
        blend_mode: str | None = None,
    ) -> CrackInferencePipeline:
        target_model = model_name or self.default_model_name

        if target_model not in self._loaded_pipelines:
            self.load_model(target_model)

        pipeline = self._loaded_pipelines[target_model]

        # Apply runtime overrides if provided
        if patch_size is not None:
            pipeline.patch_size = patch_size
        if overlap_ratio is not None:
            pipeline.overlap_ratio = overlap_ratio
        if confidence_threshold is not None:
            pipeline.confidence_threshold = confidence_threshold
        if use_tta is not None:
            pipeline.use_tta = use_tta
        if use_clahe is not None and hasattr(pipeline, "preprocessor"):
            pipeline.preprocessor.use_clahe = use_clahe
        if blend_mode is not None:
            pipeline.blend_mode = blend_mode

        return pipeline

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
        model_name: str | None = None,
        confidence_threshold: float | None = None,
        patch_size: int | None = None,
        overlap_ratio: float | None = None,
        use_tta: bool | None = None,
        use_clahe: bool | None = None,
        blend_mode: str | None = None,
    ) -> dict[str, Any]:
        pipeline = self.get_pipeline(
            model_name=model_name,
            patch_size=patch_size,
            overlap_ratio=overlap_ratio,
            confidence_threshold=confidence_threshold,
            use_tta=use_tta,
            use_clahe=use_clahe,
            blend_mode=blend_mode,
        )

        # Call findcrack pipeline (now supports np.ndarray input)
        raw_result = pipeline.predict(image)
        raw_result["active_model"] = model_name or self.default_model_name
        return raw_result


model_manager = ModelManager()
