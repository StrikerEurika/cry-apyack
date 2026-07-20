import logging
from pathlib import Path

import numpy as np

from findcrack import CrackInferencePipeline, load_model, list_models

from app.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton that holds a pre-loaded CrackInferencePipeline."""

    def __init__(self) -> None:
        self._pipeline: CrackInferencePipeline | None = None
        self._current_variant: str | None = None

    @property
    def pipeline(self) -> CrackInferencePipeline:
        if self._pipeline is None:
            raise RuntimeError("ModelManager not initialized — call load_default() first")
        return self._pipeline

    def load_default(self) -> None:
        variant = settings.default_model
        logger.info("Loading model variant: %s on device=%s", variant, settings.device)
        model = load_model(variant, device=settings.device, local_checkpoint=True)
        self._pipeline = CrackInferencePipeline(
            model,
            device=settings.device,
            patch_size=settings.default_patch_size,
            confidence_threshold=settings.default_confidence_threshold,
            use_tta=settings.use_tta,
            use_clahe=settings.use_clahe,
        )
        self._current_variant = variant
        logger.info("Model loaded: %s", variant)

    def predict(self, image: np.ndarray, confidence_threshold: float | None = None) -> dict:
        pipe = self.pipeline
        if confidence_threshold is not None:
            pipe.confidence_threshold = confidence_threshold
        return pipe.predict(image)

    def available_models(self) -> list[str]:
        return list_models()


model_manager = ModelManager()
