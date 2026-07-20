from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    default_model: str = "Seg_YOLO26n-seg-v1_crack-seg"
    device: str = "cpu"
    default_confidence_threshold: float = 0.5
    default_patch_size: int = 512
    use_tta: bool = False
    use_clahe: bool = True

    model_config = {"env_prefix": "CRACK_"}


settings = Settings()
