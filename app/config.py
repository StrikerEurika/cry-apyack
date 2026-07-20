from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    default_model: str = "Seg_YOLO26n-seg-v1_crack-seg"
    device: str = "cpu"
    default_confidence_threshold: float = 0.5
    default_patch_size: int = 512
    default_overlap_ratio: float = 0.2
    default_blend_mode: str = "average"
    use_tta: bool = False
    use_clahe: bool = True
    findcrack_checkpoints_dir: str = "../findcrack/checkpoints"

    model_config = {"env_prefix": "CRACK_"}


settings = Settings()
