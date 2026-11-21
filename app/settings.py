from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vessel_speed_knots: float = 8.0
    bbox_margin_deg: float = 0.15
    max_span_deg: float = 3.0
    refine_factor: int = 1
    cache_dir: Path = Path("data/cache")
    smoothing: bool = True
    smoothing_samples: int = 8

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


settings = Settings()
