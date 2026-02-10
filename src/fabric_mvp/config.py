from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FABRIC_", env_file=".env", extra="ignore")

    labels_path: Path = Path("datasets/labels.json")
    segmentation_model_path: Path | None = Path("checkpoints/segmentation_best.pt")
    classification_model_path: Path | None = Path("checkpoints/classification_best.pt")
    segmentation_onnx_path: Path | None = Path("exports/segmentation.onnx")
    classification_onnx_path: Path | None = Path("exports/classification.onnx")
    backend: str = "auto"  # auto|onnx|torch
    max_input_side: int = 1600  # resize long side before inference to reduce OOM risk (Render free)
    max_display_side: int = 1024  # cap images returned to UI (overlay/masks) to reduce RAM/bandwidth
    log_rss: bool = True  # print rss before/after inference to stdout for debugging on Render
    max_concurrent_inferences: int = 1  # 1 = serialize inference (recommended). 0 = unlimited.
    min_width: int = 256
    min_height: int = 256
    blur_threshold: float = 80.0
    dark_pixel_ratio_threshold: float = 0.55
    bright_pixel_ratio_threshold: float = 0.55

    # Feedback storage
    # - local: write to feedback_dir (use /tmp on Render)
    # - github: upload to GitHub repo via Contents API (recommended on Render Free)
    # - disabled: do nothing
    feedback_backend: str = "local"  # local|github|disabled
    feedback_dir: Path = Path("/tmp/fabric_mvp_feedback")
    feedback_github_repo: str | None = None  # "owner/repo"
    feedback_github_token: str | None = None  # PAT with repo access to the target repo (set as secret env var)
    feedback_github_branch: str = "main"
    feedback_github_path_prefix: str = "feedback"


settings = Settings()
