from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QAConfig:
    min_width: int = 256
    min_height: int = 256
    blur_threshold: float = 80.0
    dark_ratio_threshold: float = 0.55
    bright_ratio_threshold: float = 0.55


def laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_status(image_bgr: np.ndarray, dark_thr: int = 30, bright_thr: int = 225) -> tuple[str, float, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    total = gray.size
    dark_ratio = float(np.sum(gray <= dark_thr) / total)
    bright_ratio = float(np.sum(gray >= bright_thr) / total)

    if dark_ratio > 0.55:
        return "under", dark_ratio, bright_ratio
    if bright_ratio > 0.55:
        return "over", dark_ratio, bright_ratio
    return "ok", dark_ratio, bright_ratio


def run_quality_checks(image_bgr: np.ndarray, cfg: QAConfig) -> dict:
    h, w = image_bgr.shape[:2]
    blur = laplacian_variance(image_bgr)
    exp_status, dark_ratio, bright_ratio = exposure_status(image_bgr)

    low_res = h < cfg.min_height or w < cfg.min_width
    blurry = blur < cfg.blur_threshold
    under = dark_ratio > cfg.dark_ratio_threshold
    over = bright_ratio > cfg.bright_ratio_threshold

    return {
        "blur": round(blur, 2),
        "exposure": exp_status,
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "resolution": {"width": w, "height": h, "ok": not low_res},
        "retake": bool(low_res or blurry or under or over),
    }
