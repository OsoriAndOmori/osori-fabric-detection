from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from fabric_mvp.config import settings
from fabric_mvp.util.memory import get_rss_mb

logger = logging.getLogger(__name__)


def decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid image content")
    return image


def create_app(predictor=None) -> FastAPI:
    app = FastAPI(title="Fabric Detection MVP", version="0.1.0")

    if predictor is not None:
        app.state.predictor = predictor
    else:
        from fabric_mvp.inference.predictor import HybridPredictor

        app.state.predictor = HybridPredictor(settings)

    # Optional UI mount: FABRIC_ENABLE_GRADIO_UI=true (default true)
    enable_gradio = os.getenv("FABRIC_ENABLE_GRADIO_UI", "true").lower() in {"1", "true", "yes", "on"}
    if enable_gradio:
        try:
            import gradio as gr
            from fabric_mvp.ui.gradio_ui import create_gradio_ui

            gradio_app = create_gradio_ui(app.state.predictor)
            app = gr.mount_gradio_app(app, gradio_app, path="/ui")
        except Exception as exc:
            logger.warning("Gradio UI mount skipped: %s", exc)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "service": "fabric-detection-mvp"}

    @app.post("/predict")
    async def predict(image: UploadFile = File(...)) -> dict:
        try:
            raw = await image.read()
            image_bgr = decode_image(raw)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image: {e}") from e

        try:
            rss_before = get_rss_mb()
            pred = app.state.predictor.predict(image_bgr)
            rss_after = get_rss_mb()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"predictor unavailable: {e}") from e

        if getattr(settings, "log_rss", True) and rss_before is not None and rss_after is not None:
            msg = f"predict rss_mb before={rss_before:.2f} after={rss_after:.2f} delta={rss_after - rss_before:.2f}"
            # Render logs always capture stdout; logging config can vary.
            print(msg, flush=True)
            logger.info(msg)

        return {
            "qa": pred.qa,
            "top3": pred.top3,
            "blend_percent": pred.blend_percent,
            "evidence": getattr(pred, "evidence", {}) or {},
            "model_version": pred.model_version,
        }

    return app


class _UnavailablePredictor:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def predict(self, image_bgr: np.ndarray):  # noqa: ARG002
        raise RuntimeError(self.reason)


try:
    app = create_app()
except Exception as exc:
    logger.exception("Failed to initialize default predictor: %s", exc)
    app = create_app(predictor=_UnavailablePredictor(str(exc)))
