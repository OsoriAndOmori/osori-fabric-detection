from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from fabric_mvp.config import settings
from fabric_mvp.util.memory import get_rss_mb

logger = logging.getLogger(__name__)

def _static_path(filename: str) -> str:
    p = os.path.join(os.path.dirname(__file__), "..", "static", filename)
    return os.path.abspath(p)


def decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid image content")
    return image


def create_app(predictor=None) -> FastAPI:
    app = FastAPI(title="Fabric Detection MVP", version="0.1.0")

    # Favicon routes must be registered BEFORE mounting Gradio at `/ui`,
    # otherwise Gradio will serve its own favicon for `/ui/favicon.ico`.
    @app.get("/ui/favicon.ico", include_in_schema=False)
    def ui_favicon_ico():
        ico = _static_path("favicon.ico")
        if os.path.exists(ico):
            return FileResponse(ico, media_type="image/x-icon")
        return RedirectResponse(url="/ui/favicon.png", status_code=307)

    @app.get("/ui/favicon.png", include_in_schema=False)
    def ui_favicon_png():
        png = _static_path("favicon.png")
        if os.path.exists(png):
            return FileResponse(png, media_type="image/png")
        raise HTTPException(status_code=404, detail="favicon not configured")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon_ico():
        ico = _static_path("favicon.ico")
        if os.path.exists(ico):
            return FileResponse(ico, media_type="image/x-icon")
        # Fallback to PNG if ico wasn't generated.
        return RedirectResponse(url="/favicon.png", status_code=307)

    @app.get("/favicon.png", include_in_schema=False)
    def favicon_png():
        png = _static_path("favicon.png")
        if os.path.exists(png):
            return FileResponse(png, media_type="image/png")
        raise HTTPException(status_code=404, detail="favicon not configured")

    @app.get("/logo.png", include_in_schema=False)
    def logo_png():
        png = _static_path("logo.png")
        if os.path.exists(png):
            return FileResponse(png, media_type="image/png")
        raise HTTPException(status_code=404, detail="logo not configured")

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
            ui_path = "/ui"
            app = gr.mount_gradio_app(app, gradio_app, path=ui_path)

            # Railway (and some reverse proxies) can break Gradio's base-path detection when mounted under
            # a subpath like `/ui`, causing the frontend to call `/gradio_api/*` and `/theme.css` at root.
            # Add a small compatibility rewrite so both `/ui/gradio_api/*` and `/gradio_api/*` work.
            GRADIO_COMPAT_PREFIXES = ("/gradio_api", "/assets", "/static", "/theme.css")

            @app.middleware("http")
            async def _rewrite_gradio_paths(request, call_next):  # noqa: ANN001
                path = request.scope.get("path") or ""
                if path.startswith(ui_path + "/"):
                    return await call_next(request)
                if path == ui_path:
                    # Normalize to trailing slash
                    request.scope["path"] = ui_path + "/"
                    return await call_next(request)
                if any(path == p or path.startswith(p + "/") for p in GRADIO_COMPAT_PREFIXES):
                    request.scope["path"] = ui_path + path
                return await call_next(request)
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
