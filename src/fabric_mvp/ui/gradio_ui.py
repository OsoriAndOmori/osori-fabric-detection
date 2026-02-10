from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import gradio as gr
import numpy as np

from fabric_mvp.config import settings
from fabric_mvp.feedback.store import store_feedback
from fabric_mvp.util.memory import get_rss_mb

logger = logging.getLogger(__name__)


def _extract_editor_rgb(editor_value: Any) -> np.ndarray | None:
    """
    Gradio ImageEditor value is typically a dict-like object:
      {background: ..., layers: [...], composite: ...}
    We prefer composite (cropped result).
    """
    if editor_value is None:
        return None

    # Avoid `or` with numpy arrays (truth value is ambiguous).
    img = None
    if isinstance(editor_value, dict):
        img = editor_value.get("composite")
        if img is None:
            img = editor_value.get("background")
    else:
        img = getattr(editor_value, "composite", None)
        if img is None:
            img = getattr(editor_value, "background", None)

    if img is None:
        return None

    arr = np.asarray(img)
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def _cap_long_side_rgb(image_rgb: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image_rgb
    h, w = image_rgb.shape[:2]
    if max(h, w) <= max_side:
        return image_rgb
    scale = float(max_side) / float(max(h, w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _format_top3(top3: list[dict[str, Any]]) -> str:
    rows = []
    for i, item in enumerate(top3, start=1):
        rows.append(f"{i}. **{item['label']}**: `{item['prob']:.4f}`")
    return "\n".join(rows) if rows else "No prediction"


def _format_blend(blend: dict[str, float]) -> str:
    if not blend:
        return "No blend estimate"
    rows = [f"- **{k}**: `{v:.2f}%`" for k, v in blend.items()]
    return "\n".join(rows)


def _format_evidence(evidence: dict[str, Any] | None) -> str:
    if not evidence:
        return "No evidence"

    rows = [f"- Mode: `{evidence.get('mode', '-')}`"]
    ratios = evidence.get("class_pixel_ratio", {})
    if ratios:
        rows.append("- Pixel ratio by class:")
        for k, v in ratios.items():
            rows.append(f"  - `{k}`: **{v:.2f}%**")

    mixed = evidence.get("mixed_detected", False)
    mixed_classes = evidence.get("mixed_classes", [])
    rows.append(f"- Mixed detected: **{'YES' if mixed else 'NO'}**")
    if mixed_classes:
        rows.append(f"- Mixed classes: `{', '.join(mixed_classes)}`")

    return "\n".join(rows)


def _save_feedback(feedback_dir: Path, payload: dict[str, Any], image_rgb: np.ndarray) -> tuple[Path, Path]:
    feedback_dir.mkdir(parents=True, exist_ok=True)
    images_dir = feedback_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rec_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    image_path = images_dir / f"{rec_id}.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    line_path = feedback_dir / "feedback.jsonl"
    with line_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**payload, "image_path": str(image_path)}, ensure_ascii=False) + "\n")

    return image_path, line_path


def create_gradio_ui(predictor) -> gr.Blocks:
    css = """
    .app-shell {
      max-width: 1180px;
      margin: 0 auto;
      background: linear-gradient(140deg, #f8f9f5 0%, #ece6da 100%);
      border: 1px solid #d8d1c3;
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(76, 63, 47, 0.14);
    }
    .hero-title {
      font-size: 34px;
      font-weight: 800;
      letter-spacing: 0.3px;
      color: #2f2a22;
      margin-bottom: 4px;
    }
    .hero-sub {
      color: #5d5446;
      margin-bottom: 14px;
      font-size: 15px;
    }
    .result-card {
      border: 1px solid #d8d1c3;
      border-radius: 14px;
      padding: 12px 14px;
      background: #fffdf8;
    }
    .retake-yes { color: #a4142a; font-weight: 700; }
    .retake-no { color: #0f7a46; font-weight: 700; }
    """

    # Default to /tmp so Render Free can write (no persistent disk, but /tmp is writable).
    feedback_dir = Path(getattr(settings, "feedback_dir", "/tmp/fabric_mvp_feedback"))

    with gr.Blocks(title="Fabric Detector", css=css, theme=gr.themes.Soft(primary_hue="amber")) as demo:
        infer_state = gr.State(value=None)

        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML('<div class="hero-title">Fabric Quick Classifier</div>')
            gr.HTML(
                '<div class="hero-sub">Upload an image, then drag-crop the region you want to validate. The model runs ONLY on the cropped ROI. Segmentation overlay shows evidence. If wrong, save correct label as feedback.</div>'
            )

            with gr.Row():
                image_input = gr.ImageEditor(
                    type="numpy",
                    label="Upload + Crop ROI (drag to crop)",
                    sources=["upload", "clipboard"],
                    height=520,
                    # Keep editing surface stable even for huge images so the toolbar doesn't get pushed away.
                    canvas_size=(980, 520),
                    transforms=("crop",),
                    crop_size=None,  # free aspect ratio
                    layers=False,
                    brush=False,
                    eraser=False,
                    show_download_button=False,
                )
                overlay_output = gr.Image(type="numpy", label="Detection Evidence Overlay")

            show_evidence = gr.Checkbox(
                value=False,
                label="Show evidence overlay/masks (uses more memory)",
            )

            with gr.Row():
                wool_mask = gr.Image(type="numpy", label="Wool Region Mask")
                cashmere_mask = gr.Image(type="numpy", label="Cashmere Region Mask")

            with gr.Row():
                with gr.Column(scale=1):
                    retake_html = gr.HTML('<div class="result-card">Retake: -</div>')
                    top3_md = gr.Markdown("Top predictions will appear here.", elem_classes=["result-card"])
                    blend_md = gr.Markdown("Blend percent will appear here.", elem_classes=["result-card"])
                    evidence_md = gr.Markdown("Evidence details will appear here.", elem_classes=["result-card"])
                with gr.Column(scale=1):
                    qa_json = gr.JSON(label="QA Details")

            predict_btn = gr.Button("Analyze", variant="primary")

            gr.Markdown("### Correct Label Feedback")
            with gr.Row():
                feedback_label = gr.Radio(choices=["wool", "cashmere"], label="Correct label", value="cashmere")
                sample_id = gr.Textbox(label="Sample ID (optional)")
            feedback_note = gr.Textbox(label="Note (optional)", lines=2)
            feedback_btn = gr.Button("Save Feedback", variant="secondary")
            feedback_status = gr.Markdown("Feedback not saved yet.")

            def infer(editor_value: Any, show_ev: bool):
                empty_overlay = np.zeros((256, 256, 3), dtype=np.uint8)
                image_rgb = _extract_editor_rgb(editor_value)
                if image_rgb is None:
                    return (
                        '<div class="result-card">Retake: -</div>',
                        "No image uploaded.",
                        "No blend estimate.",
                        "No evidence.",
                        {},
                        empty_overlay,
                        empty_overlay,
                        empty_overlay,
                        None,
                    )

                # Resize ROI early to reduce memory usage on Render Free and avoid sending huge arrays to the client.
                image_rgb = _cap_long_side_rgb(image_rgb, int(getattr(settings, "max_input_side", 1600)))

                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                rss_before = get_rss_mb()
                pred = predictor.predict(image_bgr, include_visuals=bool(show_ev))
                rss_after = get_rss_mb()
                if getattr(settings, "log_rss", True) and rss_before is not None and rss_after is not None:
                    msg = f"ui infer rss_mb before={rss_before:.2f} after={rss_after:.2f} delta={rss_after - rss_before:.2f}"
                    print(msg, flush=True)
                    logger.info(msg)

                retake = pred.qa.get("retake", False)
                klass = "retake-yes" if retake else "retake-no"
                retake_text = "RETAKE REQUIRED" if retake else "OK"
                retake_block = f'<div class="result-card">Retake: <span class="{klass}">{retake_text}</span></div>'

                top3_text = _format_top3(pred.top3)
                blend_text = _format_blend(pred.blend_percent)
                evidence_text = _format_evidence(pred.evidence)

                qa_payload = {
                    **pred.qa,
                    "model_version": pred.model_version,
                }

                display_max = int(getattr(settings, "max_display_side", 1024))
                if show_ev and pred.overlay_rgb is not None:
                    overlay = pred.overlay_rgb
                else:
                    # Keep outputs small by default. (The cropped ROI is already visible in the editor.)
                    overlay = empty_overlay

                masks = pred.class_masks_rgb or {}
                wool = masks.get("wool", empty_overlay) if (show_ev and masks) else empty_overlay
                cashmere = masks.get("cashmere", empty_overlay) if (show_ev and masks) else empty_overlay

                # Do NOT keep image arrays in server-side state (helps avoid OOM on Render Free).
                state = {
                    "pred": {
                        "top3": pred.top3,
                        "blend_percent": pred.blend_percent,
                        "qa": pred.qa,
                        "model_version": pred.model_version,
                        "evidence": pred.evidence,
                    }
                }

                return retake_block, top3_text, blend_text, evidence_text, qa_payload, overlay, wool, cashmere, state

            def save_feedback(editor_value: Any, state: dict | None, correct_label: str, sample: str, note: str):
                if state is None:
                    return "No inference result available. Analyze an image first."

                image_rgb = _extract_editor_rgb(editor_value)
                if image_rgb is None:
                    return "No image available. Upload/crop an image first."

                image_rgb = _cap_long_side_rgb(image_rgb, int(getattr(settings, "max_input_side", 1600)))
                pred = state.get("pred", {})

                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "correct_label": correct_label,
                    "sample_id": sample.strip() if sample else "",
                    "note": note.strip() if note else "",
                    "pred_top1": pred.get("top3", [{}])[0].get("label", ""),
                    "pred_top3": pred.get("top3", []),
                    "blend_percent": pred.get("blend_percent", {}),
                    "qa": pred.get("qa", {}),
                    "evidence": pred.get("evidence", {}),
                    "model_version": pred.get("model_version", ""),
                }

                # Prefer configured backend (GitHub recommended on Render Free).
                try:
                    result = store_feedback(settings=settings, payload=payload, image_rgb=image_rgb)
                    return f"Saved feedback via `{result.backend}`: `{result.detail}`"
                except Exception as exc:
                    # Fallback: local write (useful for local dev).
                    img_path, line_path = _save_feedback(feedback_dir, payload, image_rgb)
                    return f"Feedback backend failed (`{exc}`); saved locally: `{img_path}` and `{line_path}`"

            predict_btn.click(
                fn=infer,
                inputs=[image_input, show_evidence],
                outputs=[retake_html, top3_md, blend_md, evidence_md, qa_json, overlay_output, wool_mask, cashmere_mask, infer_state],
            )

            feedback_btn.click(
                fn=save_feedback,
                inputs=[image_input, infer_state, feedback_label, sample_id, feedback_note],
                outputs=[feedback_status],
            )

    return demo
