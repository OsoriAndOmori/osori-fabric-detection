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
        # Display as simple percent (no decimals) for readability.
        prob = float(item.get("prob", 0.0) or 0.0)
        pct = int(round(prob * 100.0))
        rows.append(f"{i}. **{item['label']}**: `{pct}%`")
    return "\n".join(rows) if rows else "No prediction"


def _format_blend(blend: dict[str, float]) -> str:
    if not blend:
        return "No blend estimate"
    rows = [f"- **{k}**: `{int(round(v))}%`" for k, v in blend.items()]
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
    .hero {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 4px;
    }
    .hero-logo {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      object-fit: cover;
      border: 1px solid #d8d1c3;
      background: #fffdf8;
      box-shadow: 0 8px 18px rgba(76, 63, 47, 0.10);
      flex: 0 0 auto;
    }
    .hero-text { min-width: 0; }
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

    /* Watermark inside the upload/crop area */
    #roi_editor { position: relative; }
    #roi_editor::before {
      content: "";
      position: absolute;
      inset: 0;
      background: url("/favicon.png") center / 320px 320px no-repeat;
      opacity: 0.10;
      pointer-events: none;
      z-index: 1;
    }
    /* Keep actual editor above watermark layer */
    #roi_editor > * { position: relative; z-index: 2; }

    .footer {
      margin-top: 14px;
      padding-top: 10px;
      border-top: 1px solid rgba(216, 209, 195, 0.75);
      color: #6b6256;
      font-size: 13px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .footer a {
      color: #6b4d2a;
      text-decoration: none;
      font-weight: 600;
    }
    .footer a:hover { text-decoration: underline; }
    """

    # Default to /tmp so Render Free can write (no persistent disk, but /tmp is writable).
    feedback_dir = Path(getattr(settings, "feedback_dir", "/tmp/fabric_mvp_feedback"))

    head = """
    <meta property="og:title" content="오소리 섬유 분류 로봇" />
    <meta property="og:image" content="/favicon.png" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="오소리 섬유 분류 로봇" />
    <meta name="twitter:image" content="/favicon.png" />
    <link rel="icon" href="/favicon.ico" />
    <link rel="apple-touch-icon" href="/favicon.png" />
    """

    with gr.Blocks(title="오소리 섬유 분류 로봇", css=css, theme=gr.themes.Soft(primary_hue="amber"), head=head) as demo:
        infer_state = gr.State(value=None)

        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                '<div class="hero">'
                '<img class="hero-logo" src="/logo.png" alt="logo" />'
                '<div class="hero-text"><div class="hero-title">오소리와 같이 만드는 울 / 캐시미어 분류 로봇</div></div>'
                "</div>"
            )
            gr.HTML(
                '<div class="hero-sub">이미지를 업로드한 뒤, 검증하고 싶은 영역을 드래그해서 크롭하세요.<br>모델은 크롭된 ROI(관심 영역)에 대해서만 분석합니다. 세그멘테이션 오버레이는 판별 근거를 보여줍니다.<br>결과가 틀리면 정답 라벨을 선택해 피드백으로 저장해주세요ㅠㅠ<br>(팁: 아래 툴바에서 Crop 아이콘을 선택한 뒤 드래그하세요.)</div>'
            )

            with gr.Row():
                image_input = gr.ImageEditor(
                    type="numpy",
                    label="Upload + Crop ROI (drag to crop)",
                    sources=["upload", "clipboard"],
                    height=520,
                    # Keep editing surface stable even for huge images so the toolbar doesn't get pushed away.
                    canvas_size=(980, 520),
                    elem_id="roi_editor",
                    transforms=("crop",),
                    crop_size=None,  # free aspect ratio
                    layers=False,
                    brush=False,
                    eraser=False,
                    show_download_button=False,
                    interactive=True,
                )
                overlay_output = gr.Image(type="numpy", label="Detection Evidence Overlay", visible=False)

            show_evidence = gr.Checkbox(
                value=False,
                label="분류 근거 오버레이/마스크 표시(메모리 사용량 증가)",
            )

            predict_btn = gr.Button("Analyze", variant="primary")

            with gr.Row():
                wool_mask = gr.Image(type="numpy", label="울 영역 마스크", visible=False)
                cashmere_mask = gr.Image(type="numpy", label="캐시미어 영역 마스크", visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    retake_html = gr.HTML('<div class="result-card">재촬영: -</div>')
                    top3_md = gr.Markdown("Top 예측 결과가 여기에 표시됩니다.", elem_classes=["result-card"])
                    blend_md = gr.Markdown("혼용률(%) 추정 결과가 여기에 표시됩니다.", elem_classes=["result-card"])
                    evidence_md = gr.Markdown("근거(픽셀 비율 등)가 여기에 표시됩니다.", elem_classes=["result-card"])
                with gr.Column(scale=1):
                    qa_json = gr.JSON(label="품질 검사(QA) 상세")

            gr.Markdown("### 정답 피드백")
            with gr.Row():
                feedback_label = gr.Radio(choices=["wool", "cashmere"], label="정답 라벨", value="cashmere")
                sample_id = gr.Textbox(label="샘플 ID(선택)")
            feedback_note = gr.Textbox(label="메모(선택)", lines=2)
            feedback_btn = gr.Button("피드백 저장", variant="secondary")
            feedback_status = gr.Markdown("아직 피드백이 저장되지 않았습니다.")

            gr.HTML(
                '<div class="footer">'
                '<div>Build: <code>fabric-mvp</code></div>'
                '<div><a href="https://github.com/OsoriAndOmori/osori-fabric-detection" target="_blank" rel="noopener noreferrer">GitHub: OsoriAndOmori/osori-fabric-detection</a></div>'
                "</div>"
            )

            def infer(editor_value: Any, show_ev: bool):
                empty_overlay = np.zeros((256, 256, 3), dtype=np.uint8)
                image_rgb = _extract_editor_rgb(editor_value)
                if image_rgb is None:
                    return (
                        '<div class="result-card">재촬영: -</div>',
                        "이미지가 업로드되지 않았습니다.",
                        "혼용률 추정 없음.",
                        "근거 없음.",
                        {},
                        gr.update(value=empty_overlay, visible=False),
                        gr.update(value=empty_overlay, visible=False),
                        gr.update(value=empty_overlay, visible=False),
                        gr.update(value="cashmere"),
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
                retake_text = "재촬영 필요" if retake else "정상"
                retake_block = f'<div class="result-card">재촬영: <span class="{klass}">{retake_text}</span></div>'

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

                # Default feedback label to the opposite of the model's top-1 prediction.
                pred_top1 = (pred.top3[0]["label"] if pred.top3 else "").strip().lower()
                if pred_top1 == "wool":
                    default_feedback = "cashmere"
                elif pred_top1 == "cashmere":
                    default_feedback = "wool"
                else:
                    default_feedback = "cashmere"

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

                return (
                    retake_block,
                    top3_text,
                    blend_text,
                    evidence_text,
                    qa_payload,
                    gr.update(value=_cap_long_side_rgb(overlay, display_max), visible=bool(show_ev)),
                    gr.update(value=_cap_long_side_rgb(wool, display_max), visible=bool(show_ev)),
                    gr.update(value=_cap_long_side_rgb(cashmere, display_max), visible=bool(show_ev)),
                    gr.update(value=default_feedback),
                    state,
                )

            def save_feedback(editor_value: Any, state: dict | None, correct_label: str, sample: str, note: str):
                if state is None:
                    return "분석 결과가 없습니다. 먼저 Analyze를 눌러주세요."

                image_rgb = _extract_editor_rgb(editor_value)
                if image_rgb is None:
                    return "이미지가 없습니다. 먼저 이미지를 업로드/크롭하세요."

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
                    return f"피드백 저장 완료(`{result.backend}`): `{result.detail}`"
                except Exception as exc:
                    # Fallback: local write (useful for local dev).
                    img_path, line_path = _save_feedback(feedback_dir, payload, image_rgb)
                    return f"피드백 백엔드 실패(`{exc}`)로 로컬 저장: `{img_path}` / `{line_path}`"

            predict_btn.click(
                fn=infer,
                inputs=[image_input, show_evidence],
                outputs=[retake_html, top3_md, blend_md, evidence_md, qa_json, overlay_output, wool_mask, cashmere_mask, feedback_label, infer_state],
            )

            feedback_btn.click(
                fn=save_feedback,
                inputs=[image_input, infer_state, feedback_label, sample_id, feedback_note],
                outputs=[feedback_status],
            )

    return demo
