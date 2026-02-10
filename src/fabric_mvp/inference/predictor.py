from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

from fabric_mvp import __version__
from fabric_mvp.config import Settings
from fabric_mvp.data.constants import IMAGENET_MEAN, IMAGENET_STD
from fabric_mvp.qa import QAConfig, run_quality_checks


@dataclass
class PredictionResult:
    qa: dict[str, Any]
    top3: list[dict[str, Any]]
    blend_percent: dict[str, float]
    model_version: str
    evidence: dict[str, Any] | None = None
    overlay_rgb: np.ndarray | None = None
    class_masks_rgb: dict[str, np.ndarray] | None = None


def _softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)


def _topk_from_scores(labels: list[str], scores: np.ndarray, k: int = 3) -> list[dict[str, Any]]:
    idx = np.argsort(scores)[::-1][:k]
    return [{"label": labels[i], "prob": round(float(scores[i]), 4)} for i in idx]


class HybridPredictor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.classes = self._load_classes(settings.labels_path)
        self.fiber_classes = self.classes[1:]

        self.qa_config = QAConfig(
            min_width=settings.min_width,
            min_height=settings.min_height,
            blur_threshold=settings.blur_threshold,
            dark_ratio_threshold=settings.dark_pixel_ratio_threshold,
            bright_ratio_threshold=settings.bright_pixel_ratio_threshold,
        )

        self.backend = settings.backend

        self.seg_model_onnx = None
        self.clf_model_onnx = None
        self.seg_model_torch = None
        self.clf_model_torch = None

        self._load_models()

    def _load_classes(self, labels_path: Path) -> list[str]:
        with labels_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        classes = payload["classes"]
        if classes[0] != "background":
            raise ValueError("labels.json must start with 'background'")
        return classes

    def _load_models(self) -> None:
        backend = self.backend

        if backend in {"auto", "onnx"}:
            if ort is None:
                raise RuntimeError("onnxruntime is not installed")
            sess_opts = ort.SessionOptions()
            # Reduce memory growth on small-memory hosts (Render Free).
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            sess_opts.enable_mem_pattern = False
            sess_opts.enable_cpu_mem_arena = False

            if self.settings.segmentation_onnx_path and self.settings.segmentation_onnx_path.exists():
                self.seg_model_onnx = ort.InferenceSession(
                    str(self.settings.segmentation_onnx_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
            if self.settings.classification_onnx_path and self.settings.classification_onnx_path.exists():
                self.clf_model_onnx = ort.InferenceSession(
                    str(self.settings.classification_onnx_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )

        if backend in {"auto", "torch"}:
            # Torch backend is optional. Runtime Docker image does not include torch.
            try:
                import torch

                from fabric_mvp.models.classifier import build_efficientnet
                from fabric_mvp.models.unet import UNet
            except Exception:
                return

            if self.seg_model_onnx is None and self.settings.segmentation_model_path and self.settings.segmentation_model_path.exists():
                num_classes = len(self.classes)
                model = UNet(in_channels=3, num_classes=num_classes)
                checkpoint = torch.load(self.settings.segmentation_model_path, map_location="cpu")
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
                self.seg_model_torch = model

            if self.clf_model_onnx is None and self.settings.classification_model_path and self.settings.classification_model_path.exists():
                num_classes = len(self.fiber_classes)
                model = build_efficientnet(num_classes=num_classes, pretrained=False)
                checkpoint = torch.load(self.settings.classification_model_path, map_location="cpu")
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
                self.clf_model_torch = model

    @property
    def has_any_model(self) -> bool:
        return any([self.seg_model_onnx, self.clf_model_onnx, self.seg_model_torch, self.clf_model_torch])

    def _preprocess(self, image_bgr: np.ndarray, size: int) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        chw = np.transpose(image, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def _predict_segmentation(self, image_bgr: np.ndarray) -> np.ndarray | None:
        inp = self._preprocess(image_bgr, size=512)

        if self.seg_model_onnx is not None:
            input_name = self.seg_model_onnx.get_inputs()[0].name
            logits = self.seg_model_onnx.run(None, {input_name: inp})[0]
            probs = _softmax_np(logits, axis=1)
            pred_mask = np.argmax(probs, axis=1).squeeze(0).astype(np.uint8)
            return pred_mask

        if self.seg_model_torch is not None:
            import torch
            import torch.nn.functional as F

            with torch.no_grad():
                logits_t = self.seg_model_torch(torch.from_numpy(inp))
                probs_t = F.softmax(logits_t, dim=1)
                pred_mask = torch.argmax(probs_t, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                return pred_mask

        return None

    def _predict_classification(self, image_bgr: np.ndarray) -> np.ndarray | None:
        inp = self._preprocess(image_bgr, size=224)

        if self.clf_model_onnx is not None:
            input_name = self.clf_model_onnx.get_inputs()[0].name
            logits = self.clf_model_onnx.run(None, {input_name: inp})[0]
            probs = _softmax_np(logits, axis=1).squeeze(0)
            return probs

        if self.clf_model_torch is not None:
            import torch
            import torch.nn.functional as F

            with torch.no_grad():
                logits_t = self.clf_model_torch(torch.from_numpy(inp))
                probs_t = F.softmax(logits_t, dim=1).squeeze(0).cpu().numpy()
                return probs_t

        return None

    def _segmentation_blend(self, pred_mask: np.ndarray) -> dict[str, float]:
        non_bg = pred_mask[pred_mask > 0]
        if non_bg.size == 0:
            return {label: 0.0 for label in self.fiber_classes}

        total = float(non_bg.size)
        out: dict[str, float] = {}
        for class_idx, label in enumerate(self.fiber_classes, start=1):
            pct = float((non_bg == class_idx).sum() / total * 100.0)
            out[label] = round(pct, 2)
        return out

    def _segmentation_visuals(self, image_bgr: np.ndarray, pred_mask_small: np.ndarray, max_side: int = 1024) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # Create visualization at capped resolution to avoid OOM on large inputs.
        h, w = image_bgr.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w)))
        out_w = max(1, int(w * scale))
        out_h = max(1, int(h * scale))

        img_vis = cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        mask_vis = cv2.resize(pred_mask_small, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        color_map = {
            0: (0, 0, 0),
            1: (65, 150, 255),
            2: (255, 176, 0),
        }

        overlay = img_vis.copy()
        class_masks_rgb: dict[str, np.ndarray] = {}

        for class_idx, label in enumerate(self.fiber_classes, start=1):
            cls_mask = (mask_vis == class_idx).astype(np.uint8)
            cls_color = np.array(color_map.get(class_idx, (128, 128, 128)), dtype=np.uint8)

            color_layer = np.zeros_like(img_vis, dtype=np.uint8)
            color_layer[cls_mask > 0] = cls_color
            overlay = np.where(color_layer > 0, (0.55 * overlay + 0.45 * color_layer).astype(np.uint8), overlay)

            mask_rgb = np.zeros_like(img_vis, dtype=np.uint8)
            mask_rgb[cls_mask > 0] = cls_color
            class_masks_rgb[label] = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), class_masks_rgb

    def predict(self, image_bgr: np.ndarray, *, include_visuals: bool = False) -> PredictionResult:
        # Cap input size early to avoid OOM on small-memory hosts (Render Free).
        h, w = image_bgr.shape[:2]
        max_side = int(getattr(self.settings, "max_input_side", 1600))
        if max_side > 0 and max(h, w) > max_side:
            scale = float(max_side) / float(max(h, w))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        qa = run_quality_checks(image_bgr, self.qa_config)

        if not self.has_any_model:
            top3 = [{"label": lbl, "prob": round(1 / max(1, len(self.fiber_classes)), 4)} for lbl in self.fiber_classes[:3]]
            blend = {lbl: round(100.0 / max(1, len(self.fiber_classes)), 2) for lbl in self.fiber_classes}
            return PredictionResult(qa=qa, top3=top3, blend_percent=blend, model_version=f"{__version__}-no-model")

        pred_mask = self._predict_segmentation(image_bgr)
        if pred_mask is not None:
            blend = self._segmentation_blend(pred_mask)
            scores = np.array([blend.get(lbl, 0.0) / 100.0 for lbl in self.fiber_classes], dtype=np.float32)
            top3 = _topk_from_scores(self.fiber_classes, scores, k=min(3, len(self.fiber_classes)))

            sorted_labels = sorted(blend.items(), key=lambda kv: kv[1], reverse=True)
            mixed_classes = [k for k, v in sorted_labels if v >= 15.0]

            evidence = {
                "mode": "segmentation",
                "class_pixel_ratio": blend,
                "mixed_detected": len(mixed_classes) >= 2,
                "mixed_classes": mixed_classes,
            }

            overlay_rgb = None
            class_masks_rgb = None
            if include_visuals:
                overlay_rgb, class_masks_rgb = self._segmentation_visuals(image_bgr, pred_mask)

            return PredictionResult(
                qa=qa,
                top3=top3,
                blend_percent=blend,
                model_version=__version__,
                evidence=evidence,
                overlay_rgb=overlay_rgb,
                class_masks_rgb=class_masks_rgb,
            )

        probs = self._predict_classification(image_bgr)
        if probs is None:
            top3 = [{"label": lbl, "prob": 0.0} for lbl in self.fiber_classes[:3]]
            return PredictionResult(qa=qa, top3=top3, blend_percent={}, model_version=__version__)

        top3 = _topk_from_scores(self.fiber_classes, probs, k=min(3, len(self.fiber_classes)))

        blend = {}
        norm = float(np.sum(probs)) + 1e-8
        for idx, label in enumerate(self.fiber_classes):
            pct = float(probs[idx] / norm * 100.0)
            blend[label] = round(pct, 2)

        mixed_classes = [k for k, v in blend.items() if v >= 15.0]
        evidence = {
            "mode": "classification",
            "class_pixel_ratio": blend,
            "mixed_detected": len(mixed_classes) >= 2,
            "mixed_classes": mixed_classes,
        }

        return PredictionResult(qa=qa, top3=top3, blend_percent=blend, model_version=__version__, evidence=evidence)
