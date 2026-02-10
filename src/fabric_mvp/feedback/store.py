from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import requests

from fabric_mvp.config import Settings


@dataclass
class FeedbackResult:
    backend: str
    record_id: str
    detail: str


def _now_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:10]}"


def _encode_png_rgb(image_rgb) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode png")
    return buf.tobytes()


def _github_put_file(*, token: str, repo: str, branch: str, path: str, content_bytes: bytes, message: str) -> None:
    # GitHub Contents API: PUT /repos/{owner}/{repo}/contents/{path}
    # https://docs.github.com/en/rest/repos/contents#create-or-update-file-contents
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
        "branch": branch,
    }
    r = requests.put(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"github upload failed: {r.status_code} {r.text[:300]}")


def store_feedback(*, settings: Settings, payload: dict[str, Any], image_rgb) -> FeedbackResult:
    """
    Store a feedback record + image.

    Render Free has no persistent disk; use `FABRIC_FEEDBACK_BACKEND=github` with a private repo.
    """
    backend = (settings.feedback_backend or "disabled").strip().lower()
    rec_id = _now_id()

    if backend == "disabled":
        return FeedbackResult(backend="disabled", record_id=rec_id, detail="feedback disabled")

    if backend == "local":
        base = Path(settings.feedback_dir)
        base.mkdir(parents=True, exist_ok=True)
        images_dir = base / "images"
        records_dir = base / "records"
        images_dir.mkdir(parents=True, exist_ok=True)
        records_dir.mkdir(parents=True, exist_ok=True)

        img_path = images_dir / f"{rec_id}.png"
        rec_path = records_dir / f"{rec_id}.json"

        png = _encode_png_rgb(image_rgb)
        img_path.write_bytes(png)
        rec_path.write_text(json.dumps({**payload, "record_id": rec_id, "image_path": str(img_path)}, ensure_ascii=False), encoding="utf-8")

        return FeedbackResult(backend="local", record_id=rec_id, detail=f"saved to {rec_path}")

    if backend == "github":
        if not settings.feedback_github_repo or not settings.feedback_github_token:
            raise RuntimeError("github feedback backend requires FABRIC_FEEDBACK_GITHUB_REPO and FABRIC_FEEDBACK_GITHUB_TOKEN")

        prefix = (settings.feedback_github_path_prefix or "feedback").strip("/").strip()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        img_rel = f"{prefix}/{day}/images/{rec_id}.png"
        rec_rel = f"{prefix}/{day}/records/{rec_id}.json"

        png = _encode_png_rgb(image_rgb)
        rec_bytes = json.dumps({**payload, "record_id": rec_id, "image_path": img_rel}, ensure_ascii=False, indent=2).encode("utf-8")

        _github_put_file(
            token=settings.feedback_github_token,
            repo=settings.feedback_github_repo,
            branch=settings.feedback_github_branch,
            path=img_rel,
            content_bytes=png,
            message=f"feedback: add image {rec_id}",
        )
        _github_put_file(
            token=settings.feedback_github_token,
            repo=settings.feedback_github_repo,
            branch=settings.feedback_github_branch,
            path=rec_rel,
            content_bytes=rec_bytes,
            message=f"feedback: add record {rec_id}",
        )

        return FeedbackResult(backend="github", record_id=rec_id, detail=f"uploaded to {settings.feedback_github_repo}:{rec_rel}")

    raise RuntimeError(f"unknown feedback backend: {backend}")

