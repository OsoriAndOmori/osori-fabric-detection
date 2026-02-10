from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


@dataclass
class GitHubFeedbackConfig:
    repo: str  # owner/repo
    token: str
    branch: str = "main"
    base_dir: str = "feedback"  # folder inside the repo


def _gh_put_file(*, cfg: GitHubFeedbackConfig, path: str, content_bytes: bytes, message: str) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{cfg.repo}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": cfg.branch,
    }

    r = requests.put(
        url,
        headers={
            "Authorization": f"Bearer {cfg.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json=payload,
        timeout=30,
    )

    if r.status_code >= 300:
        raise RuntimeError(f"GitHub upload failed {r.status_code}: {r.text[:300]}")

    return r.json()


def upload_feedback_to_github(
    *,
    cfg: GitHubFeedbackConfig,
    record: dict[str, Any],
    image_rgb,
    record_id: str,
) -> dict[str, str]:
    """Uploads one feedback record + image as two immutable files.

    This avoids 'append to jsonl' conflicts and works without persistent disk.
    """

    ts = datetime.now(timezone.utc)
    day = ts.strftime("%Y-%m-%d")

    img_rel = f"{cfg.base_dir}/images/{day}/{record_id}.png"
    rec_rel = f"{cfg.base_dir}/records/{day}/{record_id}.json"

    import cv2
    import numpy as np

    img_rgb = np.asarray(image_rgb)
    ok, png = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode feedback image")

    _gh_put_file(
        cfg=cfg,
        path=img_rel,
        content_bytes=png.tobytes(),
        message=f"feedback: add image {record_id}",
    )

    record_payload = {**record, "image_path": img_rel}
    _gh_put_file(
        cfg=cfg,
        path=rec_rel,
        content_bytes=json.dumps(record_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        message=f"feedback: add record {record_id}",
    )

    return {"record_path": rec_rel, "image_path": img_rel}
