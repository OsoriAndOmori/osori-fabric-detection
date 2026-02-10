from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from fabric_mvp.api.main import create_app


class FakePredictor:
    def predict(self, image_bgr: np.ndarray):
        class Result:
            qa = {"blur": 120.0, "exposure": "ok", "retake": False}
            top3 = [
                {"label": "wool", "prob": 0.7},
                {"label": "cotton", "prob": 0.2},
                {"label": "polyester", "prob": 0.1},
            ]
            blend_percent = {"wool": 70.0, "cotton": 20.0, "polyester": 10.0}
            model_version = "0.1.0-test"

        return Result()


def test_predict_response_format() -> None:
    app = create_app(predictor=FakePredictor())
    client = TestClient(app)

    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    response = client.post(
        "/predict",
        files={"image": ("sample.jpg", encoded.tobytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()

    assert "qa" in payload
    assert "top3" in payload
    assert "blend_percent" in payload
    assert "model_version" in payload
    assert len(payload["top3"]) == 3
