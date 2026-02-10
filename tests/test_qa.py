from __future__ import annotations

import numpy as np

from fabric_mvp.qa import QAConfig, run_quality_checks


def test_qa_retake_on_low_resolution() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    qa = run_quality_checks(image, QAConfig(min_width=256, min_height=256))
    assert qa["retake"] is True
    assert qa["resolution"]["ok"] is False


def test_qa_exposure_status_over() -> None:
    image = np.full((512, 512, 3), 255, dtype=np.uint8)
    qa = run_quality_checks(image, QAConfig(min_width=256, min_height=256))
    assert qa["exposure"] == "over"
    assert qa["retake"] is True
