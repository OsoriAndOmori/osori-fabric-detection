from __future__ import annotations

import csv
import json
from pathlib import Path


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    classes = payload.get("classes", [])
    if not classes or not isinstance(classes, list):
        raise ValueError("labels.json must include non-empty 'classes' list")
    if classes[0] != "background":
        raise ValueError("labels.json classes[0] must be 'background' for segmentation mode")

    return classes


def load_classification_csv(csv_path: Path) -> list[tuple[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {csv_path}")

    rows: list[tuple[str, int]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "filename" not in row or "class" not in row:
                raise ValueError("labels.csv must have columns: filename,class")
            rows.append((row["filename"], int(row["class"])))

    return rows
