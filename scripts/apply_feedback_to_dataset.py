from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def build_foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    eq = cv2.equalizeHist(blur)
    _, th_bin = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ratio_bin = float(np.count_nonzero(th_bin) / th_bin.size)
    ratio_inv = float(np.count_nonzero(th_inv) / th_inv.size)
    fg = th_bin if abs(ratio_bin - 0.25) <= abs(ratio_inv - 0.25) else th_inv

    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
    return fg


def load_labels(labels_path: Path) -> dict[str, int]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    classes = payload["classes"]
    # classification IDs are without background
    return {classes[i]: i - 1 for i in range(1, len(classes))}


def read_existing_filenames(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    out: set[str] = set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add(row["filename"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply UI feedback records into training dataset")
    parser.add_argument("--feedback", type=Path, default=Path("feedback/feedback.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/labels.json"))
    parser.add_argument("--max-items", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    if not args.feedback.exists():
        raise FileNotFoundError(f"feedback file not found: {args.feedback}")

    class_to_id = load_labels(args.labels)

    train_img_dir = args.data_root / "train/images"
    train_mask_dir = args.data_root / "train/masks"
    train_csv = args.data_root / "train/labels.csv"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_mask_dir.mkdir(parents=True, exist_ok=True)
    train_csv.parent.mkdir(parents=True, exist_ok=True)

    existing = read_existing_filenames(train_csv)
    to_append: list[tuple[str, int]] = []

    lines = args.feedback.read_text(encoding="utf-8").splitlines()
    if args.max_items > 0:
        lines = lines[-args.max_items :]

    added = 0
    skipped = 0

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        rec = json.loads(line)

        label = (rec.get("correct_label") or "").strip().lower()
        image_path = Path(rec.get("image_path", ""))

        if label not in class_to_id:
            skipped += 1
            continue
        if not image_path.exists():
            skipped += 1
            continue

        out_name = f"feedback_{image_path.stem}_{idx:04d}.png"
        if out_name in existing:
            skipped += 1
            continue

        out_img_path = train_img_dir / out_name
        shutil.copy2(image_path, out_img_path)

        image = cv2.imread(str(out_img_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue

        fg = build_foreground_mask(image)
        mask = np.zeros(fg.shape, dtype=np.uint8)
        # segmentation IDs: background=0, wool=1, cashmere=2
        seg_id = class_to_id[label] + 1
        mask[fg > 0] = np.uint8(seg_id)
        cv2.imwrite(str(train_mask_dir / f"{Path(out_name).stem}.png"), mask)

        to_append.append((out_name, class_to_id[label]))
        existing.add(out_name)
        added += 1

    write_header = not train_csv.exists()
    with train_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["filename", "class"])
        writer.writerows(to_append)

    print(f"feedback applied: added={added}, skipped={skipped}, train_csv={train_csv}")


if __name__ == "__main__":
    main()
