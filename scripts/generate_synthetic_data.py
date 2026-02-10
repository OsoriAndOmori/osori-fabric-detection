from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np


CLASSES = ["background", "wool", "cotton", "polyester", "nylon"]


def draw_fiber_lines(canvas: np.ndarray, mask: np.ndarray, class_id: int, n_lines: int) -> None:
    h, w = canvas.shape[:2]
    color_base = [int(v) for v in np.random.randint(50, 230, size=3)]

    for _ in range(n_lines):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        angle = random.uniform(0, np.pi)
        length = random.randint(w // 6, w // 2)
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
        thickness = random.randint(2, 6)

        color = tuple(int(np.clip(c + random.randint(-20, 20), 0, 255)) for c in color_base)
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.line(mask, (x1, y1), (x2, y2), int(class_id), thickness)


def generate_one(size: int = 512) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = np.random.randint(20, 45)
    mask = np.zeros((size, size), dtype=np.uint8)

    chosen = random.sample([1, 2, 3, 4], k=random.randint(1, 3))
    counts: dict[int, int] = {}

    for cls in chosen:
        n_lines = random.randint(25, 60)
        draw_fiber_lines(image, mask, cls, n_lines=n_lines)
        counts[cls] = n_lines

    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image, mask, counts


def write_classification_csv(out_path: Path, rows: list[tuple[str, int]]) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    parser.add_argument("--train-size", type=int, default=60)
    parser.add_argument("--val-size", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    (args.output / "train/images").mkdir(parents=True, exist_ok=True)
    (args.output / "train/masks").mkdir(parents=True, exist_ok=True)
    (args.output / "val/images").mkdir(parents=True, exist_ok=True)
    (args.output / "val/masks").mkdir(parents=True, exist_ok=True)

    cls_train_rows = []
    cls_val_rows = []

    for split, n in [("train", args.train_size), ("val", args.val_size)]:
        for idx in range(n):
            image, mask, counts = generate_one(size=args.image_size)
            name = f"{split}_{idx:04d}.png"

            img_path = args.output / split / "images" / name
            msk_path = args.output / split / "masks" / f"{Path(name).stem}.png"

            cv2.imwrite(str(img_path), image)
            cv2.imwrite(str(msk_path), mask)

            if counts:
                dominant = max(counts.items(), key=lambda x: x[1])[0]
                row = (name, dominant - 1)  # classification excludes background
                if split == "train":
                    cls_train_rows.append(row)
                else:
                    cls_val_rows.append(row)

    with (args.output / "labels.json").open("w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES}, f, indent=2)

    write_classification_csv(args.output / "train/labels.csv", cls_train_rows)
    write_classification_csv(args.output / "val/labels.csv", cls_val_rows)

    print(f"Synthetic dataset generated at: {args.output}")


if __name__ == "__main__":
    main()
