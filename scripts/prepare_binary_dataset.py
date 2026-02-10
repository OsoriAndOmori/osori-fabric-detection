from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def copy_split(files: list[Path], out_dir: Path, prefix: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, src in enumerate(files):
        name = f"{prefix}_{i:04d}{src.suffix.lower()}"
        dst = out_dir / name
        shutil.copy2(src, dst)
        names.append(name)
    return names


def build_foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    eq = cv2.equalizeHist(blur)

    _, th_bin = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def score(mask: np.ndarray) -> float:
        ratio = float(np.count_nonzero(mask) / mask.size)
        return abs(ratio - 0.25)

    fg = th_bin if score(th_bin) <= score(th_inv) else th_inv
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
    return fg


def write_masks_for_split(
    image_dir: Path,
    mask_dir: Path,
    rows: list[tuple[str, int]],
    class_to_mask_id: dict[int, int],
) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    for filename, cls in rows:
        image_path = image_dir / filename
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read image for mask generation: {image_path}")

        fg = build_foreground_mask(image)
        mask = np.zeros(fg.shape, dtype=np.uint8)
        mask_id = class_to_mask_id[cls]
        mask[fg > 0] = np.uint8(mask_id)

        mask_path = mask_dir / f"{Path(filename).stem}.png"
        cv2.imwrite(str(mask_path), mask)


def write_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare binary wool/cashmere dataset for classification training")
    parser.add_argument("--source", type=Path, default=Path("/Users/1113259/Desktop/sample"))
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--cashmere-dirs",
        type=str,
        default="cashmere,cashmere2",
        help="comma separated folder names under --source treated as cashmere",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="clean existing output/train|val folders before copy")
    parser.add_argument("--generate-masks", action="store_true", help="generate pseudo segmentation masks")
    args = parser.parse_args()

    wool_dir = args.source / "wool"
    cashmere_dir_names = [x.strip() for x in args.cashmere_dirs.split(",") if x.strip()]
    cashmere_dirs = [args.source / name for name in cashmere_dir_names]

    if not wool_dir.exists():
        raise FileNotFoundError("source must include 'wool' folder")
    if not any(d.exists() for d in cashmere_dirs):
        raise FileNotFoundError(f"source must include one of cashmere folders: {cashmere_dir_names}")

    random.seed(args.seed)

    wool = collect_images(wool_dir)
    cashmere: list[Path] = []
    used_cashmere_dirs: list[str] = []
    for d in cashmere_dirs:
        if d.exists():
            cashmere.extend(collect_images(d))
            used_cashmere_dirs.append(d.name)

    if not wool or not cashmere:
        raise ValueError("both classes must contain at least one image")

    random.shuffle(wool)
    random.shuffle(cashmere)

    wool_val_n = max(1, int(len(wool) * args.val_ratio))
    cashmere_val_n = max(1, int(len(cashmere) * args.val_ratio))

    wool_val = wool[:wool_val_n]
    wool_train = wool[wool_val_n:]

    cashmere_val = cashmere[:cashmere_val_n]
    cashmere_train = cashmere[cashmere_val_n:]

    train_img_dir = args.output / "train/images"
    val_img_dir = args.output / "val/images"

    if args.clean and args.output.exists():
        for p in [args.output / "train", args.output / "val"]:
            if p.exists():
                shutil.rmtree(p)

    train_wool_names = copy_split(wool_train, train_img_dir, "wool")
    train_cashmere_names = copy_split(cashmere_train, train_img_dir, "cashmere")
    val_wool_names = copy_split(wool_val, val_img_dir, "wool")
    val_cashmere_names = copy_split(cashmere_val, val_img_dir, "cashmere")

    train_rows = [(name, 0) for name in train_wool_names] + [(name, 1) for name in train_cashmere_names]
    val_rows = [(name, 0) for name in val_wool_names] + [(name, 1) for name in val_cashmere_names]
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    (args.output / "train").mkdir(parents=True, exist_ok=True)
    (args.output / "val").mkdir(parents=True, exist_ok=True)

    write_csv(args.output / "train/labels.csv", train_rows)
    write_csv(args.output / "val/labels.csv", val_rows)

    if args.generate_masks:
        # classification labels: wool=0, cashmere=1
        # segmentation mask ids: background=0, wool=1, cashmere=2
        class_to_mask_id = {0: 1, 1: 2}
        write_masks_for_split(train_img_dir, args.output / "train/masks", train_rows, class_to_mask_id)
        write_masks_for_split(val_img_dir, args.output / "val/masks", val_rows, class_to_mask_id)

    with (args.output / "labels.json").open("w", encoding="utf-8") as f:
        json.dump({"classes": ["background", "wool", "cashmere"]}, f, indent=2)

    print("Prepared binary dataset")
    print(f"- cashmere source dirs: {used_cashmere_dirs}")
    print(f"- train: wool={len(train_wool_names)}, cashmere={len(train_cashmere_names)}")
    print(f"- val: wool={len(val_wool_names)}, cashmere={len(val_cashmere_names)}")
    print(f"- masks: {'generated' if args.generate_masks else 'not generated'}")
    print(f"- output: {args.output}")


if __name__ == "__main__":
    main()
