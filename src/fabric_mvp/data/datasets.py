from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from fabric_mvp.data.constants import IMAGENET_MEAN, IMAGENET_STD
from fabric_mvp.data.schema import load_classification_csv


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"cannot read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int64)


def segmentation_transforms(size: int, train: bool) -> A.Compose:
    common = [
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(transpose_mask=True),
    ]
    if not train:
        return A.Compose(common)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            *common,
        ]
    )


def classification_transforms(size: int, train: bool) -> A.Compose:
    common = [
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    if not train:
        return A.Compose(common)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussNoise(var_limit=(5, 25), p=0.2),
            *common,
        ]
    )


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        num_classes: int,
        size: int = 512,
        train: bool = True,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transforms = segmentation_transforms(size=size, train=train)

        self.images = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not self.images:
            raise ValueError(f"no images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[idx]
        mask_path = self.mask_dir / (image_path.stem + ".png")
        if not mask_path.exists():
            raise FileNotFoundError(f"mask not found for image: {image_path.name}")

        image = read_image(image_path)
        mask = read_mask(mask_path)

        max_cls = int(mask.max())
        min_cls = int(mask.min())
        if min_cls < 0 or max_cls >= self.num_classes:
            raise ValueError(
                f"mask class range invalid at {mask_path}: min={min_cls}, max={max_cls}, num_classes={self.num_classes}"
            )

        transformed = self.transforms(image=image, mask=mask)
        image_t = transformed["image"].float()
        mask_t = transformed["mask"].long()
        return image_t, mask_t


class ClassificationDataset(Dataset):
    def __init__(self, image_dir: Path, csv_path: Path, size: int = 224, train: bool = True) -> None:
        self.image_dir = image_dir
        self.samples = load_classification_csv(csv_path)
        self.transforms = classification_transforms(size=size, train=train)

        if not self.samples:
            raise ValueError(f"no samples in {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, class_id = self.samples[idx]
        image_path = self.image_dir / filename
        image = read_image(image_path)
        transformed = self.transforms(image=image)
        image_t = transformed["image"].float()
        label_t = torch.tensor(class_id, dtype=torch.long)
        return image_t, label_t
