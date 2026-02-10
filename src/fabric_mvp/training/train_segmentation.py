from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from fabric_mvp.data.datasets import SegmentationDataset
from fabric_mvp.data.schema import load_labels
from fabric_mvp.models.unet import UNet
from fabric_mvp.training.common import EarlyStopping, save_checkpoint


def compute_mean_iou(logits: torch.Tensor, masks: torch.Tensor, num_classes: int) -> float:
    preds = torch.argmax(logits, dim=1)
    ious = []
    for cls in range(1, num_classes):
        pred_cls = preds == cls
        mask_cls = masks == cls
        inter = torch.logical_and(pred_cls, mask_cls).sum().item()
        union = torch.logical_or(pred_cls, mask_cls).sum().item()
        if union == 0:
            continue
        ious.append(inter / (union + 1e-6))
    return float(np.mean(ious)) if ious else 0.0


def run_epoch(model, loader, criterion, device, optimizer=None, num_classes=2):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    count = 0

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_iou += compute_mean_iou(logits.detach(), masks, num_classes=num_classes)
        count += 1

    return total_loss / max(1, count), total_iou / max(1, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/labels.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/segmentation_best.pt"))
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classes = load_labels(args.labels)
    num_classes = len(classes)

    train_ds = SegmentationDataset(
        image_dir=args.data_root / "train/images",
        mask_dir=args.data_root / "train/masks",
        num_classes=num_classes,
        size=args.image_size,
        train=True,
    )
    val_ds = SegmentationDataset(
        image_dir=args.data_root / "val/images",
        mask_dir=args.data_root / "val/masks",
        num_classes=num_classes,
        size=args.image_size,
        train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=2)

    device = torch.device(args.device)
    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    stopper = EarlyStopping(patience=args.patience)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = run_epoch(model, train_loader, criterion, device, optimizer, num_classes)
        val_loss, val_iou = run_epoch(model, val_loader, criterion, device, None, num_classes)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_mIoU={train_iou:.4f} "
            f"val_loss={val_loss:.4f} val_mIoU={val_iou:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mIoU": train_iou,
                "val_loss": val_loss,
                "val_mIoU": val_iou,
            }
        )

        if val_iou >= stopper.best:
            save_checkpoint(
                args.output,
                model,
                {
                    "classes": classes,
                    "num_classes": num_classes,
                    "image_size": args.image_size,
                    "val_mIoU": val_iou,
                },
            )

        if stopper.step(val_iou):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    log_path = args.output.parent / "segmentation_train_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Saved checkpoint: {args.output}")
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
