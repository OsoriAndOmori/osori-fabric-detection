from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from fabric_mvp.data.datasets import ClassificationDataset
from fabric_mvp.data.schema import load_labels
from fabric_mvp.models.classifier import build_efficientnet
from fabric_mvp.training.common import EarlyStopping, save_checkpoint


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
        total_loss += loss.item()

    return total_loss / max(1, len(loader)), correct / max(1, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/labels.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/classification_best.pt"))
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = load_labels(args.labels)
    # classification uses non-background classes only
    class_names = classes[1:]
    num_classes = len(class_names)

    train_ds = ClassificationDataset(
        image_dir=args.data_root / "train/images",
        csv_path=args.data_root / "train/labels.csv",
        size=args.image_size,
        train=True,
    )
    val_ds = ClassificationDataset(
        image_dir=args.data_root / "val/images",
        csv_path=args.data_root / "val/labels.csv",
        size=args.image_size,
        train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=2)

    device = torch.device(args.device)
    model = build_efficientnet(num_classes=num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    stopper = EarlyStopping(patience=args.patience)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, None)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc >= stopper.best:
            save_checkpoint(
                args.output,
                model,
                {
                    "classes": class_names,
                    "num_classes": num_classes,
                    "image_size": args.image_size,
                    "val_acc": val_acc,
                },
            )

        if stopper.step(val_acc):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    log_path = args.output.parent / "classification_train_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Saved checkpoint: {args.output}")
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
