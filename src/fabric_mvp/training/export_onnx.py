from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fabric_mvp.data.schema import load_labels
from fabric_mvp.models.classifier import build_efficientnet
from fabric_mvp.models.unet import UNet


def export_segmentation(weights: Path, labels_path: Path, output: Path, image_size: int = 512) -> None:
    classes = load_labels(labels_path)
    model = UNet(in_channels=3, num_classes=len(classes))
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported segmentation ONNX: {output}")


def export_classification(weights: Path, labels_path: Path, output: Path, image_size: int = 224) -> None:
    classes = load_labels(labels_path)
    model = build_efficientnet(num_classes=len(classes) - 1, pretrained=False)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported classification ONNX: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["segmentation", "classification"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("datasets/labels.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.task == "segmentation":
        export_segmentation(args.weights, args.labels, args.output, image_size=args.image_size or 512)
    else:
        export_classification(args.weights, args.labels, args.output, image_size=args.image_size or 224)


if __name__ == "__main__":
    main()
