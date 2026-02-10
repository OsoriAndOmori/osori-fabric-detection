#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline:
# 1) Prepare dataset from /Users/1113259/Desktop/sample (wool + cashmere + cashmere2)
# 2) Train classification + segmentation (pseudo masks)
# 3) Export ONNX
# 4) Build Docker image for linux/amd64 or multi-platform and push to GHCR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SOURCE_DIR="/Users/1113259/Desktop/sample"

usage() {
  cat <<USAGE
Usage:
  scripts/train_build_push_ghcr.sh \
    --image ghcr.io/<owner>/<repo> \
    --tag <tag> \
    [--source /path/to/sample] \
    [--platform linux/amd64|linux/amd64,linux/arm64] \
    [--push]

Auth (recommended):
  Option A: env vars
    export GHCR_USERNAME=<github_username>
    export GHCR_TOKEN=<github_pat_with_write:packages>

  Option B: local secret file (auto-loaded, do NOT commit)
    $ROOT_DIR/.secrets/ghcr.env
    ~/.config/fabric_mvp/ghcr.env
    (format: GHCR_USERNAME=... and GHCR_TOKEN=...)

Examples:
  export GHCR_USERNAME=OsoriAndOmori
  export GHCR_TOKEN=...  # DO NOT COMMIT

  # Build + push linux/amd64 only
  scripts/train_build_push_ghcr.sh --image ghcr.io/osoriandomori/osori-fabric-detection --tag latest --platform linux/amd64 --push

  # Multi-platform
  scripts/train_build_push_ghcr.sh --image ghcr.io/osoriandomori/osori-fabric-detection --tag v0.1.0 --platform linux/amd64,linux/arm64 --push
USAGE
}

IMAGE=""
TAG=""
SOURCE_DIR="$DEFAULT_SOURCE_DIR"
PLATFORM="linux/amd64"
DO_PUSH="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --source) SOURCE_DIR="$2"; shift 2;;
    --platform) PLATFORM="$2"; shift 2;;
    --push) DO_PUSH="true"; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ -z "$IMAGE" || -z "$TAG" ]]; then
  echo "ERROR: --image and --tag are required" >&2
  usage
  exit 2
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "ERROR: sample source dir not found: $SOURCE_DIR" >&2
  exit 2
fi

echo "==> Source:   $SOURCE_DIR"
echo "==> Image:    $IMAGE:$TAG"
echo "==> Platform: $PLATFORM"
echo "==> Push:     $DO_PUSH"

# Load GHCR credentials from local secret file if env vars are missing.
load_ghcr_env() {
  local f="$1"
  if [[ -f "$f" ]]; then
    # shellcheck disable=SC1090
    source "$f"
  fi
}

if [[ "${DO_PUSH}" == "true" ]]; then
  if [[ -z "${GHCR_USERNAME:-}" || -z "${GHCR_TOKEN:-}" ]]; then
    mkdir -p "$ROOT_DIR/.secrets" "$HOME/.config/fabric_mvp" || true
    load_ghcr_env "$ROOT_DIR/.secrets/ghcr.env"
    load_ghcr_env "$HOME/.config/fabric_mvp/ghcr.env"
  fi
fi

# Ensure buildx exists
if ! docker buildx inspect >/dev/null 2>&1; then
  docker buildx create --name fabricbuilder --use >/dev/null
fi

docker buildx inspect --bootstrap >/dev/null

# Login if pushing
if [[ "$DO_PUSH" == "true" ]]; then
  if [[ -z "${GHCR_USERNAME:-}" || -z "${GHCR_TOKEN:-}" ]]; then
    echo "ERROR: GHCR credentials missing." >&2
    echo "Set env vars (GHCR_USERNAME/GHCR_TOKEN) or create one of:" >&2
    echo "  $ROOT_DIR/.secrets/ghcr.env" >&2
    echo "  $HOME/.config/fabric_mvp/ghcr.env" >&2
    exit 2
  fi
  echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin >/dev/null
  echo "==> Logged in to GHCR as $GHCR_USERNAME"
fi

# 1) Build base image used for training/export (needs torch)
# Note: training runs locally and produces ONNX + checkpoints.
docker build -f Dockerfile.train -t fabric-mvp:train .

# 2) Prepare dataset (needs OpenCV, so run through local venv if present; fallback to container)
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
  python scripts/prepare_binary_dataset.py \
    --source "$SOURCE_DIR" \
    --output datasets \
    --val-ratio 0.2 \
    --clean \
    --generate-masks
else
  docker run --rm \
    -v "$ROOT_DIR:/app" \
    -v "$SOURCE_DIR:/sample" \
    -w /app \
    -e PYTHONPATH=/app/src \
    fabric-mvp:train \
    python scripts/prepare_binary_dataset.py \
      --source /sample \
      --output datasets \
      --val-ratio 0.2 \
      --clean \
      --generate-masks
fi

# 3) Train (CPU)
docker run --rm \
  -v "$ROOT_DIR:/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  fabric-mvp:train \
  python -m fabric_mvp.training.train_classification \
    --data-root datasets \
    --labels datasets/labels.json \
    --epochs 10 \
    --batch-size 8 \
    --image-size 224 \
    --output checkpoints/classification_best.pt

docker run --rm \
  -v "$ROOT_DIR:/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  fabric-mvp:train \
  python -m fabric_mvp.training.train_segmentation \
    --data-root datasets \
    --labels datasets/labels.json \
    --epochs 4 \
    --batch-size 4 \
    --image-size 128 \
    --output checkpoints/segmentation_best.pt

# 4) Export ONNX
mkdir -p exports

docker run --rm \
  -v "$ROOT_DIR:/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  fabric-mvp:train \
  python -m fabric_mvp.training.export_onnx \
    --task classification \
    --weights checkpoints/classification_best.pt \
    --labels datasets/labels.json \
    --output exports/classification.onnx

docker run --rm \
  -v "$ROOT_DIR:/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  fabric-mvp:train \
  python -m fabric_mvp.training.export_onnx \
    --task segmentation \
    --weights checkpoints/segmentation_best.pt \
    --labels datasets/labels.json \
    --output exports/segmentation.onnx \
    --image-size 512

# 5) Build and push runtime image
# This build uses .dockerignore to avoid copying datasets/train|val.
EXTRA_TAGS=()
if [[ "$TAG" != "latest" ]]; then
  EXTRA_TAGS+=("-t" "$IMAGE:latest")
fi

if [[ "$DO_PUSH" == "true" ]]; then
  docker buildx build \
    --platform "$PLATFORM" \
    -t "$IMAGE:$TAG" \
    -t "$IMAGE:sha-$(git rev-parse --short HEAD)" \
    "${EXTRA_TAGS[@]}" \
    --push \
    .
  echo "==> Pushed: $IMAGE:$TAG"
else
  docker buildx build \
    --platform "$PLATFORM" \
    -t "$IMAGE:$TAG" \
    "${EXTRA_TAGS[@]}" \
    --load \
    .
  echo "==> Built locally (loaded): $IMAGE:$TAG"
fi

echo "==> Done"
