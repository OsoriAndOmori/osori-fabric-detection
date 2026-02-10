# Fabric Detection MVP (Wool vs Cashmere)

Python Docker 단일 이미지로 운영 가능한 MVP입니다.

- 입력: 현미경/확대 섬유 이미지 1장
- 출력:
  - Top 분류(`wool`, `cashmere`)
  - 혼용률(%) 추정 (Segmentation 픽셀 비율 기반, fallback은 tile 분류 분포)
  - QA 기반 `retake` 플래그
- UI:
  - API: `/predict`
  - Gradio 웹 UI: `/ui`
  - 오답 정답 라벨 저장:
    - Local: `/tmp/fabric_mvp_feedback/*` (Render Free에서는 영구 저장 안됨)
    - GitHub: private repo로 업로드(권장)

## Project Tree

```text
/Users/1113259/IdeaProjects/osori-fabric-detection
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── scripts/
│   ├── apply_feedback_to_dataset.py
│   ├── generate_synthetic_data.py
│   └── prepare_binary_dataset.py
├── src/fabric_mvp/
│   ├── api/main.py
│   ├── config.py
│   ├── qa.py
│   ├── inference/predictor.py
│   ├── models/
│   │   ├── classifier.py
│   │   └── unet.py
│   ├── training/
│   │   ├── train_classification.py
│   │   ├── train_segmentation.py
│   │   └── export_onnx.py
│   └── ui/gradio_ui.py
└── tests/
    ├── test_api.py
    └── test_qa.py
```

## 1) 샘플 폴더로 데이터셋 준비

사용자 데이터 구조:

```text
/Users/1113259/Desktop/sample/
├── wool/
├── cashmere/
└── cashmere2/    # 있으면 자동 포함
```

학습용 포맷 생성:

```bash
cd /Users/1113259/IdeaProjects/osori-fabric-detection
python3 scripts/prepare_binary_dataset.py \
  --source /Users/1113259/Desktop/sample \
  --output datasets \
  --val-ratio 0.2 \
  --clean \
  --generate-masks
```

결과:
- `datasets/train/images/*`
- `datasets/train/labels.csv` (`filename,class`)
- `datasets/train/masks/*.png` (`0=background, 1=wool, 2=cashmere`)
- `datasets/val/images/*`
- `datasets/val/labels.csv`
- `datasets/val/masks/*.png`
- `datasets/labels.json` = `{"classes": ["background", "wool", "cashmere"]}`

`--generate-masks`는 이미지에서 섬유 전경을 자동 추정한 pseudo mask를 만듭니다(빠른 MVP용).  
기본값으로 `cashmere,cashmere2` 폴더를 합쳐 cashmere 클래스로 학습합니다.

## 2) Docker로 학습 (Python 3.11 고정)

### 2-1. 이미지 빌드

```bash
docker build -t fabric-mvp:local .
```

### 2-2. 분류 학습

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  fabric-mvp:local \
  python -m fabric_mvp.training.train_classification \
    --data-root datasets \
    --labels datasets/labels.json \
    --epochs 8 \
    --batch-size 8 \
    --image-size 224 \
    --output checkpoints/classification_best.pt
```

### 2-3. Segmentation(U-Net) 학습

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  fabric-mvp:local \
  python -m fabric_mvp.training.train_segmentation \
    --data-root datasets \
    --labels datasets/labels.json \
    --epochs 3 \
    --batch-size 4 \
    --image-size 128 \
    --output checkpoints/segmentation_best.pt
```

### 2-4. ONNX export

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  fabric-mvp:local \
  python -m fabric_mvp.training.export_onnx \
    --task segmentation \
    --weights checkpoints/segmentation_best.pt \
    --labels datasets/labels.json \
    --output exports/segmentation.onnx

docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  fabric-mvp:local \
  python -m fabric_mvp.training.export_onnx \
    --task classification \
    --weights checkpoints/classification_best.pt \
    --labels datasets/labels.json \
    --output exports/classification.onnx
```

## 3) 실행 (API + UI)

모델이 생성된 상태에서 다시 빌드하면(`exports/`, `checkpoints/` 포함) 단일 이미지 배포 가능:

```bash
docker build -t fabric-mvp:runtime .
docker run --rm -p 8080:80 \
  -e FABRIC_BACKEND=auto \
  fabric-mvp:runtime
```

접속:
- API health: `http://127.0.0.1:8080/health`
- Gradio UI: `http://127.0.0.1:8080/ui`
- UI에서 `Show evidence overlay/masks`를 켜면(메모리 더 사용) 오버레이(근거)와 클래스별 마스크를 확인 가능
- 예측이 틀리면 정답(`wool/cashmere`)을 선택해 피드백 저장 가능(아래 참고)

샘플 API 호출:

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
  -H "accept: application/json" \
  -F "image=@datasets/val/images/wool_0000.jpg"
```

## 4) Render 배포 (ARM 맥북 -> AMD64 이미지)

Render 런타임은 Linux/amd64 기준으로 생각하면 안전합니다. Mac ARM에서 `buildx`로 amd64 이미지를 빌드/푸시하세요.

```bash
# 1) buildx 준비
docker buildx create --name fabricbuilder --use || true
docker buildx inspect --bootstrap

# 2) amd64 이미지 빌드 + 레지스트리 푸시
docker buildx build \
  --platform linux/amd64 \
  -t <YOUR_REGISTRY>/fabric-mvp:0.1.0 \
  --push \
  .
```

Render 설정:
- Service type: Web Service
- Runtime: Docker image
- Image URL: `<YOUR_REGISTRY>/fabric-mvp:0.1.0`
- Port: Render가 주는 `PORT`를 그대로 사용 (Docker CMD가 `${PORT:-80}` 사용)
- Start command: 기본 Docker `CMD` 사용
- Env:
  - `FABRIC_BACKEND=auto`
  - (권장) 메모리 안정화:
    - `FABRIC_MAX_INPUT_SIDE=1200`
    - `FABRIC_MAX_DISPLAY_SIDE=768`
  - (선택) 피드백을 GitHub로 저장(Render Free 권장):
    - `FABRIC_FEEDBACK_BACKEND=github`
    - `FABRIC_FEEDBACK_GITHUB_REPO=<owner>/<repo>`
    - `FABRIC_FEEDBACK_GITHUB_TOKEN=<PAT>`
    - `FABRIC_FEEDBACK_GITHUB_BRANCH=main`

## 8) GitHub Actions 워크플로우

1. `/.github/workflows/merge-develop-to-main.yml`
- 수동 실행(`workflow_dispatch`)
- `develop` 브랜치를 `main`으로 merge 후 push

2. `/.github/workflows/docker-ghcr.yml`
- `main` 브랜치 push 시 자동 실행
- GHCR(`ghcr.io/<owner>/<repo>`)로 멀티아키텍처 이미지 push

## 9) 피드백 반영 재학습

Local backend(`FABRIC_FEEDBACK_BACKEND=local`)에서 저장된 피드백을 train 데이터에 반영:

```bash
python3 scripts/apply_feedback_to_dataset.py \
  --feedback /tmp/fabric_mvp_feedback/feedback.jsonl \
  --data-root datasets \
  --labels datasets/labels.json
```

그 다음 분류/세그멘테이션을 다시 학습하면 됩니다.

Render Free는 persistent disk가 없어서 local feedback은 재시작 시 사라질 수 있습니다.  
Render에서는 GitHub backend로 올리고, 로컬에서 pull 받은 뒤 반영하는 흐름을 권장합니다.

## 10) 로컬에서 학습 + GHCR 푸시(amd64/멀티플랫폼)

한 번에 처리하는 스크립트:

```bash
# Option A: env vars
export GHCR_USERNAME=OsoriAndOmori
export GHCR_TOKEN=...   # write:packages 필요

# Option B: local secret file (recommended, do NOT commit)
mkdir -p .secrets
cat > .secrets/ghcr.env <<'EOF'
GHCR_USERNAME=OsoriAndOmori
GHCR_TOKEN=...
EOF
chmod 600 .secrets/ghcr.env

# linux/amd64로 빌드해서 GHCR push
scripts/train_build_push_ghcr.sh \
  --image ghcr.io/osoriandomori/osori-fabric-detection \
  --tag latest \
  --platform linux/amd64 \
  --push

# 멀티플랫폼(amd64+arm64)
scripts/train_build_push_ghcr.sh \
  --image ghcr.io/osoriandomori/osori-fabric-detection \
  --tag v0.1.0 \
  --platform linux/amd64,linux/arm64 \
  --push
```

## 5) QA 기준

- Blur: Laplacian variance (`blur_threshold` 기본 80.0)
- Exposure: 과다 암/과다 명 비율
- Resolution: 기본 최소 256x256
- 하나라도 실패 시 `retake=true`

조정 가능한 env:
- `FABRIC_MIN_WIDTH`, `FABRIC_MIN_HEIGHT`
- `FABRIC_BLUR_THRESHOLD`
- `FABRIC_DARK_PIXEL_RATIO_THRESHOLD`
- `FABRIC_BRIGHT_PIXEL_RATIO_THRESHOLD`

## 5-1) 환경변수 전체 목록

모든 환경변수는 prefix `FABRIC_`를 사용합니다. (예: `FABRIC_BACKEND=auto`)

| Env | Default | Description | When To Set |
|---|---:|---|---|
| `FABRIC_ENABLE_GRADIO_UI` | `true` | `/ui` Gradio UI 마운트 여부 (`true/false`) | API만 운영 시 `false` |
| `FABRIC_BACKEND` | `auto` | 추론 백엔드 (`auto|onnx|torch`) | 배포는 `auto` 권장 |
| `FABRIC_LABELS_PATH` | `datasets/labels.json` | 클래스 라벨 파일 경로 | 커스텀 labels 사용 시 |
| `FABRIC_SEGMENTATION_MODEL_PATH` | `checkpoints/segmentation_best.pt` | Segmentation(torch) 체크포인트 경로 | torch 백엔드 강제 시 |
| `FABRIC_CLASSIFICATION_MODEL_PATH` | `checkpoints/classification_best.pt` | Classification(torch) 체크포인트 경로 | torch 백엔드 강제 시 |
| `FABRIC_SEGMENTATION_ONNX_PATH` | `exports/segmentation.onnx` | Segmentation(ONNX) 모델 경로 | ONNX 위치 변경 시 |
| `FABRIC_CLASSIFICATION_ONNX_PATH` | `exports/classification.onnx` | Classification(ONNX) 모델 경로 | ONNX 위치 변경 시 |
| `FABRIC_MAX_INPUT_SIDE` | `1600` | 입력/ROI long-side가 이 값 초과 시 축소 후 추론(메모리 안정화) | Render Free 권장 `1200` |
| `FABRIC_MAX_DISPLAY_SIDE` | `1024` | UI로 반환하는 프리뷰/오버레이/마스크 long-side 캡(메모리/대역폭 절감) | Render Free 권장 `768` |
| `FABRIC_MIN_WIDTH` | `256` | QA 최소 너비 | 현미경 이미지가 작으면 조정 |
| `FABRIC_MIN_HEIGHT` | `256` | QA 최소 높이 | 현미경 이미지가 작으면 조정 |
| `FABRIC_BLUR_THRESHOLD` | `80.0` | QA blur 기준(Laplacian variance) | 카메라/배율 바뀌면 튜닝 |
| `FABRIC_DARK_PIXEL_RATIO_THRESHOLD` | `0.55` | QA 노출(너무 어두움) 픽셀 비율 기준 | 조명 조건 바뀌면 튜닝 |
| `FABRIC_BRIGHT_PIXEL_RATIO_THRESHOLD` | `0.55` | QA 노출(너무 밝음) 픽셀 비율 기준 | 조명 조건 바뀌면 튜닝 |
| `FABRIC_FEEDBACK_BACKEND` | `local` | 피드백 저장 방식 (`local|github|disabled`) | Render Free는 `github` 권장 |
| `FABRIC_FEEDBACK_DIR` | `/tmp/fabric_mvp_feedback` | `local` 백엔드 저장 디렉토리 | 로컬 개발 시 커스텀 |
| `FABRIC_FEEDBACK_GITHUB_REPO` | (unset) | GitHub 저장 repo (`owner/repo`) | `FABRIC_FEEDBACK_BACKEND=github`일 때 필수 |
| `FABRIC_FEEDBACK_GITHUB_TOKEN` | (unset) | GitHub PAT (repo write 권한) | `github`일 때 필수 (Secret로 설정) |
| `FABRIC_FEEDBACK_GITHUB_BRANCH` | `main` | GitHub 업로드 대상 브랜치 | 필요 시 변경 |
| `FABRIC_FEEDBACK_GITHUB_PATH_PREFIX` | `feedback` | GitHub 업로드 경로 prefix | 피드백 경로 구조 변경 시 |

추가로 플랫폼 공통 env:

| Env | Default | Description | When To Set |
|---|---:|---|---|
| `PORT` | (platform injected) | Render가 주는 서비스 포트. Docker CMD가 `${PORT:-80}` 사용 | Render에서 자동 |

## 6) 테스트

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  -e PYTHONPATH=/app/src \
  fabric-mvp:local pytest -q
```

## 7) 트러블슈팅

1. `labels.csv` 형식 오류
- 반드시 `filename,class` 헤더 사용

2. 클래스 폴더명 오타
- `cashmere` 폴더명을 정확히 사용 (`cachemere` 아님)

3. 모델이 로드되지 않고 `model_version: 0.1.0-no-model`
- `exports/classification.onnx` 파일 경로/ENV 확인
- 학습 후 재빌드하여 이미지 내부에 모델 포함

4. ARM 맥에서 빌드했는데 Render 실행 실패
- `docker buildx build --platform linux/amd64`로 다시 빌드/푸시
