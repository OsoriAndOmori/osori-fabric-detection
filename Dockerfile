FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.runtime.txt ./
RUN pip install --upgrade pip && pip install -r requirements.runtime.txt

COPY src ./src
COPY scripts ./scripts
# Runtime only needs class labels, not the (potentially large) training dataset.
COPY datasets/labels.json ./datasets/labels.json
COPY exports ./exports
COPY checkpoints ./checkpoints
COPY README.md ./README.md

EXPOSE 80

# Render typically injects PORT; default to 80 for local runs.
CMD ["sh", "-lc", "uvicorn fabric_mvp.api.main:app --host 0.0.0.0 --port ${PORT:-80}"]
