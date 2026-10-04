# syntax=docker/dockerfile:1

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build-time OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="edge-analytics-team"
LABEL version="0.3.0"
LABEL description="Semantic-Aware Edge Video Analytics Service"

WORKDIR /app

# Runtime OS libraries required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY requirements.txt ./

# Pre-download YOLOv8 nano weights at build time so the model is
# cached in the image layer and doesn't require network at runtime.
RUN python - <<'PYEOF'
from ultralytics import YOLO
YOLO("yolov8n.pt")
PYEOF

# Non-root user for production security
RUN useradd -m -u 1001 appuser && chown -R appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Entrypoint: production-grade uvicorn with 2 workers
ENTRYPOINT ["python", "-m", "uvicorn", "src.server:app", \
    "--host", "0.0.0.0", "--port", "8000", \
    "--workers", "2", "--log-level", "info"]
