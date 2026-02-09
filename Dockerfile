# =============================================================================
# Stage 1: Builder - install dependencies and download models
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# System dependencies for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Use CPU-only torch (avoids 2GB+ CUDA libs)
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Install torch CPU-only first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install pinned dependencies from requirements.txt for reproducible builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and install package (--no-deps since deps already installed)
# Note: pyproject.toml is copied last to maximize layer caching. If only
# pyproject.toml changes (e.g., version bump), only this layer rebuilds.
COPY pyproject.toml .
COPY sage/ sage/
RUN pip install --no-cache-dir . --no-deps

# Pre-download models to cache directory
ENV HF_HOME=/app/.cache/huggingface

# Download E5-small embedding model (~134MB)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/e5-small-v2')"

# Download HHEM hallucination detection model (~892MB)
# HHEM uses custom config pointing to foundation T5 model for tokenizer
RUN python -c "\
from transformers import AutoConfig, AutoTokenizer; \
from huggingface_hub import hf_hub_download; \
config = AutoConfig.from_pretrained('vectara/hallucination_evaluation_model', trust_remote_code=True); \
AutoTokenizer.from_pretrained(config.foundation); \
AutoConfig.from_pretrained(config.foundation); \
hf_hub_download('vectara/hallucination_evaluation_model', 'model.safetensors')"


# =============================================================================
# Stage 2: Runtime - slim image with only what's needed
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Only curl for healthcheck (no build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user with UID 1000 (required by HF Spaces)
RUN useradd -m -u 1000 user

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/sage /app/sage

# Copy pre-downloaded models from builder
COPY --from=builder /app/.cache /app/.cache

# Environment
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Fix ownership for non-root user
RUN chown -R user:user /app

USER user

# Default port 7860 for HF Spaces; overridden by PORT env var at runtime
ENV PORT=7860
EXPOSE 7860

# Health check with startup grace period (models take ~30s to load)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:${PORT:-7860}/health || exit 1

CMD ["python", "-m", "sage.api.run", "--host", "0.0.0.0"]
