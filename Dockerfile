FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Ensure src is importable
ENV PYTHONPATH=/app

# Download real incident datasets (fallback to synthetic if unavailable)
RUN python scripts/prepare_datasets.py || true

# Generate task JSON files from real logpai/loghub data
# Falls back to keeping existing hardcoded tasks if download fails
RUN python scripts/generate_tasks_from_loghub.py || true

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Gradio mounted at / handles HF Spaces iframe correctly
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
