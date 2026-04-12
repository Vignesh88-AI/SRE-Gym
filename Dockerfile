FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Ensure src is importable
ENV PYTHONPATH=/app

EXPOSE 7860

# Health check — automated validator pings /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Use uvicorn directly for reliable startup on HF Spaces
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7860"]
