FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install CPU-only PyTorch (small, no CUDA)
#    Pin to a sane version; adjust if needed.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1

# 2) Install Python deps (sentence-transformers etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) App code
COPY app ./app

ENV MODEL_NAME=all-MiniLM-L6-v2 \
    LOG_LEVEL=INFO

EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:8000", "app.main:app"]
