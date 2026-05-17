# Dockerfile - Using ONNX Runtime CPU-only version
FROM python:3.10-slim

WORKDIR /app

# Minimal dependencies (no execstack needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install with CPU-only ONNX Runtime (no executable stack issues)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install --no-cache-dir --no-deps insightface==0.7.3 && \
    pip install --no-cache-dir fastapi uvicorn python-multipart python-dotenv httpx scikit-learn Pillow cryptography

# Install ONNX Runtime CPU-only from alternative source
RUN pip install --no-cache-dir onnxruntime==1.14.0 --no-binary=onnxruntime

COPY . .

RUN mkdir -p models && mkdir -p /root/.insightface

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]