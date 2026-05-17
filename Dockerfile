# Dockerfile - Production Ready for Render
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including execstack tool
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    python3-dev \
    wget \
    execstack \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install --no-cache-dir onnxruntime==1.14.0 && \
    pip install --no-cache-dir --no-deps insightface==0.7.3 && \
    pip install --no-cache-dir fastapi uvicorn python-multipart python-dotenv httpx scikit-learn Pillow cryptography

# CRITICAL FIX: Disable executable stack for ONNX Runtime shared libraries
RUN find /usr/local/lib/python3.10/site-packages/onnxruntime/capi -name "*.so" -exec execstack -c {} \; 2>/dev/null || true

# Also fix any insightface compiled libraries
RUN find /usr/local/lib/python3.10/site-packages/insightface -name "*.so" -exec execstack -c {} \; 2>/dev/null || true

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models && mkdir -p /root/.insightface

# Download model at build time (optional, avoids runtime download)
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')" 2>/dev/null || echo "Model will download at runtime"

# Run as non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]