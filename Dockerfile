# Dockerfile - With execstack fix
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including execstack
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    make \
    cmake \
    python3-dev \
    wget \
    execstack \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install --no-cache-dir onnxruntime==1.15.1 && \
    pip install --no-cache-dir --no-deps insightface==0.7.3 && \
    pip install --no-cache-dir fastapi uvicorn python-multipart python-dotenv httpx scikit-learn Pillow cryptography

# Fix ONNX Runtime executable stack issue
RUN find /usr/local/lib/python3.10/site-packages/onnxruntime/capi -name "*.so" -exec execstack -c {} \; 2>/dev/null || true

COPY . .

RUN mkdir -p models && mkdir -p /root/.insightface

# Pre-download insightface model
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')" 2>/dev/null || true

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]