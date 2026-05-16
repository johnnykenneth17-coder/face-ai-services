# Dockerfile - Production tested
FROM python:3.10-slim

WORKDIR /app

# Install minimal build tools (no OpenCV system deps needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    make \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .

# Install with --no-deps for insightface to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78
RUN pip install --no-cache-dir onnxruntime==1.15.1
RUN pip install --no-cache-dir --no-deps insightface==0.7.3
RUN pip install --no-cache-dir fastapi uvicorn python-multipart python-dotenv httpx scikit-learn Pillow cryptography

COPY . .

# Pre-download insightface model
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')" 2>/dev/null || true

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]