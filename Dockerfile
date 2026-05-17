# Dockerfile - Complete working version
FROM python:3.10-slim

WORKDIR /app

# Install minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    python3-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install all packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78
RUN pip install --no-cache-dir onnxruntime==1.14.0
RUN pip install --no-cache-dir --no-deps insightface==0.7.3
RUN pip install --no-cache-dir fastapi uvicorn python-multipart python-dotenv httpx scikit-learn Pillow cryptography

# Copy application
COPY . .

# Create directories
RUN mkdir -p models && mkdir -p /root/.insightface

# Pre-download model
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')" 2>/dev/null || echo "Model download skipped"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]