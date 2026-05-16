# Dockerfile - Complete working version
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    make \
    cmake \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages with specific versions to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install --no-cache-dir insightface==0.7.3 onnxruntime==1.15.1 && \
    pip install --no-cache-dir fastapi==0.104.1 uvicorn[standard]==0.24.0 && \
    pip install --no-cache-dir python-multipart python-dotenv httpx scikit-learn Pillow && \
    pip install --no-cache-dir cryptography==41.0.7

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p models && mkdir -p /root/.insightface

# Expose port
EXPOSE 8001

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]