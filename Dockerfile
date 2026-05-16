# Dockerfile - Corrected package names
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies with corrected package names
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    make \
    cmake \
    python3-dev \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgomp1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages in specific order
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

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]