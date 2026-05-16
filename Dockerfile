# Dockerfile - Using pre-built image with OpenCV dependencies
FROM python:3.10-slim

# Skip apt-get entirely - use pre-installed wheels
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python packages (OpenCV will use pre-compiled wheels)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p models ~/.insightface

# Run with reduced memory usage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--limit-concurrency", "10"]