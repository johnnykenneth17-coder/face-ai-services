# Dockerfile - Production Ready for Render
FROM python:3.10-slim

WORKDIR /app

# Install only essential build tools (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    python3-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install packages in correct order (NO --no-binary flag)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install --no-cache-dir onnxruntime==1.14.0 && \
    pip install --no-cache-dir --no-deps insightface==0.7.3 && \
    pip install --no-cache-dir fastapi uvicorn python-multipart python-dotenv httpx Pillow

# Fix ONNX Runtime executable stack issue (CRITICAL)
RUN find /usr/local/lib/python3.10/site-packages/onnxruntime/capi -name "*.so" -exec execstack -c {} \; 2>/dev/null || \
    (apt-get update && apt-get install -y execstack && \
     find /usr/local/lib/python3.10/site-packages/onnxruntime/capi -name "*.so" -exec execstack -c {} \; && \
     apt-get remove -y execstack && apt-get autoremove -y)

# Copy application
COPY . .

# Create directories
RUN mkdir -p models && mkdir -p /root/.insightface

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]