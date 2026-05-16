# Dockerfile - Step-by-step installation
FROM python:3.10-slim

WORKDIR /app

# Install numpy first (critical for insightface)
RUN pip install --no-cache-dir numpy==1.23.5

# Install OpenCV
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Install insightface and its dependencies
RUN pip install --no-cache-dir insightface==0.7.3 onnxruntime==1.15.1

# Install remaining packages
RUN pip install --no-cache-dir fastapi==0.104.1 uvicorn[standard]==0.24.0
RUN pip install --no-cache-dir python-multipart python-dotenv httpx scikit-learn Pillow

# Copy application
COPY . .

RUN mkdir -p models ~/.insightface

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]