# Dockerfile - Using pre-built base image
FROM python:3.10-slim

WORKDIR /app

# Install only runtime dependencies (no build tools needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements with pre-built wheels
COPY requirements.txt .

# Install packages (insightface will use pre-built wheel from PyPI)
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]