# Dockerfile - With build tools for InsightFace
FROM python:3.10-slim

WORKDIR /app

# Install build tools and dependencies (REQUIRED for insightface)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    make \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Copy requirements
COPY requirements.txt .

# Install Python packages (build tools already installed)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=appuser:appuser . .

RUN mkdir -p models ~/.insightface

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]