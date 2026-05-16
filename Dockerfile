# Dockerfile - Using micromamba (handles compilers automatically)
FROM mambaorg/micromamba:1.5.1

USER root
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create environment with insightface
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

COPY . .

ENV PATH="/opt/conda/bin:${PATH}"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]