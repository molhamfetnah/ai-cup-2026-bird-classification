# AI Cup 2026 Bird Classification - Production Docker Image
# Multi-stage build for optimized image size

FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY run_baseline.py run_pipeline.py ./

# Create directories for data and outputs
RUN mkdir -p data outputs models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Production stage
FROM base as production

# Add metadata
LABEL maintainer="Mulham Fetaineh"
LABEL description="Bird Classification System for Wind Turbine Safety"
LABEL version="1.0"

# Default command - run baseline model
CMD ["python", "run_baseline.py"]

# Development stage with jupyter
FROM base as development

# Install development tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    matplotlib

# Copy notebooks
COPY notebooks/ ./notebooks/

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Training stage
FROM base as training

# Copy data directory structure
COPY data/ ./data/

# Run full training pipeline
CMD ["python", "run_pipeline.py"]
