# Docker Deployment Guide

## Quick Start

### Build and Run Baseline Model
```bash
# Build the production image
docker build --target production -t bird-classifier:latest .

# Run baseline classification
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs bird-classifier:latest
```

### Development Environment with Jupyter
```bash
# Start Jupyter Lab
docker-compose up jupyter

# Access at: http://localhost:8888
# Token: (none - open access for local development)
```

### Full Training Pipeline
```bash
# Build and run training
docker-compose up trainer

# Monitor logs
docker-compose logs -f trainer
```

## Docker Compose Services

### `bird-classifier` (Production)
- Runs baseline model inference
- Read-only data mount
- Outputs to `./outputs/`

### `jupyter` (Development)
- Jupyter Lab environment
- Port: 8888
- All volumes mounted for development

### `trainer` (Training)
- Full ensemble training pipeline
- Resource limits: 4 CPUs, 8GB RAM
- Saves models to `./models/`

## Advanced Usage

### Custom Commands
```bash
# Run specific script
docker run -v $(pwd)/data:/app/data bird-classifier:latest python run_pipeline.py

# Interactive shell
docker run -it bird-classifier:latest /bin/bash

# Test safety controller
docker run bird-classifier:latest python src/safety_controller.py
```

### Resource Management
```bash
# Limit CPU and memory
docker run --cpus=2 --memory=4g bird-classifier:latest

# Use GPU (if available)
docker run --gpus all bird-classifier:latest
```

## Production Deployment

### Cloud Deployment (AWS ECS, GCP Cloud Run, Azure Container Instances)
```bash
# Tag for registry
docker tag bird-classifier:latest your-registry.io/bird-classifier:v1.0

# Push to registry
docker push your-registry.io/bird-classifier:v1.0
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bird-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bird-classifier
  template:
    metadata:
      labels:
        app: bird-classifier
    spec:
      containers:
      - name: classifier
        image: bird-classifier:latest
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
```

## Image Sizes
- Base image: ~1.2GB
- Production: ~1.3GB
- Development (with Jupyter): ~1.5GB

## Security Notes
- Development Jupyter runs without token (use only locally)
- Data mounted as read-only in production
- No credentials stored in image
- Use secrets management for production deployments
