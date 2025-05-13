# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements 
COPY requirements.txt .

# Install necessary Python dependencies
#add install boto3 for S3 bucket operations with MinIO
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install boto3 prometheus-fastapi-instrumentator 

# Create a non-root user
RUN useradd -m appuser


# Copy the application code
COPY app.py .
COPY qwen_vl_utils.py .

# Create directory for the  model checkpoints
RUN mkdir -p /app/models && \
    mkdir -p /app/logs

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables for model configuration
ENV CHECKPOINT_PATH=/app/models/checkpoint-600 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/app

# Expose the FastAPI port
EXPOSE 8080

# Health check 
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the FastAPI application
CMD ["python3", "app.py"]

