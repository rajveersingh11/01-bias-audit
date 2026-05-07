FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.cache/torch

# Install system dependencies
# - libgomp: OpenMP runtime for torch
# - libopenblas: Linear algebra for numpy/scipy
# - git: For potential git-based operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
COPY config/ ./config/
COPY main.py params.yaml dvc.yaml ./

# Create necessary directories
RUN mkdir -p data/raw data/processed outputs/model outputs/tokenizer outputs/metrics outputs/cache outputs/reports/charts logs

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Health check - verify imports work
RUN python -c "from src import get_logger; print('✓ Import check passed')"

# Default command runs the full pipeline
CMD ["python", "main.py"]

# Alternative: Run DVC pipeline
# CMD ["dvc", "repro"]

# Alternative: Interactive shell
# CMD ["/bin/bash"]

LABEL maintainer="Portfolio" \
      description="Bias Audit: ML Fairness Evaluation Pipeline" \
      version="0.1.0"
