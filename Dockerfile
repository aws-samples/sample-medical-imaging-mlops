# MedML Training Platform Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs/tensorboard data/raw data/processed data/models

# Expose ports
EXPOSE 8000 5000 6006

# Create startup script
RUN echo '#!/bin/bash\n\
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db &\n\
tensorboard --logdir=./logs/tensorboard --port=6006 --host=0.0.0.0 &\n\
python main.py' > start_docker.sh && chmod +x start_docker.sh

# Add non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

# Start the application
CMD ["./start_docker.sh"]
