# MedML Training Platform Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

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

# Start the application
CMD ["./start_docker.sh"]