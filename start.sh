#!/bin/bash

# MedML Training Platform Startup Script
# Usage:
#   ./start.sh          # Start the app (skip Docker builds)
#   ./start.sh --build  # Build and push Docker images, then start

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_IMAGES=false

if [[ "${1:-}" == "--build" ]]; then
    BUILD_IMAGES=true
fi

echo "🚀 Starting MedML Training Platform..."

# Load .env.local if present
if [[ -f "$SCRIPT_DIR/.env.local" ]]; then
    set -a
    source "$SCRIPT_DIR/.env.local"
    set +a
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
if command -v uv &>/dev/null; then
    uv pip install --quiet -r requirements.txt
else
    python -m ensurepip --upgrade 2>/dev/null || true
    python -m pip install --quiet -r requirements.txt
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs/tensorboard
mkdir -p data/raw data/processed data/models

# Build Docker images only if --build flag is passed
if [[ "$BUILD_IMAGES" == true ]]; then
    if [[ -n "${AWS_ACCOUNT_ID:-}" ]]; then
        AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
        ECR_PREFIX="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

        echo "🐳 Building and pushing Docker images..."

        # Login to AWS DLC registry (base images)
        echo "   Authenticating with AWS DLC registry..."
        aws ecr get-login-password --region us-east-1 \
          | docker login --username AWS --password-stdin \
            763104351884.dkr.ecr.us-east-1.amazonaws.com 2>/dev/null

        echo "   Building training image (classification)..."
        "$SCRIPT_DIR/build_and_push.sh" classification

        echo "   Building training image (segmentation)..."
        "$SCRIPT_DIR/build_and_push.sh" segmentation

        echo "   Building inference image (classification)..."
        "$SCRIPT_DIR/build_and_push_inference.sh" classification

        echo "   Building inference image (segmentation)..."
        "$SCRIPT_DIR/build_and_push_inference.sh" segmentation

        echo "   ✅ Docker images built and pushed."
    else
        echo "⚠️  AWS_ACCOUNT_ID not set in .env.local — cannot build images."
    fi
fi

# Start MLflow server in background (only if using local MLflow)
MLFLOW_PID=""
if [[ "${MLFLOW_TRACKING_URI:-}" != arn:* && "${MLFLOW_TRACKING_URI:-}" != http* ]]; then
    MLFLOW_TRACKING_URI="http://localhost:5000"
    echo "🧪 Starting local MLflow server..."
    python -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db &
    MLFLOW_PID=$!
elif [[ "${MLFLOW_TRACKING_URI:-}" == arn:* ]]; then
    echo "🧪 Using SageMaker-managed MLflow: $MLFLOW_TRACKING_URI"
else
    echo "🧪 Using remote MLflow: $MLFLOW_TRACKING_URI"
fi

# Wait a moment for services to start
sleep 2

# Start the main application
echo "🌐 Starting FastAPI application..."
echo "📍 Application will be available at: http://localhost:8000"
if [[ -n "$MLFLOW_PID" ]]; then
    echo "🧪 MLflow UI available at: http://localhost:5000"
fi
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    [[ -n "${MLFLOW_PID:-}" ]] && kill "$MLFLOW_PID" 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start FastAPI application
python main.py

# If we reach here, cleanup
cleanup