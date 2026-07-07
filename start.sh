#!/bin/bash

# MedML Training Platform Startup Script

echo "🚀 Starting MedML Training Platform..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs/tensorboard
mkdir -p data/raw data/processed data/models

# Start MLflow server in background
echo "🧪 Starting MLflow server..."
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db &
MLFLOW_PID=$!

# Start TensorBoard in background
echo "📊 Starting TensorBoard..."
tensorboard --logdir=./logs/tensorboard --port=6006 --host=0.0.0.0 &
TENSORBOARD_PID=$!

# Wait a moment for services to start
sleep 3

# Start the main application
echo "🌐 Starting FastAPI application..."
echo "📍 Application will be available at: http://localhost:8000"
echo "🧪 MLflow UI available at: http://localhost:5000"
echo "📊 TensorBoard available at: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill "$MLFLOW_PID" 2>/dev/null
    kill "$TENSORBOARD_PID" 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start FastAPI application
python main.py

# If we reach here, cleanup
cleanup