#!/bin/bash
# GenMed AI — Quick Start Script
# Usage: chmod +x run.sh && ./run.sh

echo "========================================="
echo "  🏥 GenMed AI — Starting Services"
echo "========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.9+"
    exit 1
fi

# Install deps if needed
echo "📦 Checking dependencies..."
pip install -r requirements.txt -q

# Train model if not exists
if [ ! -f "models/model.pkl" ]; then
    echo "🤖 Training ML model (first time)..."
    python3 models/train_model.py
fi

# Start backend in background
echo "🚀 Starting FastAPI backend on port 8000..."
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to be ready
sleep 3

# Start frontend
echo "🌐 Starting Streamlit frontend on port 8501..."
streamlit run frontend/app.py --server.port 8501

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
