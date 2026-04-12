#!/usr/bin/env bash
# Development startup script (local, without Docker)

set -e

echo "=========================================="
echo "🏥 Medical RAG - Development Mode"
echo "=========================================="
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION found"
echo

# Check if .env exists
if [ ! -f .env ]; then
    echo "📋 Creating .env from .env.example..."
    cp .env.example .env
    echo "✏️  Please edit .env and add your OPENAI_API_KEY"
    exit 1
fi


echo "✅ Configuration found"
echo

# Check for uv
if command -v uv &> /dev/null; then
    echo "✅ uv found, installing dependencies..."
    uv sync
else
    echo "⚠️  uv not found, installing via pip..."
    pip install uv
    uv sync
fi

echo
echo "🚀 Checking Qdrant..."

# Check if Qdrant is running
if ! curl -f http://localhost:6333/ > /dev/null 2>&1; then
    echo "⚠️  Qdrant not running at http://localhost:6333"
    echo "💡 Start Qdrant with:"
    echo "   docker run -p 6333:6333 qdrant/qdrant"
    echo
    read -p "Continue without Qdrant? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Qdrant is running"
fi

echo
echo "=========================================="
echo "✅ Starting Chainlit..."
echo "=========================================="
echo

# Load environment and start Chainlit using uv environment

uv run chainlit run app/chainlit/chainlit_app.py --host 0.0.0.0 --port 8001
