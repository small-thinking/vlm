#!/bin/bash

set -e  # Exit on error

echo "🚀 Setting up VLM project..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo "✅ uv installed successfully!"
else
    echo "✅ uv is already installed"
fi

# Check Python version
echo "🐍 Checking Python version..."
if ! uv run python --version | grep -q "3.1[1-9]"; then
    echo "⚠️  Warning: Python 3.11+ is required"
fi

# Sync dependencies
echo "📥 Installing dependencies with uv sync..."
uv sync

# Verify installation
echo "🔍 Verifying PyTorch installation..."
if uv run python src/vlm/scripts/verify_pytorch.py; then
    echo ""
    echo "✅ Setup complete! Your environment is ready."
    echo ""
    echo "To run scripts, use:"
    echo "  uv run python <script-path>"
else
    echo "❌ Verification failed. Please check the error messages above."
    exit 1
fi
