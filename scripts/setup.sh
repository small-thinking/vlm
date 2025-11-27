#!/bin/bash

set -e  # Exit on error

echo "🚀 Setting up VLM project..."

# Check for required tools
if ! command -v curl &> /dev/null; then
    echo "❌ Error: curl is required but not found. Please install curl first."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for this session (try common locations)
    if [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    elif [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Verify uv is now available
    if ! command -v uv &> /dev/null; then
        echo "❌ Error: uv was installed but not found in PATH."
        echo "   Please restart your terminal or run:"
        echo "   export PATH=\"\$HOME/.cargo/bin:\$PATH\""
        echo "   Then run this script again."
        exit 1
    fi
    
    echo "✅ uv installed successfully!"
else
    echo "✅ uv is already installed"
fi

# Check Python version
echo "🐍 Checking Python version..."
PYTHON_VERSION=$(uv run python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo "❌ Error: Python 3.11+ is required, but found Python $PYTHON_VERSION"
    echo "   Please install Python 3.11 or later:"
    echo "   - macOS: brew install python@3.11"
    echo "   - Linux: Use your package manager (apt, yum, etc.)"
    echo "   - Or download from https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Sync dependencies
echo "📥 Installing dependencies with uv sync..."
uv sync

# Verify installation
echo "🔍 Verifying PyTorch installation..."
if uv run python scripts/verify_pytorch.py; then
    echo ""
    echo "✅ Setup complete! Your environment is ready."
    echo ""
    echo "To run scripts, use:"
    echo "  uv run python <script-path>"
else
    echo "❌ Verification failed. Please check the error messages above."
    exit 1
fi

# Ask user if they want to download the dataset
echo ""
read -p "📦 Download LLaVA-Pretrain dataset? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Downloading and preparing LLaVA-Pretrain dataset..."
    
    # Install huggingface-hub if not already installed
    uv pip install huggingface-hub
    
    # Run the preparation script
    # We assume this script is run from the project root, so scripts/prepare_dataset.py is correct
    uv run python scripts/prepare_dataset.py
    
else
    echo "⏭️  Skipping dataset download."
    echo "   You can download it later by running:"
    echo "   uv pip install huggingface-hub"
    echo "   uv run python scripts/prepare_dataset.py"
fi
