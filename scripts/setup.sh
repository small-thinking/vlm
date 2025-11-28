#!/bin/bash

set -e  # Exit on error

echo "🚀 Setting up VLM project..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is required but not found."
    echo "   Please install conda or miniconda first:"
    echo "   - macOS/Linux: https://docs.conda.io/en/latest/miniconda.html"
    echo "   - Or use: brew install miniconda"
    exit 1
fi

echo "✅ conda is installed"

# Check if environment already exists
ENV_NAME="vlm_env"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✅ Conda environment '${ENV_NAME}' already exists. Using existing environment."
else
    # Create conda environment with Python 3.11
    echo "📦 Creating conda environment '${ENV_NAME}' with Python 3.11..."
    conda create -n ${ENV_NAME} python=3.11 -y
    echo "✅ Environment '${ENV_NAME}' created"
fi

# Detect platform and install PyTorch with appropriate backend
echo "🔍 Detecting platform..."
ARCH=$(uname -m)
OS=$(uname -s)

# Check for CUDA availability
HAS_CUDA=false
if command -v nvidia-smi &> /dev/null; then
    HAS_CUDA=true
    echo "✅ NVIDIA GPU detected (CUDA available)"
fi

# Install PyTorch based on platform (in the conda environment)
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    # Mac M-series (Apple Silicon) - MPS support
    echo "📥 Installing PyTorch 2.9.0 for Mac M-series (MPS support)..."
    conda run -n ${ENV_NAME} pip install torch==2.9.0 torchvision torchaudio
elif [[ "$OS" == "Linux" && ("$ARCH" == "aarch64" || "$ARCH" == "arm64") ]]; then
    # Linux aarch64 - install with CUDA 12.8 support
    echo "📥 Installing PyTorch 2.9.0 for Linux aarch64 with CUDA 12.8 support..."
    conda run -n ${ENV_NAME} pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
elif [[ "$HAS_CUDA" == true ]]; then
    # CUDA system (x86_64) - install with CUDA 12.8 support
    echo "📥 Installing PyTorch 2.9.0 with CUDA 12.8 support..."
    conda run -n ${ENV_NAME} pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
    # Default (CPU or unknown) - install CPU version
    echo "📥 Installing PyTorch 2.9.0 (CPU version)..."
    conda run -n ${ENV_NAME} pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies via pip from pyproject.toml
# pip will skip torch packages since they're already installed
echo "📥 Installing other dependencies via pip..."
conda run -n ${ENV_NAME} pip install -e .

# Verify installation
echo "🔍 Verifying PyTorch installation..."
if conda run -n ${ENV_NAME} python scripts/verify_pytorch.py; then
    echo ""
    echo "✅ Setup complete! Your environment is ready."
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate ${ENV_NAME}"
    echo ""
    echo "To run scripts, use:"
    echo "  conda run -n ${ENV_NAME} python <script-path>"
    echo "  (or activate the environment first with 'conda activate ${ENV_NAME}')"
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
    conda run -n ${ENV_NAME} python scripts/prepare_dataset.py
else
    echo "⏭️  Skipping dataset download."
    echo "   You can download it later by running:"
    echo "   conda run -n ${ENV_NAME} python scripts/prepare_dataset.py"
fi
