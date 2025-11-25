# vlm

Vision Language Model learning project.

## Installation

Run the automated setup script:

```bash
./setup.sh
```

This will automatically:
- Install `uv` package manager (if needed)
- Install all dependencies (PyTorch 2.9, torchaudio, torchvision)
- Verify your installation

## Stack

- **Python**: 3.11+
- **PyTorch**: 2.9 (with MPS support for M3 Mac, CUDA for A100/B100)
- **Package Manager**: uv (recommended) or pip/conda
- **Dependencies**: See [pyproject.toml](pyproject.toml)