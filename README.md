# vlm

Replicating the LLaVA series of vision-language models.

![LLaVA Model Architecture](resources/llava-1.png)

## Setup

```bash
sh scripts/setup.sh
```

The script installs `uv`, dependencies, and optionally downloads the dataset. If `uv` isn't found after installation, restart your terminal or run `export PATH="$HOME/.cargo/bin:$PATH"`.

Verify installation:
```bash
uv run python scripts/verify_pytorch.py
uv run python scripts/verify_llava.py
```

## Dataset

Download LLaVA-Pretrain dataset (~100GB):
```bash
uv pip install huggingface-hub
uv run python scripts/prepare_dataset.py
```

## Usage

```bash
# Training
uv run python src/vlm/train/run.py --data_path ~/dataset/llava-pretrain/blip_laion_cc_sbu_558k.json --image_folder ~/dataset/llava-pretrain

# Inference
uv run python src/vlm/inference/inference.py --checkpoint ~/models/llava/checkpoint_phase1.pt --image_path <path> --text "Describe this image"
```

## Troubleshooting

- **Python version error**: Install Python 3.11+ (`brew install python@3.11` on macOS)
- **uv not found**: Restart terminal or `export PATH="$HOME/.cargo/bin:$PATH"`
- **PyTorch issues**: Check Python version, try `uv sync --reinstall`

## Stack

- Python 3.11+, PyTorch 2.9, uv package manager
- MPS support (Apple Silicon), CUDA support (NVIDIA GPUs)