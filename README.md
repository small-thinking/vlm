# vlm

Replicating the LLaVA series of vision-language models.

![LLaVA Model Architecture](resources/llava-1.png)

## Setup

```bash
sh scripts/setup.sh
```

## Dataset

To manually download and prepare the LLaVA-Pretrain dataset:

```bash
uv pip install huggingface-hub
uv run python scripts/prepare_dataset.py
```

## Stack

- **Python**: 3.11+
- **PyTorch**: 2.9 (with MPS support for M3 Mac, CUDA for A100/B100)
- **Package Manager**: uv