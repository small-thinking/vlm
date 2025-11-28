# vlm

Replicating the LLaVA series of vision-language models.

![LLaVA Model Architecture](resources/llava-qwen.png)

## Setup

```bash
sh scripts/setup.sh
```

The script creates a conda environment named `vlm_env` and automatically detects your platform to install PyTorch with the appropriate backend:
- **Mac M-series (Apple Silicon)**: PyTorch 2.9.0 with MPS support
- **Linux aarch64**: PyTorch 2.9.0 with CUDA 12.8 support
- **CUDA systems** (NVIDIA GPUs): PyTorch 2.9.0 with CUDA 12.8 support
- **Other platforms**: PyTorch 2.9.0 CPU version

Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed first.

After setup, activate the environment:
```bash
conda activate vlm_env
```

Verify installation:
```bash
conda run -n vlm_env python scripts/verify_pytorch.py
conda run -n vlm_env python scripts/verify_llava.py
```

Or if you've activated the environment:
```bash
conda activate vlm_env
python scripts/verify_pytorch.py
python scripts/verify_llava.py
```

## Dataset

Download LLaVA-Pretrain dataset (~100GB):
```bash
conda run -n vlm_env python scripts/prepare_dataset.py
```

Or with the environment activated:
```bash
conda activate vlm_env
python scripts/prepare_dataset.py
```

## Usage

```bash
# Training (with conda run)
conda run -n vlm_env python src/vlm/train/run.py --data_path ~/dataset/llava-pretrain/blip_laion_cc_sbu_558k.json --image_folder ~/dataset/llava-pretrain

# Training (with activated environment)
conda activate vlm_env
python src/vlm/train/run.py --data_path ~/dataset/llava-pretrain/blip_laion_cc_sbu_558k.json --image_folder ~/dataset/llava-pretrain

# Inference
conda run -n vlm_env python src/vlm/inference/inference.py --checkpoint ~/models/llava/checkpoint_phase1.pt --image_path <path> --text "Describe this image"
```

## Troubleshooting

- **Python version error**: The conda environment will automatically install Python 3.11
- **conda not found**: Install conda or miniconda from https://docs.conda.io/en/latest/miniconda.html
- **PyTorch issues**: Try recreating the environment: `conda env remove -n vlm_env && sh scripts/setup.sh`
- **Environment activation issues**: Run `conda init` and restart your terminal

## Stack

- Python 3.11+, PyTorch 2.9, conda package manager
- MPS support (Apple Silicon), CUDA support (NVIDIA GPUs)