"""Model loading utilities for inference."""
from pathlib import Path
from typing import Optional
import torch
from PIL import Image

from ..models.llava import LLaVAModel
from ..configs.model_config import LLaVAConfig


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[LLaVAConfig] = None,
    device: Optional[torch.device] = None,
) -> LLaVAModel:
    """Load LLaVA model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration. If None, uses default config.
        device: Device to load model on. If None, auto-detects.
        
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    config = config or LLaVAConfig()
    model = LLaVAModel(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    
    return model

