"""Model loading utilities for inference."""
from pathlib import Path
from typing import Optional
import torch

from ..models.llava import LLaVAModel
from ..configs.model_config import LLaVAConfig


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[LLaVAConfig] = None,
    device: Optional[torch.device] = None,
) -> LLaVAModel:
    """Load LLaVA model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (supports ~ expansion)
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

    # Expand ~ to home directory if present
    expanded_path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(str(expanded_path), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    # Ensure consistent dtype across all model components
    # Check what dtype the connector was saved in
    # (most likely to reflect training dtype)
    connector_param = next(model.connector.parameters())
    target_dtype = connector_param.dtype

    # Only convert if it's a mixed precision dtype (bf16 or fp16)
    # This ensures all components use the same dtype as trained connector
    if target_dtype in (torch.bfloat16, torch.float16):
        # Convert vision encoder to match connector dtype
        # Use try-except to handle any conversion issues gracefully
        if hasattr(model.vision_encoder, 'model'):
            try:
                # Check current dtype first to avoid unnecessary conversion
                vision_param = next(
                    model.vision_encoder.model.parameters()
                )
                if vision_param.dtype != target_dtype:
                    # Use .to() which is safe for inference
                    # (converts params and buffers)
                    # For inference, converting buffers is acceptable
                    model.vision_encoder.model = (
                        model.vision_encoder.model.to(dtype=target_dtype)
                    )
            except Exception as e:
                # If conversion fails, log warning but continue
                # Inference code will handle dtype mismatches at runtime
                print(
                    f"Warning: Could not convert vision encoder to "
                    f"{target_dtype}: {e}. "
                    "Will handle dtype conversion at inference time."
                )

        # Language model should already match, but ensure consistency
        if hasattr(model.language_model, 'model'):
            # Only convert if it's not already in the target dtype
            lm_param = next(model.language_model.parameters())
            if lm_param.dtype != target_dtype:
                try:
                    model.language_model.model = (
                        model.language_model.model.to(dtype=target_dtype)
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not convert language model to "
                        f"{target_dtype}: {e}. "
                        "Will handle dtype conversion at inference time."
                    )

    return model
