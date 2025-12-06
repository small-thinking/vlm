"""Utility functions for logging model architecture and components."""
import torch.nn as nn

from ..models.llava import LLaVAModel


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a module.

    Args:
        module: PyTorch module
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in module.parameters()
                   if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_number(num: int) -> str:
    """Format large numbers with K/M/B suffixes.

    Args:
        num: Number to format

    Returns:
        Formatted string (e.g., "1.5B", "250M")
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def log_model_components(
    model: LLaVAModel,
    rank: int = 0,
    use_wandb: bool = False
) -> None:
    """Log detailed information about model components.

    Args:
        model: LLaVA model instance
        rank: Process rank (only rank 0 logs)
        use_wandb: Whether wandb is enabled (suppresses some prints)
    """
    if rank != 0:
        return

    # Get underlying model if wrapped in DDP
    underlying_model = (
        model.module if hasattr(model, 'module') else model
    )

    print("\n" + "=" * 70)
    print("Model Architecture Summary")
    print("=" * 70)

    # Vision Encoder
    vision_encoder = underlying_model.vision_encoder
    vision_config = underlying_model.config.vision_encoder
    vision_total = count_parameters(vision_encoder)
    vision_trainable = count_parameters(
        vision_encoder, trainable_only=True
    )
    vision_frozen = vision_total - vision_trainable

    print(f"\n📷 Vision Encoder: {vision_encoder.__class__.__name__}")
    print(f"   Model: {vision_config.model_name}")
    print(f"   Hidden size: {vision_encoder.hidden_size:,}")
    print(f"   Image size: {vision_encoder.image_size}px")
    print(f"   Visual tokens: {vision_encoder.num_visual_tokens}")
    print(f"   Total parameters: {vision_total:,} "
          f"({format_number(vision_total)})")
    print(f"   Trainable: {vision_trainable:,} "
          f"({format_number(vision_trainable)})")
    print(f"   Frozen: {vision_frozen:,} "
          f"({format_number(vision_frozen)})")
    status = '🔒 Frozen' if vision_config.freeze else '🔓 Trainable'
    print(f"   Status: {status}")

    # Connector
    connector = underlying_model.connector
    connector_config = underlying_model.config.connector
    connector_total = count_parameters(connector)
    connector_trainable = count_parameters(
        connector, trainable_only=True
    )
    connector_frozen = connector_total - connector_trainable

    print(f"\n🔗 Connector: {connector.__class__.__name__}")
    print(f"   Architecture: {connector_config.num_layers} layer(s)")
    if connector_config.num_layers > 1:
        print(f"   Hidden dim: {connector_config.hidden_dim}")
    print(f"   Input dim: {vision_encoder.hidden_size:,}")
    lm_hidden = underlying_model.language_model.hidden_size
    print(f"   Output dim: {lm_hidden:,}")
    print(f"   Activation: {connector_config.activation}")
    print(f"   Total parameters: {connector_total:,} "
          f"({format_number(connector_total)})")
    print(f"   Trainable: {connector_trainable:,} "
          f"({format_number(connector_trainable)})")
    print(f"   Frozen: {connector_frozen:,} "
          f"({format_number(connector_frozen)})")

    # Check if connector is trainable (depends on training stage)
    connector_is_trainable = any(
        p.requires_grad for p in connector.parameters()
    )
    conn_status = ('🔓 Trainable' if connector_is_trainable
                   else '🔒 Frozen')
    print(f"   Status: {conn_status}")

    # Language Model
    language_model = underlying_model.language_model
    lm_config = underlying_model.config.language_model
    lm_total = count_parameters(language_model)
    lm_trainable = count_parameters(
        language_model, trainable_only=True
    )
    lm_frozen = lm_total - lm_trainable

    print(f"\n💬 Language Model: {language_model.__class__.__name__}")
    print(f"   Model: {lm_config.model_name}")
    print(f"   Hidden size: {language_model.hidden_size:,}")
    print(f"   Total parameters: {lm_total:,} "
          f"({format_number(lm_total)})")
    print(f"   Trainable: {lm_trainable:,} "
          f"({format_number(lm_trainable)})")
    print(f"   Frozen: {lm_frozen:,} ({format_number(lm_frozen)})")

    # Check if LM is trainable (depends on training stage)
    lm_is_trainable = any(
        p.requires_grad for p in language_model.parameters()
    )
    lm_status = '🔓 Trainable' if lm_is_trainable else '🔒 Frozen'
    print(f"   Status: {lm_status}")

    # Total Model Summary
    total_params = count_parameters(underlying_model)
    total_trainable = count_parameters(
        underlying_model, trainable_only=True
    )
    total_frozen = total_params - total_trainable

    print("\n📊 Total Model Summary")
    print(f"   Total parameters: {total_params:,} "
          f"({format_number(total_params)})")
    print(f"   Trainable: {total_trainable:,} "
          f"({format_number(total_trainable)})")
    print(f"   Frozen: {total_frozen:,} "
          f"({format_number(total_frozen)})")
    if total_params > 0:
        trainable_ratio = (total_trainable / total_params) * 100
        print(f"   Trainable ratio: {trainable_ratio:.2f}%")

    print("=" * 70 + "\n")
