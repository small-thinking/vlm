"""Model configuration dataclasses."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionEncoderConfig:
    """Configuration for vision encoder.

    Examples:
        # Default CLIP ViT-L/14 with 336px resolution
        config = VisionEncoderConfig()

        # CLIP ViT-L/14 with 224px resolution
        config = VisionEncoderConfig(
            model_name="openai/clip-vit-large-patch14"
        )

        # CLIP ViT-B/16
        config = VisionEncoderConfig(model_name="openai/clip-vit-base-patch16")
    """

    model_name: str = "openai/clip-vit-large-patch14-336"
    """HuggingFace model name for the vision encoder.

    Common options:
        - "openai/clip-vit-large-patch14-336" (default, 336px resolution)
        - "openai/clip-vit-large-patch14" (224px)
        - "openai/clip-vit-base-patch16" (base model)
        - "openai/clip-vit-base-patch32" (base model, larger patches)
    """

    freeze: bool = True
    """Whether to freeze vision encoder weights during training."""


@dataclass
class ConnectorConfig:
    """Configuration for the connector/projection layer."""

    num_layers: int = 1
    """Number of MLP layers.

    1 = linear projection, 2 = MLP with hidden layer.
    """

    hidden_dim: Optional[int] = None
    """Hidden dimension for MLP. Only used if num_layers > 1."""

    activation: str = "gelu"
    """Activation function for MLP layers."""


@dataclass
class LanguageModelConfig:
    """Configuration for language model."""

    model_name: str = "Qwen/Qwen2.5-1.5B"
    """HuggingFace model name for the language model."""

    freeze: bool = True
    """Whether to freeze language model weights during training.

    Default is True for LLaVA stage 1 training (only train connector).
    Set to False for stage 2 training (train connector + LLM).
    """


@dataclass
class LLaVAConfig:
    """Main configuration for LLaVA model."""

    vision_encoder: VisionEncoderConfig = None
    connector: ConnectorConfig = None
    language_model: LanguageModelConfig = None

    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.vision_encoder is None:
            self.vision_encoder = VisionEncoderConfig()
        if self.connector is None:
            self.connector = ConnectorConfig()
        if self.language_model is None:
            self.language_model = LanguageModelConfig()
