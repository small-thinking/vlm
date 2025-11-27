"""LLaVA: Large Language and Vision Assistant."""
from typing import Optional
import torch
import torch.nn as nn

from .vision_encoder import CLIPVisionEncoder
from .connector import MLPConnector
from .language_model import Qwen2_5LM
from ..configs.model_config import LLaVAConfig


class LLaVAModel(nn.Module):
    """LLaVA multimodal model combining vision encoder, connector, and LLM."""
    
    def __init__(self, config: Optional[LLaVAConfig] = None):
        """Initialize LLaVA model.
        
        Args:
            config: LLaVA configuration. If None, uses default config.
        """
        super().__init__()
        self.config = config or LLaVAConfig()
        
        # Initialize components
        self.vision_encoder = CLIPVisionEncoder(
            model_name=self.config.vision_encoder.model_name,
            freeze=self.config.vision_encoder.freeze
        )
        self.language_model = Qwen2_5LM(
            model_name=self.config.language_model.model_name,
            freeze=self.config.language_model.freeze
        )
        self.connector = MLPConnector(
            vision_dim=self.vision_encoder.hidden_size,
            llm_dim=self.language_model.hidden_size,
            num_layers=self.config.connector.num_layers,
            hidden_dim=self.config.connector.hidden_dim,
            activation=self.config.connector.activation
        )
    
    def freeze_module(self, module: nn.Module, freeze: bool):
        """Freeze or unfreeze a module."""
        for param in module.parameters():
            param.requires_grad = not freeze
        module.eval() if freeze else module.train()
    
    def set_training_stage(self, stage: int):
        """Set training stage for LLaVA.
        
        Stage 1: Train connector only (alignment)
        Stage 2: Train connector + LLM (instruction tuning)
        """
        if stage == 1:
            self.freeze_module(self.vision_encoder, True)
            self.freeze_module(self.language_model, True)
            self.freeze_module(self.connector, False)
        elif stage == 2:
            self.freeze_module(self.vision_encoder, True)
            self.freeze_module(self.language_model, False)
            self.freeze_module(self.connector, False)
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to visual embeddings in LLM space."""
        # Optimization: Use no_grad if vision encoder is frozen
        if not next(self.vision_encoder.parameters()).requires_grad:
            with torch.no_grad():
                features = self.vision_encoder(images)
        else:
            features = self.vision_encoder(images)
            
        return self.connector(features)
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward pass through LLaVA model."""
        # Convert text token IDs to embeddings
        text_embeds = None
        if input_ids is not None:
            text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # Process images if present
        visual_embeds = None
        if images is not None:
            visual_embeds = self.encode_images(images)
            # Extend attention mask for visual tokens
            if attention_mask is not None:
                visual_mask = torch.ones(
                    visual_embeds.size()[:-1],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat(
                    [visual_mask, attention_mask], dim=1
                )
            # Extend labels for visual tokens (masked with -100)
            if labels is not None:
                visual_labels = torch.full(
                    visual_embeds.size()[:-1],
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                labels = torch.cat([visual_labels, labels], dim=1)
        
        # Concatenate visual and text embeddings
        if visual_embeds is not None and text_embeds is not None:
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        elif visual_embeds is not None:
            inputs_embeds = visual_embeds
        elif text_embeds is not None:
            inputs_embeds = text_embeds
        
        # Always use inputs_embeds, never pass input_ids to language model
        return self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
        )
