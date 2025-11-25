"""Vision encoder components."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class VisionEncoder(ABC, nn.Module):
    """Abstract base class for vision encoders."""
    
    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to visual features.
        
        Args:
            images: Image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Visual features of shape (batch_size, num_patches, hidden_dim)
        """
        pass
    
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the output hidden dimension."""
        pass


class CLIPVisionEncoder(VisionEncoder):
    """CLIP vision encoder wrapper."""
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", freeze: bool = True):
        """
        Initialize CLIP vision encoder.
        
        Args:
            model_name: HuggingFace model name for CLIP vision model
            freeze: Whether to freeze the encoder weights
        """
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using CLIP vision encoder.
        
        Args:
            images: Image tensor of shape (batch_size, channels, height, width)
                   Expected to be normalized by CLIP processor
            
        Returns:
            Visual features of shape (batch_size, num_patches, hidden_dim)
        """
        outputs = self.model(pixel_values=images)
        # Use the last hidden state which contains all patch embeddings
        return outputs.last_hidden_state
    
    @property
    def hidden_size(self) -> int:
        """Return CLIP vision model hidden size."""
        return self.model.config.hidden_size
