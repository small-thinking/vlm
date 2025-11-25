"""Connector/projection layer for bridging vision and language."""
import torch
import torch.nn as nn


class MLPConnector(nn.Module):
    """MLP-based connector for visual feature projection.
    
    Note: The Connector base class has been removed. This class now inherits
    directly from nn.Module.
    """
    
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_layers: int = 1,
        hidden_dim: int = None,
        activation: str = "gelu"
    ):
        """
        Initialize MLP connector.
        
        Args:
            vision_dim: Input dimension from vision encoder
            llm_dim: Output dimension for language model
            num_layers: Number of layers (1 = linear projection, 2+ = MLP)
            hidden_dim: Hidden dimension for MLP (required if num_layers > 1)
            activation: Activation function name ("gelu", "relu", "silu")
        """
        super().__init__()
        
        if num_layers == 1:
            # Simple linear projection
            self.mlp = nn.Linear(vision_dim, llm_dim)
        else:
            # Multi-layer MLP
            if hidden_dim is None:
                raise ValueError("hidden_dim must be provided when num_layers > 1")
            
            layers = []
            # First layer
            layers.append(nn.Linear(vision_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(self._get_activation(activation))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, llm_dim))
            
            self.mlp = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Project visual features to LLM embedding space."""
        return self.mlp(visual_features)
