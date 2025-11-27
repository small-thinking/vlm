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
        
        # Initialize weights with small values to prevent explosion
        self._initialize_weights()
    
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
    
    def _initialize_weights(self):
        """Initialize connector weights with small values.
        
        Uses Xavier uniform initialization with a small gain to prevent
        large initial activations that can cause training instability.
        """
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for stability
                # Standard Xavier init uses gain=1.0, we use 0.1 for smaller weights
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Project visual features to LLM embedding space."""
        return self.mlp(visual_features)
