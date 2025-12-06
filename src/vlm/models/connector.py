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
        """Initialize connector weights with adaptive scaling.
        
        Uses Xavier uniform initialization with adaptive gain. The connector
        must produce outputs in a similar range to the LLM's text embeddings
        (from frozen embedding layer) since they're concatenated together.
        
        For Phase 1 training, only the connector is trainable, so proper
        initialization is critical to prevent numerical instability when
        visual embeddings are fed into the frozen LLM.
        """
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                # Get input and output dimensions
                input_dim = module.weight.shape[1]
                output_dim = module.weight.shape[0]
                
                # Adaptive gain based on output dimension (LLM embedding size)
                # Text embeddings from LLM are typically in range [-2, 2] or so
                # We want connector outputs to be in a similar range
                # Base gain of 0.1 works for Qwen2.5-1.5B (~1536 dim)
                # Scale up for larger models to maintain similar output scale
                if output_dim <= 1536:
                    # Small model (Qwen2.5-1.5B)
                    adaptive_gain = 0.1
                elif output_dim <= 2048:
                    # Medium model (Qwen2.5-3B, Qwen3-4B)
                    adaptive_gain = 0.15
                else:
                    # Large model (Qwen2.5-7B+)
                    adaptive_gain = 0.2
                
                # Cap gain to prevent too large initializations
                adaptive_gain = min(adaptive_gain, 0.3)
                nn.init.xavier_uniform_(module.weight, gain=adaptive_gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Project visual features to LLM embedding space."""
        return self.mlp(visual_features)
