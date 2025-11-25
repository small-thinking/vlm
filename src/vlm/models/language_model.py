"""Language model components."""
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LanguageModel(ABC, nn.Module):
    """Abstract base class for language models."""
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through language model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            inputs_embeds: Input embeddings (batch_size, seq_len, hidden_dim)
            labels: Labels for language modeling loss (batch_size, seq_len)
            
        Returns:
            Model output (logits or loss depending on implementation)
        """
        pass
    
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the model hidden dimension."""
        pass
    
    @abstractmethod
    def get_input_embeddings(self) -> nn.Module:
        """Get the input embedding layer."""
        pass


class Qwen2_5LM(LanguageModel):
    """Qwen 2.5 language model wrapper."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B", freeze: bool = False):
        """
        Initialize Qwen 2.5 language model.
        
        Args:
            model_name: HuggingFace model name for Qwen 2.5 model
            freeze: Whether to freeze the model weights
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through Qwen 2.5 model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            inputs_embeds: Input embeddings (batch_size, seq_len, hidden_dim)
            labels: Labels for language modeling loss (batch_size, seq_len)
            
        Returns:
            Model output containing logits and optional loss
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
        )
    
    @property
    def hidden_size(self) -> int:
        """Return Qwen 2.5 model hidden size."""
        return self.model.config.hidden_size
    
    def get_input_embeddings(self) -> nn.Module:
        """Get the input embedding layer."""
        return self.model.get_input_embeddings()
