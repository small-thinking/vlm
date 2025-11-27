"""Core inference logic for LLaVA."""
from typing import Optional
import torch
from PIL import Image

from ..models.llava import LLaVAModel


def generate_response(
    model: LLaVAModel,
    image_path: Optional[str] = None,
    text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: Optional[torch.device] = None,
) -> str:
    """Generate response from LLaVA model.
    
    Args:
        model: LLaVA model instance
        image_path: Path to image file (optional)
        text: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        device: Device to run inference on
        
    Returns:
        Generated text response
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    tokenizer = model.language_model.tokenizer
    
    # Process image if provided
    pixel_values = None
    if image_path:
        image = Image.open(image_path).convert('RGB')
        processed = model.vision_encoder.processor(
            images=image,
            return_tensors='pt'
        )
        pixel_values = processed['pixel_values'].to(device)
    
    # Tokenize text
    text_input = f"Human: {text}\nAssistant:" if text else "Assistant:"
    encoding = tokenizer(
        text_input, return_tensors='pt', add_special_tokens=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Prepare initial inputs_embeds
    with torch.no_grad():
        # Get initial embeddings
        text_embeds = model.language_model.get_input_embeddings()(input_ids)
        visual_embeds = None
        
        if pixel_values is not None:
            visual_embeds = model.encode_images(pixel_values)
            # Extend attention mask for visual tokens
            visual_mask = torch.ones(
                visual_embeds.size()[:-1],
                dtype=attention_mask.dtype,
                device=device
            )
            attention_mask = torch.cat([visual_mask, attention_mask], dim=1)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        else:
            inputs_embeds = text_embeds
        
        # Autoregressive generation
        generated_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = model.language_model.model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            
            # Sample next token
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # Update for next iteration
            embed_layer = model.language_model.get_input_embeddings()
            next_embed = embed_layer(next_token_id)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=device, dtype=attention_mask.dtype)
            ], dim=1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        
        # Decode only generated tokens
        response = tokenizer.decode(
            generated_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        )
    
    return response.strip()

