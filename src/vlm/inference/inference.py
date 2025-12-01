"""Core inference logic for LLaVA."""
from typing import Optional, Iterator
import torch
from PIL import Image

from ..models.llava import LLaVAModel


def _get_model_dtype(model: LLaVAModel) -> torch.dtype:
    """Get the dtype of the model parameters.
    
    Args:
        model: LLaVA model instance
        
    Returns:
        Model dtype (bfloat16, float16, or float32)
    """
    # Check connector dtype first (most likely to be in training dtype)
    connector_param = next(model.connector.parameters())
    if connector_param.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return connector_param.dtype
    
    # Fall back to language model dtype
    lm_param = next(model.language_model.parameters())
    return lm_param.dtype


def generate_response(
    model: LLaVAModel,
    image_path: Optional[str] = None,
    text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: Optional[torch.device] = None,
    use_greedy: bool = False,
) -> str:
    """Generate response from LLaVA model.

    Args:
        model: LLaVA model instance
        image_path: Path to image file (optional)
        text: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        device: Device to run inference on
        use_greedy: If True, use greedy decoding (deterministic).
            If False, use sampling.

    Returns:
        Generated text response
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    tokenizer = model.language_model.tokenizer
    
    # Get model dtype to ensure consistency
    model_dtype = _get_model_dtype(model)
    
    # Process image if provided
    pixel_values = None
    if image_path:
        with Image.open(image_path) as image:
            image_rgb = image.convert('RGB')
            processed = model.vision_encoder.processor(
                images=image_rgb,
                return_tensors='pt'
            )
            pixel_values = processed['pixel_values'].to(device)
            # Convert pixel_values to model dtype to avoid dtype mismatches
            if pixel_values.dtype != model_dtype and pixel_values.is_floating_point():
                pixel_values = pixel_values.to(dtype=model_dtype)
    
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
            # Ensure visual_embeds match text_embeds dtype
            if visual_embeds.dtype != text_embeds.dtype:
                visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)
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
        embed_layer = model.language_model.get_input_embeddings()
        
        # Collect stop token IDs to check
        stop_token_ids = []
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
            stop_token_ids.append(tokenizer.pad_token_id)
        
        for _ in range(max_new_tokens):
            outputs = model.language_model.model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            
            # Sample or greedily decode next token
            logits = outputs.logits[:, -1, :]
            
            if use_greedy:
                # Greedy decoding: always pick most likely token
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Sampling: use temperature and multinomial (stochastic)
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Check for stop tokens - break immediately if found
            next_token_id_value = next_token_id.item()
            if next_token_id_value in stop_token_ids:
                break
            
            # Update for next iteration
            next_embed = embed_layer(next_token_id)
            # Ensure next_embed matches inputs_embeds dtype
            if next_embed.dtype != inputs_embeds.dtype:
                next_embed = next_embed.to(dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=device, dtype=attention_mask.dtype)
            ], dim=1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        
        # Decode only generated tokens
        start_idx = input_ids.shape[1]
        response = tokenizer.decode(
            generated_ids[0, start_idx:],
            skip_special_tokens=True
        )
    
    return response.strip()


def generate_response_stream(
    model: LLaVAModel,
    image_path: Optional[str] = None,
    text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: Optional[torch.device] = None,
    use_greedy: bool = False,
) -> Iterator[str]:
    """Generate streaming response from LLaVA model.
    
    Yields tokens as they are generated, allowing for real-time streaming.

    Args:
        model: LLaVA model instance
        image_path: Path to image file (optional)
        text: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        device: Device to run inference on
        use_greedy: If True, use greedy decoding (deterministic).
            If False, use sampling.

    Yields:
        Generated text tokens as they are produced
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    tokenizer = model.language_model.tokenizer
    
    # Get model dtype to ensure consistency
    model_dtype = _get_model_dtype(model)
    
    # Process image if provided
    pixel_values = None
    if image_path:
        with Image.open(image_path) as image:
            image_rgb = image.convert('RGB')
            processed = model.vision_encoder.processor(
                images=image_rgb,
                return_tensors='pt'
            )
            pixel_values = processed['pixel_values'].to(device)
            # Convert pixel_values to model dtype to avoid dtype mismatches
            if pixel_values.dtype != model_dtype and pixel_values.is_floating_point():
                pixel_values = pixel_values.to(dtype=model_dtype)
    
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
            # Ensure visual_embeds match text_embeds dtype
            if visual_embeds.dtype != text_embeds.dtype:
                visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)
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
        embed_layer = model.language_model.get_input_embeddings()
        generated_text = ""
        
        # Collect stop token IDs to check
        stop_token_ids = []
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
            stop_token_ids.append(tokenizer.pad_token_id)
        
        for _ in range(max_new_tokens):
            outputs = model.language_model.model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            
            # Sample or greedily decode next token
            logits = outputs.logits[:, -1, :]
            
            if use_greedy:
                # Greedy decoding: always pick most likely token
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Sampling: use temperature and multinomial (stochastic)
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Check for stop tokens - break immediately if found
            next_token_id_value = next_token_id.item()
            if next_token_id_value in stop_token_ids:
                break
            
            # Decode the new token and yield the delta
            new_token = tokenizer.decode(
                next_token_id[0],
                skip_special_tokens=True
            )
            
            # Only yield if we have new text (skip special tokens that decode to empty)
            if new_token:
                generated_text += new_token
                yield new_token
            
            # Update for next iteration
            next_embed = embed_layer(next_token_id)
            # Ensure next_embed matches inputs_embeds dtype
            if next_embed.dtype != inputs_embeds.dtype:
                next_embed = next_embed.to(dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=device, dtype=attention_mask.dtype)
            ], dim=1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
