"""LLaVA Instruct Dataset for Phase 2 training.
"""
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor
import numpy as np

from vlm.configs.data_config import DataConfig


class LLaVAInstructDataset(Dataset):
    """Dataset for LLaVA Phase 2 instruction tuning.
    
    Args:
        data_path: Path to the dataset JSON file
        image_folder: Path to the folder containing images (optional)
            If None, images are expected to be embedded in the dataset
        image_processor: Processor for vision encoder (e.g., CLIPImageProcessor)
        tokenizer: Tokenizer for language model
        max_length: Maximum sequence length (includes visual tokens)
        num_visual_tokens: Number of visual tokens from vision encoder
            (default 257 for CLIP ViT-L/14: 256 patches + 1 CLS)
    """
    
    def __init__(
        self,
        data_path: str,
        image_folder: Optional[str] = None,
        image_processor: Optional[CLIPImageProcessor] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 768,
        num_visual_tokens: int = 257,  # CLIP ViT-L/14: 256 patches + 1 CLS
    ):
        # Expand ~ to home directory for both paths
        self.data_path = str(Path(data_path).expanduser())
        self.image_folder = (
            Path(image_folder).expanduser() if image_folder else None
        )
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_visual_tokens = num_visual_tokens
        
        # Load dataset from JSON file
        print(f"Loading dataset from {self.data_path}...")
        with open(self.data_path, 'r') as f:
            self.raw_data = json.load(f)
        
        # Build lightweight index: (conversation_idx, turn_start_idx) tuples
        # Only stores indices, not actual data - suitable for huge datasets
        print("Building sample index...")
        self.index = self._build_index()
        print(f"Indexed {len(self.index)} samples from {len(self.raw_data)} conversations.")
    
    def __len__(self) -> int:
        """Return the number of training samples."""
        return len(self.index)
    
    def _build_index(self) -> List[tuple]:
        """Build lightweight index: (conversation_idx, turn_start_idx) tuples.
        
        Only stores indices, not data. Suitable for huge datasets.
        
        Supports multiple formats:
        - LLaVA format: conversations = [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
        - HuggingFace chat format: messages = [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        index = []
        for conv_idx, sample in enumerate(self.raw_data):
            # Try 'conversations' first (LLaVA format)
            conversations = sample.get('conversations', None)
            
            # If not found, try 'messages' (HuggingFace chat format)
            if conversations is None:
                messages = sample.get('messages', None)
                if messages is not None:
                    # Convert messages format to conversations format
                    conversations = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                            # Map roles: user -> human, assistant -> gpt
                            if role == 'user':
                                conversations.append(
                                    {'from': 'human', 'value': content}
                                )
                            elif role == 'assistant':
                                conversations.append(
                                    {'from': 'gpt', 'value': content}
                                )
            
            # Handle different conversation formats
            if conversations is None or not isinstance(conversations, list):
                continue
            
            # Check if conversations is a list of dicts or strings
            if len(conversations) == 0:
                continue
            
            # Try to detect format by checking first element
            first_item = conversations[0]
            if isinstance(first_item, str):
                # If conversations is a list of strings, skip this sample
                # (unexpected format)
                continue
            elif not isinstance(first_item, dict):
                # Unknown format
                continue
            
            i = 0
            while i < len(conversations):
                turn = conversations[i]
                if not isinstance(turn, dict):
                    i += 1
                    continue
                
                # Check for 'human' or 'user' role
                role = turn.get('from', '') or turn.get('role', '')
                if role in ('human', 'user'):
                    # Look for next turn with 'gpt' or 'assistant' role
                    if i + 1 < len(conversations):
                        next_turn = conversations[i + 1]
                        if isinstance(next_turn, dict):
                            next_role = next_turn.get('from', '') or next_turn.get('role', '')
                            if next_role in ('gpt', 'assistant'):
                                index.append((conv_idx, i))
                                i += 2
                                continue
                    i += 1
                else:
                    i += 1
        return index
    
    def _process_image(self, image: Any) -> Optional[Image.Image]:
        """Process image from various formats to PIL Image.
        
        Args:
            image: Image in various formats (PIL, bytes, numpy, etc.)
            
        Returns:
            PIL Image in RGB format or None if conversion fails
        """
        if image is None:
            return None
        
        try:
            if isinstance(image, Image.Image):
                return image.convert('RGB')
            elif isinstance(image, bytes):
                return Image.open(io.BytesIO(image)).convert('RGB')
            elif hasattr(image, 'convert'):
                return image.convert('RGB')
            else:
                if isinstance(image, np.ndarray):
                    return Image.fromarray(image).convert('RGB')
                else:
                    return Image.fromarray(np.array(image)).convert('RGB')
        except Exception:
            return None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - pixel_values: Processed image tensor (or None)
                - input_ids: Tokenized input text
                - attention_mask: Attention mask for input
                - labels: Tokenized labels (input masked, only assistant reply)
        """
        # Get conversation and turn indices from lightweight index
        conv_idx, turn_start_idx = self.index[idx]
        sample = self.raw_data[conv_idx]
        
        # Get image from conversation (same for all turns)
        # Support both embedded images and local file paths
        raw_image = sample.get('image', None)
        image_path = sample.get('image_path', None)
        
        conversation_image = None
        if raw_image is not None:
            # Embedded image (bytes, PIL, numpy, etc.)
            conversation_image = self._process_image(raw_image)
        elif image_path is not None and self.image_folder is not None:
            # Local file path
            try:
                full_path = self.image_folder / image_path
                if full_path.exists():
                    conversation_image = Image.open(full_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load image {image_path}: {e}")
        
        # Parse the specific turn on-the-fly
        conversations = sample.get('conversations', None)
        
        # If not found, try 'messages' format (HuggingFace chat format)
        if conversations is None:
            messages = sample.get('messages', [])
            if messages:
                # Convert messages format to conversations format
                conversations = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', '')
                        content = msg.get('content', '')
                        # Map roles: user -> human, assistant -> gpt
                        if role == 'user':
                            conversations.append({'from': 'human', 'value': content})
                        elif role == 'assistant':
                            conversations.append({'from': 'gpt', 'value': content})
        
        if conversations is None:
            conversations = []
        
        # Validate conversation structure
        if not isinstance(conversations, list) or len(conversations) < turn_start_idx + 2:
            raise ValueError(
                f"Invalid conversation structure at index {idx}: "
                f"conversations is not a list or too short"
            )
        
        human_turn = conversations[turn_start_idx]
        gpt_turn = conversations[turn_start_idx + 1]
        
        # Validate turn structure
        if not isinstance(human_turn, dict) or not isinstance(gpt_turn, dict):
            raise ValueError(
                f"Invalid turn structure at index {idx}: "
                f"turns must be dictionaries"
            )
        
        # Extract text (remove <image> placeholder)
        # Handle both 'value' (LLaVA) and 'content' (HuggingFace) keys
        user_text = (
            human_turn.get('value', '') or human_turn.get('content', '')
        ).replace('<image>', '').strip()
        assistant_text = (
            gpt_turn.get('value', '') or gpt_turn.get('content', '')
        ).strip()
        
        result = {}
        if conversation_image is not None:
            try:
                processed_images = self.image_processor(
                    images=conversation_image,
                    return_tensors='pt'
                )
                # Shape: (C, H, W) - will be batched in collate_fn
                result['pixel_values'] = processed_images['pixel_values'][0]
            except Exception as e:
                print(f"Warning: Could not process image at index {idx}: {e}")
                result['pixel_values'] = None
        else:
            result['pixel_values'] = None

        # Reserve space for visual tokens (prepended in model forward)
        has_image = result.get('pixel_values') is not None
        text_max_length = (
            self.max_length - self.num_visual_tokens if has_image
            else self.max_length
        )
        
        # Tokenize separately to avoid boundary issues
        user_input_text = f"Human: {user_text}\n"
        user_encoding = self.tokenizer(
            user_input_text,
            truncation=True,
            max_length=text_max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        user_ids = user_encoding['input_ids'].squeeze(0)
        user_len = user_ids.shape[0]
        
        assistant_prompt_max_length = max(5, text_max_length - user_len)
        assistant_prompt_encoding = self.tokenizer(
            "Assistant:",
            truncation=True,
            max_length=assistant_prompt_max_length,
            return_tensors='pt',
            add_special_tokens=False
        )
        assistant_prompt_ids = assistant_prompt_encoding['input_ids'].squeeze(0)
        assistant_prompt_len = assistant_prompt_ids.shape[0]
        
        assistant_response_max_length = max(
            10, text_max_length - user_len - assistant_prompt_len
        )
        assistant_response_encoding = self.tokenizer(
            f" {assistant_text}",
            truncation=True,
            max_length=assistant_response_max_length,
            return_tensors='pt',
            add_special_tokens=False
        )
        assistant_response_ids = assistant_response_encoding['input_ids'].squeeze(0)
        
        full_ids = torch.cat([
            user_ids, assistant_prompt_ids, assistant_response_ids
        ], dim=0)
        input_len = user_len + assistant_prompt_len
        
        # Pad or truncate (visual tokens added in model forward)
        if full_ids.shape[0] < text_max_length:
            padding_length = text_max_length - full_ids.shape[0]
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
                if self.tokenizer.eos_token_id is not None
                else 0
            )
            padding = torch.full(
                (padding_length,),
                pad_token_id,
                dtype=full_ids.dtype
            )
            full_ids = torch.cat([full_ids, padding], dim=0)
            attention_mask = torch.cat([
                torch.ones(
                    user_len + assistant_prompt_len + assistant_response_ids.shape[0],
                    dtype=torch.long
                ),
                torch.zeros(padding_length, dtype=torch.long)
            ], dim=0)
        else:
            # Truncate if too long
            full_ids = full_ids[:text_max_length]
            attention_mask = torch.ones(text_max_length, dtype=torch.long)
            # Ensure at least some response tokens remain
            if input_len >= text_max_length:
                input_len = max(0, text_max_length - 10)
        
        result['input_ids'] = full_ids
        result['attention_mask'] = attention_mask
        
        # Mask input (user + prompt), keep target (assistant response)
        result['labels'] = full_ids.clone()
        result['labels'][:input_len] = -100
        
        if input_len > len(full_ids):
            raise ValueError(
                f"input_len ({input_len}) > full_ids length ({len(full_ids)})"
            )
        
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key == 'pixel_values':
            # Handle pixel_values: stack if all have images, else None batch
            pixel_values_list = [item[key] for item in batch]
            has_images = [pv is not None for pv in pixel_values_list]
            
            if all(has_images):
                collated[key] = torch.stack(pixel_values_list)
            elif any(has_images):
                # Mixed: fill missing with zeros
                first_valid = next(pv for pv in pixel_values_list if pv is not None)
                ref_shape, ref_dtype, ref_device = (
                    first_valid.shape, first_valid.dtype, first_valid.device
                )
                stacked = [
                    pv if pv is not None
                    else torch.zeros(ref_shape, dtype=ref_dtype, device=ref_device)
                    for pv in pixel_values_list
                ]
                collated[key] = torch.stack(stacked)
            else:
                collated[key] = None
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


def build_instruct_dataloader(
    config: DataConfig,
    tokenizer: Any,
    image_processor: CLIPImageProcessor,
) -> DataLoader:
    """Build DataLoader for LLaVA instruction tuning.
    
    Args:
        config: Data configuration (must have data_path and image_folder)
        tokenizer: Tokenizer for language model
        image_processor: Image processor for vision encoder
        
    Returns:
        DataLoader instance
    """
    dataset = LLaVAInstructDataset(
        data_path=config.data_path,
        image_folder=config.image_folder,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        collate_fn=collate_fn,
        pin_memory=True,
    )

