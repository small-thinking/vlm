"""LLaVA Pretrain Dataset for Phase 1 training.

        Supports LLaVA-Pretrain format:
        - Each sample has "image" (path) and "conversations" fields
        - "conversations" contains exactly one round:
          [{"from": "human", "value": "question<image>"},
           {"from": "gpt", "value": "answer"}]

Setup:
  ./setup.sh  # Will prompt to download dataset

"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor

from vlm.configs.data_config import DataConfig


class LLaVAPretrainDataset(Dataset):
    """Dataset for LLaVA Phase 1 pretraining (vision-language alignment).
    
    During training:
    - Image is converted to embeddings via vision encoder
    - Visual embeddings are projected through connector to LLM space
    - Visual embeddings are appended to the end of user input text
    - All input (user text + image) is masked in labels
    - Only assistant response is used as labels
    
    Args:
        data_path: Path to the dataset JSON file
        image_folder: Path to the folder containing images
        image_processor: Processor for vision encoder
            (e.g., CLIPImageProcessor)
        tokenizer: Tokenizer for language model
        max_length: Maximum sequence length (includes visual tokens)
        num_visual_tokens: Number of visual tokens from vision encoder
            (default 257 for CLIP ViT-L/14: 256 patches + 1 CLS)
    """
    
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        image_processor: Optional[CLIPImageProcessor] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 768,
        num_visual_tokens: int = 257,  # CLIP ViT-L/14: 256 patches + 1 CLS
    ):
        self.data_path = data_path
        self.image_folder = Path(image_folder) if image_folder else None
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_visual_tokens = num_visual_tokens
        self.data = self._preload_data()
    
    def __len__(self) -> int:
        """Return the number of training samples."""
        return len(self.data)
    
    def _preload_data(self) -> List[Dict[str, Any]]:
        """Parse LLaVA-Pretrain format into training samples."""
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)

        samples = []
        for sample in raw_data:
            conversations = sample.get('conversations', [])
            image_path = sample.get('image', None)
            
            human_msg = conversations[0]
            gpt_msg = conversations[1]
            
            # Extract user text (remove <image> placeholder)
            user_value = human_msg.get('value', '')
            user_text = user_value.replace('<image>', '').strip()
            assistant_text = gpt_msg.get('value', '').strip()
            training_sample = {
                'user_text': user_text,
                'assistant_text': assistant_text,
                'image_path': image_path,
            }
            samples.append(training_sample)
        
        print(f"Parsed {len(samples)} training samples.")
        return samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - pixel_values: Processed image tensor
                - input_ids: Tokenized input text
                - attention_mask: Attention mask for input
                - labels: Tokenized labels (input masked, only assistant reply)
        """
        sample = self.data[idx]
        
        user_text = sample.get('user_text', '')
        assistant_text = sample.get('assistant_text', '')
        image_path = sample.get('image_path', None)
        
        result = {}
        if image_path:
            try:
                image = self._load_image(image_path)
                processed_images = self.image_processor(
                    images=image,
                    return_tensors='pt'
                )
                # Shape: (C, H, W) - will be batched in collate_fn
                result['pixel_values'] = processed_images['pixel_values'][0]
            except (FileNotFoundError, Exception) as e:
                print(f"Warning: Could not load image {image_path}: {e}")
                result['pixel_values'] = None
        else:
            result['pixel_values'] = None
        
        # Create the input text and target text
        # Format: "Human: {question}\nAssistant: {answer}"
        input_text = f"Human: {user_text}\nAssistant:"
        target_text = f" {assistant_text}"
        
        # Account for visual tokens when truncating text
        # Visual tokens are prepended to text tokens in the model forward pass
        # So we need to reserve space for them in max_length
        has_image = result.get('pixel_values') is not None
        text_max_length = self.max_length
        if has_image:
            text_max_length = self.max_length - self.num_visual_tokens
        
        # Tokenize input and target separately to avoid boundary issues
        # This ensures clean token boundaries between input and target
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=text_max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        input_ids = input_encoding['input_ids'].squeeze(0)
        input_len = input_ids.shape[0]
        
        # Tokenize target separately
        # Reserve space for input tokens in the target max_length
        target_max_length = text_max_length - input_len
        if target_max_length < 10:
            target_max_length = 10
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            max_length=target_max_length,
            return_tensors='pt',
            add_special_tokens=False  # Don't add special tokens for target
        )
        target_ids = target_encoding['input_ids'].squeeze(0)
        
        # Concatenate input and target token IDs
        full_ids = torch.cat([input_ids, target_ids], dim=0)
        
        # Pad or truncate to text_max_length
        # (visual tokens will be added in model forward)
        # Note: We pad/truncate to text_max_length here, not self.max_length
        # because visual tokens are added separately in the model
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
                    input_ids.shape[0] + target_ids.shape[0],
                    dtype=torch.long
                ),
                torch.zeros(padding_length, dtype=torch.long)
            ], dim=0)
        else:
            # Truncate if too long
            full_ids = full_ids[:text_max_length]
            attention_mask = torch.ones(text_max_length, dtype=torch.long)
            # Update input_len if truncated
            if input_len >= text_max_length:
                input_len = text_max_length
        
        result['input_ids'] = full_ids
        result['attention_mask'] = attention_mask
        
        # Create labels: mask input, keep target
        result['labels'] = full_ids.clone()
        result['labels'][:input_len] = -100
        
        return result

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from relative path.
        
        Args:
            image_path: Relative path to image file
            
        Returns:
            PIL Image
        """
        if self.image_folder is None:
            raise ValueError("image_folder must be provided to load images")
        
        full_path = self.image_folder / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        return Image.open(full_path).convert('RGB')


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader.
    
    Handles batching of samples. Each sample may or may not have an image.
    Maintains batch size consistency - all tensors have same batch dimension.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched tensors with consistent batch size
    """
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key == 'pixel_values':
            # Handle pixel_values: stack if all have images, else None batch
            pixel_values_list = [item[key] for item in batch]
            has_images = [pv is not None for pv in pixel_values_list]
            
            if all(has_images):
                # All samples have images - stack normally
                collated[key] = torch.stack(pixel_values_list)
            elif any(has_images):
                # Mixed case: some have images, some don't
                # This shouldn't happen in LLaVA pretrain, handle gracefully
                # Use the first valid image shape as reference
                first_valid_idx = next(
                    i for i, pv in enumerate(pixel_values_list)
                    if pv is not None
                )
                ref_shape = pixel_values_list[first_valid_idx].shape
                ref_dtype = pixel_values_list[first_valid_idx].dtype
                ref_device = pixel_values_list[first_valid_idx].device
                
                # Create a batch with zeros for missing images
                stacked = []
                for pv in pixel_values_list:
                    if pv is not None:
                        stacked.append(pv)
                    else:
                        # Create zero tensor matching reference shape
                        stacked.append(
                            torch.zeros(
                                ref_shape, dtype=ref_dtype, device=ref_device
                            )
                        )
                collated[key] = torch.stack(stacked)
            else:
                # No images in batch
                collated[key] = None
        else:
            # Stack tensors (input_ids, attention_mask, labels)
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


def build_pretrain_dataloader(
    config: DataConfig,
    tokenizer: Any,
    image_processor: CLIPImageProcessor,
) -> DataLoader:
    """Build DataLoader for LLaVA pretraining.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer for language model
        image_processor: Image processor for vision encoder
        
    Returns:
        DataLoader instance
    """
    dataset = LLaVAPretrainDataset(
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
