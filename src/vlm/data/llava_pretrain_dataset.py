"""LLaVA Pretrain Dataset for Phase 1 training.

Dataset: liuhaotian/LLaVA-Pretrain (~558K samples)

Setup:
  ./setup.sh  # Will prompt to download dataset

Usage:
    from vlm.data import LLaVAPretrainDataset, collate_fn
    
    dataset = LLaVAPretrainDataset(
        data_path="./dataset/llava-pretrain/blip_laion_cc_sbu_558k.json",
        image_folder="./dataset/llava-pretrain",
        image_processor=clip_processor,
        tokenizer=tokenizer,
    )
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor

from vlm.configs.data_config import DataConfig


class LLaVAPretrainDataset(Dataset):
    """Dataset for LLaVA Phase 1 pretraining (vision-language alignment).
    
    This dataset is used to train the connector/projection layer while keeping
    the vision encoder and language model frozen.
    
    Args:
        data_path: Path to the dataset JSON file or HuggingFace dataset name
        image_folder: Path to the folder containing images
        image_processor: Processor for vision encoder (e.g., CLIPImageProcessor)
        tokenizer: Tokenizer for language model
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        image_processor: Optional[CLIPImageProcessor] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
    ):
        self.data_path = data_path
        self.image_folder = Path(image_folder)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset from JSON file or HuggingFace.
        
        Returns:
            List of data samples
        """
        # Check if data_path is a local file
        if os.path.isfile(self.data_path):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
        # Otherwise try loading from HuggingFace
        else:
            try:
                from datasets import load_dataset
                dataset = load_dataset(self.data_path, split='train')
                data = list(dataset)
            except Exception as e:
                raise ValueError(
                    f"Could not load data from {self.data_path}. "
                    f"Error: {e}"
                )
        
        print(f"Loaded {len(data)} samples from {self.data_path}")
        return data
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def _parse_conversation(self, conversations: List[Dict[str, str]]) -> Dict[str, str]:
        """Parse conversation into human prompt and assistant response.
        
        Args:
            conversations: List of conversation turns
            
        Returns:
            Dictionary with 'human' and 'gpt' keys
        """
        human_text = ""
        gpt_text = ""
        
        for conv in conversations:
            if conv["from"] == "human":
                # Remove <image> token from text for now
                # We'll handle image separately
                human_text = conv["value"].replace("<image>", "").strip()
            elif conv["from"] == "gpt":
                gpt_text = conv["value"]
        
        return {
            "human": human_text,
            "gpt": gpt_text
        }
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from file path.
        
        Args:
            image_path: Relative path to image from data
            
        Returns:
            PIL Image
        """
        full_path = self.image_folder / image_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        image = Image.open(full_path).convert('RGB')
        return image
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - image: Processed image tensor (if image_processor provided)
                - input_ids: Tokenized input text (if tokenizer provided)
                - attention_mask: Attention mask for input
                - labels: Tokenized labels for language modeling
                - raw_image: PIL Image (if no processor)
                - raw_text: Raw text strings (if no tokenizer)
        """
        sample = self.data[idx]
        
        # Get image path and load image
        image_path = sample.get('image', '')
        image = self._load_image(image_path)
        
        # Parse conversation
        conversations = sample.get('conversations', [])
        # Handle case where conversations might be a JSON string
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        
        parsed_conv = self._parse_conversation(conversations)
        human_text = parsed_conv['human']
        gpt_text = parsed_conv['gpt']
        
        # Create the input text (question) and target text (answer)
        # Format: "Human: {question}\nAssistant: {answer}"
        input_text = f"Human: {human_text}\nAssistant:"
        target_text = f" {gpt_text}"
        
        result = {
            'sample_id': sample.get('id', idx),
            'image_path': image_path,
        }
        
        # Process image if processor is provided
        if self.image_processor is not None:
            processed_image = self.image_processor(
                images=image,
                return_tensors='pt'
            )
            result['pixel_values'] = processed_image['pixel_values'].squeeze(0)
        else:
            result['raw_image'] = image
        
        # Process text if tokenizer is provided
        if self.tokenizer is not None:
            # Tokenize input (question)
            input_encoding = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize full text (question + answer) for labels
            full_text = input_text + target_text
            labels_encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            result['input_ids'] = input_encoding['input_ids'].squeeze(0)
            result['attention_mask'] = input_encoding['attention_mask'].squeeze(0)
            result['labels'] = labels_encoding['input_ids'].squeeze(0)
            
            # Mask out the input part in labels (we only want to predict the answer)
            input_len = input_encoding['input_ids'].shape[1]
            result['labels'][:input_len] = -100
        else:
            result['raw_text'] = {
                'input': input_text,
                'target': target_text,
                'full': input_text + target_text
            }
        
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched tensors
    """
    # Collect all keys from the first item
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key in ['sample_id', 'image_path']:
            # Keep as list for metadata
            collated[key] = [item[key] for item in batch]
        elif key == 'raw_image':
            # Keep images as list
            collated[key] = [item[key] for item in batch]
        elif key == 'raw_text':
            # Keep text as list of dicts
            collated[key] = [item[key] for item in batch]
        else:
            # Stack tensors
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


def build_dataloader(
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
