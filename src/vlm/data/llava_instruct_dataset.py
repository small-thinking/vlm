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

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas is required for Phase 2 training. "
        "Install with: pip install pandas pyarrow"
    )

from vlm.configs.data_config import Phase2DataConfig


class LLaVAInstructDataset(Dataset):
    """Dataset for LLaVA Phase 2 instruction tuning.
    
    This dataset is designed to be memory-efficient and supports:
    - Incremental loading: Only loads parquet files on-demand
    - Multi-worker DataLoader: Each worker process has independent state
    - Large datasets: Can handle theoretically infinite datasets
    
    Multi-worker Safety:
    - Each worker process gets its own dataset instance (via pickling)
    - Each worker has its own file cache (no shared state, no race conditions)
    - DataLoader's sampler ensures no data duplication between workers
    - File I/O is read-only and process-safe
    
    Note: Each worker will build its own index during initialization.
    This is redundant but necessary for process isolation.
    
    Args:
        data_path: Path to folder containing parquet files (or single file)
        image_processor: Processor for vision encoder
            (e.g., CLIPImageProcessor)
        tokenizer: Tokenizer for language model
        max_length: Maximum sequence length (includes visual tokens)
        num_visual_tokens: Number of visual tokens from vision encoder.
            If None, will be calculated from image_processor.
            (default 257 for CLIP ViT-L/14 224px: 256 patches + 1 CLS)
    """
    
    def __init__(
        self,
        data_path: str,
        image_processor: Optional[CLIPImageProcessor] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 768,
        num_visual_tokens: Optional[int] = None,
    ):
        data_path_expanded = Path(data_path).expanduser()
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Calculate num_visual_tokens from image processor if not provided
        if num_visual_tokens is None:
            if image_processor is not None:
                # CLIP ViT models use patch_size=14
                # num_patches = (image_size / patch_size)^2
                # num_visual_tokens = num_patches + 1 (CLS token)
                image_size = image_processor.size.get('height', 224)
                patch_size = 14  # Standard for ViT-L/14 models
                num_patches = (image_size // patch_size) ** 2
                self.num_visual_tokens = num_patches + 1
            else:
                # Fallback to default (224px CLIP ViT-L/14)
                self.num_visual_tokens = 257
        else:
            self.num_visual_tokens = num_visual_tokens
        
        # Validate max_length is sufficient for visual tokens
        if self.max_length < self.num_visual_tokens:
            raise ValueError(
                f"max_length ({self.max_length}) is too small for "
                f"{self.num_visual_tokens} visual tokens. "
                f"max_length must be at least {self.num_visual_tokens + 100} "
                f"to leave room for text tokens."
            )
        elif self.max_length < self.num_visual_tokens + 200:
            import warnings
            warnings.warn(
                f"max_length ({self.max_length}) leaves only "
                f"{self.max_length - self.num_visual_tokens} tokens for text "
                f"after reserving {self.num_visual_tokens} visual tokens. "
                f"Consider increasing max_length to at least "
                f"{self.num_visual_tokens + 512} for better text capacity.",
                UserWarning
            )
        
        # Load dataset from parquet files in folder
        print(f"Loading dataset from folder: {data_path_expanded}...")
        
        # Find all parquet files in the folder
        if data_path_expanded.is_file():
            # If it's a file, use it directly
            parquet_files = [data_path_expanded]
        elif data_path_expanded.is_dir():
            # If it's a directory, find all parquet files
            parquet_files = sorted(
                data_path_expanded.glob("*.parquet")
            )
            if not parquet_files:
                raise ValueError(
                    f"No parquet files found in directory: "
                    f"{data_path_expanded}"
                )
        else:
            raise ValueError(
                f"Data path must be a file or directory: {data_path_expanded}"
            )
        
        print(f"Found {len(parquet_files)} parquet file(s)")
        
        # Store parquet file paths instead of loading all data into memory
        # This allows handling theoretically infinite datasets
        self.parquet_files = parquet_files
        
        # Cache for loaded parquet files (lazy loading per file)
        # This avoids re-reading files on every access while still being
        # memory-efficient (only caches files as they're accessed)
        # Note: Each worker process has its own cache (no shared state)
        self._file_cache: Dict[int, pd.DataFrame] = {}
        
        # Build lightweight index: (file_idx, row_idx, turn_start_idx) tuples
        # Only stores indices, not actual data - suitable for huge datasets
        # Note: With multi-worker DataLoader, each worker will build its own
        # index independently. This is redundant but necessary for process isolation.
        print("Building sample index (scanning parquet files)...")
        self.index, total_conversations = self._build_index()
        print(
            f"Indexed {len(self.index)} samples from "
            f"{total_conversations} conversations."
        )
    
    def __len__(self) -> int:
        """Return the number of training samples."""
        return len(self.index)
    
    def _build_index(self) -> tuple[List[tuple], int]:
        """Build lightweight index: (file_idx, row_idx, turn_start_idx) tuples.
        
        Only stores indices, not data. Suitable for huge datasets.
        Processes parquet files incrementally without loading all data.
        
        Supports multiple formats:
        - LLaVA format: conversations = [{"from": "human", "value": "..."},
          {"from": "gpt", "value": "..."}]
        - HuggingFace chat format: messages = [{"role": "user",
          "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            Tuple of (index list, total_conversations count)
        """
        index = []
        total_conversations = 0
        
        # Process each parquet file incrementally
        for file_idx, parquet_file in enumerate(self.parquet_files):
            # Read parquet file in chunks to avoid loading all at once
            # Using iter_batches with a reasonable chunk size
            try:
                df = pd.read_parquet(parquet_file)
                # Process row by row to minimize memory usage
                for row_idx, row in df.iterrows():
                    total_conversations += 1
                    sample = row.to_dict()
                    
                    # Try 'conversations' first (LLaVA format)
                    conversations = sample.get('conversations', None)
                    if isinstance(conversations, str):
                        try:
                            conversations = json.loads(conversations)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    
                    # If not found, try 'messages' (HuggingFace chat format)
                    if conversations is None:
                        messages = sample.get('messages', None)
                        if isinstance(messages, str):
                            try:
                                messages = json.loads(messages)
                            except (json.JSONDecodeError, TypeError):
                                messages = None
                        
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
                                    next_role = (
                                        next_turn.get('from', '') or
                                        next_turn.get('role', '')
                                    )
                                    if next_role in ('gpt', 'assistant'):
                                        # Store: (file_idx, row_idx, turn_start_idx)
                                        index.append((file_idx, row_idx, i))
                                        i += 2
                                        continue
                            i += 1
                        else:
                            i += 1
            except Exception as e:
                print(
                    f"Warning: Error processing {parquet_file.name}: {e}. "
                    f"Skipping file."
                )
                continue
        
        return index, total_conversations
    
    def _process_image(self, image: Any) -> Optional[Image.Image]:
        """Process image from various formats to PIL Image.
        
        Args:
            image: Image in various formats (PIL, bytes, numpy, dict with 'bytes' key, etc.)
            
        Returns:
            PIL Image in RGB format or None if conversion fails
        """
        if image is None:
            return None
        
        try:
            # Handle dictionary format (common in parquet files)
            if isinstance(image, dict):
                # Try 'bytes' key first (common in parquet)
                if 'bytes' in image:
                    image = image['bytes']
                # Try 'image' key
                elif 'image' in image:
                    image = image['image']
                else:
                    # Try to get first value that looks like bytes
                    for key, value in image.items():
                        if isinstance(value, bytes):
                            image = value
                            break
                    else:
                        return None
            
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
        # Get file, row, and turn indices from lightweight index
        file_idx, row_idx, turn_start_idx = self.index[idx]
        
        # Load parquet file on-demand (with caching to avoid re-reading)
        # This is safe for multi-worker DataLoader: each worker process
        # has its own _file_cache instance (no shared state, no race conditions)
        if file_idx not in self._file_cache:
            parquet_file = self.parquet_files[file_idx]
            # Read-only file I/O is process-safe
            self._file_cache[file_idx] = pd.read_parquet(parquet_file)
        
        df = self._file_cache[file_idx]
        row = df.iloc[row_idx]
        sample = row.to_dict()
        
        # Get image from conversation (same for all turns)
        # Images are embedded in parquet files
        raw_image = sample.get('image', None)
        
        conversation_image = None
        if raw_image is not None:
            # Embedded image (bytes, PIL, numpy, etc.)
            conversation_image = self._process_image(raw_image)
        
        # Parse the specific turn on-the-fly
        conversations = sample.get('conversations', None)
        if isinstance(conversations, str):
            try:
                conversations = json.loads(conversations)
            except (json.JSONDecodeError, TypeError):
                conversations = None
        
        # If not found, try 'messages' format (HuggingFace chat format)
        if conversations is None:
            messages = sample.get('messages', [])
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except (json.JSONDecodeError, TypeError):
                    messages = []
            
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
        # Also mask padding tokens
        result['labels'] = full_ids.clone()
        result['labels'][:input_len] = -100  # Mask user input and assistant prompt
        
        # Mask padding tokens (where attention_mask is 0)
        if attention_mask is not None:
            padding_mask = attention_mask == 0
            result['labels'][padding_mask] = -100
        
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
    config: Phase2DataConfig,
    tokenizer: Any,
    image_processor: CLIPImageProcessor,
) -> DataLoader:
    """Build DataLoader for LLaVA instruction tuning.
    
    Args:
        config: Phase 2 data configuration
        tokenizer: Tokenizer for language model
        image_processor: Image processor for vision encoder
        
    Returns:
        DataLoader instance
    """
    dataset = LLaVAInstructDataset(
        data_path=config.data_path,
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

