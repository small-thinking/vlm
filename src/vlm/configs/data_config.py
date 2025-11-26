from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    """Configuration for LLaVA pretraining data."""
    data_path: str
    image_folder: str
    batch_size: int = 32
    num_workers: int = 4
    max_length: int = 512
    shuffle: bool = True
    drop_last: bool = True
