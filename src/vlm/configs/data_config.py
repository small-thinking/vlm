"""Data configuration classes for LLaVA training phases."""
from dataclasses import dataclass


@dataclass
class Phase1DataConfig:
    """Configuration for LLaVA Phase 1 (Pretraining) data.

    Phase 1 requires images to be stored separately, so image_folder
    is required.

    Note: max_length should accommodate visual tokens (e.g., 577 for 336px)
    plus text tokens. Default 1024 works for 336px (577 visual + ~447 text).
    Sequences longer than max_length will be truncated.
    """
    data_path: str
    image_folder: str  # Required for Phase 1
    batch_size: int = 32
    num_workers: int = 4
    max_length: int = 1024  # 336px: 577 visual + ~447 text tokens
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class Phase2DataConfig:
    """Configuration for LLaVA Phase 2 (Instruction Tuning) data.

    Phase 2 loads parquet files from a folder. All .parquet files in the
    folder will be loaded and concatenated. Images are embedded in the
    parquet files.

    Note: max_length should accommodate visual tokens (e.g., 577 for 336px)
    plus text tokens. Default 1024 works for 336px (577 visual + ~447 text).
    Sequences longer than max_length will be truncated.
    """
    data_path: str  # Path to folder containing parquet files
    batch_size: int = 32
    num_workers: int = 4
    max_length: int = 1024  # 336px: 577 visual + ~447 text tokens
    shuffle: bool = True
    drop_last: bool = True


# Backward compatibility alias
DataConfig = Phase1DataConfig
