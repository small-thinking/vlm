"""Data configuration classes for LLaVA training phases."""
from dataclasses import dataclass


@dataclass
class Phase1DataConfig:
    """Configuration for LLaVA Phase 1 (Pretraining) data.

    Phase 1 requires images to be stored separately, so image_folder
    is required.
    """
    data_path: str
    image_folder: str  # Required for Phase 1
    batch_size: int = 32
    num_workers: int = 4
    max_length: int = 512
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class Phase2DataConfig:
    """Configuration for LLaVA Phase 2 (Instruction Tuning) data.

    Phase 2 loads parquet files from a folder. All .parquet files in the
    folder will be loaded and concatenated. Images are embedded in the
    parquet files.
    """
    data_path: str  # Path to folder containing parquet files
    batch_size: int = 32
    num_workers: int = 4
    max_length: int = 768  # Longer sequences for instruction tuning
    shuffle: bool = True
    drop_last: bool = True


# Backward compatibility alias
DataConfig = Phase1DataConfig
