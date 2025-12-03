"""Utilities for DDP synchronization."""
from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist


@contextmanager
def ddp_synchronized(ddp_enabled: bool = True):
    """Context manager to ensure DDP synchronization at exit points.
    
    This context manager ensures that all ranks stay synchronized even when
    errors occur or early returns happen. It automatically calls dist.barrier()
    on exit (both normal and exception paths).
    
    Args:
        ddp_enabled: Whether DDP is enabled. If False, this is a no-op.
    
    Example:
        with ddp_synchronized(ddp_enabled=self.ddp_enabled):
            # Training code that may raise exceptions or return early
            if error_condition:
                return  # Barrier will be called automatically
            # Normal execution
    """
    try:
        yield
    finally:
        # Always synchronize on exit, whether normal or exception
        if ddp_enabled and dist.is_initialized():
            dist.barrier()


def get_ddp_enabled() -> bool:
    """Check if DDP is currently enabled.
    
    Returns:
        True if DDP is initialized and world_size > 1, False otherwise.
    """
    return dist.is_initialized() and dist.get_world_size() > 1

