"""Utility functions module."""
from .ddp_sync import ddp_synchronized, get_ddp_enabled

__all__ = ["ddp_synchronized", "get_ddp_enabled"]
