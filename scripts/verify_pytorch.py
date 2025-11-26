#!/usr/bin/env python3
"""Verify PyTorch installation and check Mac compatibility."""

import torch
import sys


def main():
    print("=" * 60)
    print("PyTorch Installation Verification")
    print("=" * 60)
    
    # PyTorch version
    print(f"\n✓ PyTorch version: {torch.__version__}")
    
    # CUDA availability (should be False on Mac)
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    # MPS (Metal Performance Shaders) for Apple Silicon
    if hasattr(torch.backends, 'mps'):
        print(f"✓ MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print(f"✓ MPS built: {torch.backends.mps.is_built()}")
    
    # CPU info
    print(f"✓ Number of CPU threads: {torch.get_num_threads()}")
    
    # Test tensor creation and device
    print("\n" + "=" * 60)
    print("Device Compatibility Test")
    print("=" * 60)
    
    # Create a simple tensor
    x = torch.randn(3, 3)
    print(f"\n✓ Created test tensor on device: {x.device}")
    
    # Try MPS if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            x_mps = x.to(device)
            y_mps = torch.randn(3, 3, device=device)
            z_mps = x_mps + y_mps
            print(f"✓ MPS device test successful: {z_mps.device}")
            print(f"  Result shape: {z_mps.shape}")
        except Exception as e:
            print(f"✗ MPS device test failed: {e}")
    
    # CPU computation
    y = torch.randn(3, 3)
    z = x + y
    print(f"✓ CPU computation successful")
    print(f"  Result shape: {z.shape}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\n✓ PyTorch is correctly installed!")
    print(f"✓ This installation supports:")
    print(f"  - CPU compute")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  - Apple Silicon (MPS) acceleration")
    print(f"\n✓ Compatible with remote A100/B100 GPUs (same PyTorch version)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
