#!/usr/bin/env python3
"""Verify fp16 support for both vision encoder and language model.

This script checks if:
1. CLIP vision encoder supports fp16 training
2. Qwen 2.5 language model supports fp16 training
3. Both models can perform forward/backward passes in fp16
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM


def check_model_config(model: nn.Module, model_name: str) -> dict:
    """Check model configuration for fp16 support indicators.
    
    Args:
        model: The model to check
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with configuration information
    """
    config_info = {
        "has_config": False,
        "torch_dtype": None,
        "model_type": None,
        "supports_fp16_theoretical": True,  # Most transformers models support fp16
    }
    
    if hasattr(model, 'config'):
        config_info["has_config"] = True
        config = model.config
        config_info["model_type"] = getattr(config, 'model_type', None)
        
        # Check if model has torch_dtype attribute
        if hasattr(config, 'torch_dtype'):
            config_info["torch_dtype"] = str(config.torch_dtype)
    
    # Check current model dtype
    try:
        first_param = next(model.parameters())
        config_info["current_dtype"] = str(first_param.dtype)
    except StopIteration:
        config_info["current_dtype"] = "unknown"
    
    return config_info


def check_model_fp16_support(model: nn.Module, model_name: str) -> dict:
    """Check if a model supports fp16 training.
    
    Args:
        model: The model to check
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with fp16 support information
    """
    results = {
        "model_name": model_name,
        "supports_fp16": False,
        "can_convert_to_half": False,
        "can_forward_pass": False,
        "can_backward_pass": False,
        "config_info": None,
        "error": None
    }
    
    # First, check model configuration
    results["config_info"] = check_model_config(model, model_name)
    
    # Try converting to half precision (works on CPU too, just for testing)
    try:
        model_half = model.half()
        first_param = next(model_half.parameters())
        if first_param.dtype == torch.float16:
            results["can_convert_to_half"] = True
            results["supports_fp16"] = True
    except Exception as e:
        results["error"] = f"Cannot convert to half precision: {str(e)}"
        return results
    
    # Check if CUDA is available for actual training tests
    if not torch.cuda.is_available():
        results["error"] = "CUDA not available - fp16 training requires GPU"
        return results
    
    device = torch.device("cuda:0")
    
    try:
        # Try converting model to half precision
        model_half = model.half()
        model_half = model_half.to(device)
        results["can_convert_to_half"] = True
        
        # Check if model parameters are in fp16
        first_param = next(model_half.parameters())
        if first_param.dtype == torch.float16:
            results["supports_fp16"] = True
        
        # Test forward pass with dummy input
        if "CLIP" in model_name:
            # CLIP vision encoder expects pixel_values
            batch_size = 2
            dummy_input = torch.randn(
                batch_size, 3, 224, 224,
                dtype=torch.float16,
                device=device
            )
            try:
                with torch.no_grad():
                    output = model_half(pixel_values=dummy_input)
                results["can_forward_pass"] = True
            except Exception as e:
                results["error"] = f"Forward pass failed: {str(e)}"
        else:
            # Language model expects input_ids or inputs_embeds
            batch_size = 2
            seq_len = 10
            hidden_size = model.config.hidden_size
            
            # Test with inputs_embeds (used in LLaVA)
            dummy_embeds = torch.randn(
                batch_size, seq_len, hidden_size,
                dtype=torch.float16,
                device=device
            )
            dummy_mask = torch.ones(
                batch_size, seq_len,
                dtype=torch.long,
                device=device
            )
            try:
                with torch.no_grad():
                    output = model_half(
                        input_ids=None,
                        attention_mask=dummy_mask,
                        inputs_embeds=dummy_embeds
                    )
                results["can_forward_pass"] = True
            except Exception as e:
                results["error"] = f"Forward pass failed: {str(e)}"
        
        # Test backward pass if forward pass succeeded
        if results["can_forward_pass"]:
            try:
                model_half.train()
                if "CLIP" in model_name:
                    dummy_input = torch.randn(
                        batch_size, 3, 224, 224,
                        dtype=torch.float16,
                        device=device
                    )
                    output = model_half(pixel_values=dummy_input)
                    # Create a dummy loss
                    if hasattr(output, 'last_hidden_state'):
                        loss = output.last_hidden_state.mean()
                    else:
                        loss = output.pooler_output.mean()
                else:
                    dummy_embeds = torch.randn(
                        batch_size, seq_len, hidden_size,
                        dtype=torch.float16,
                        device=device
                    )
                    dummy_mask = torch.ones(
                        batch_size, seq_len,
                        dtype=torch.long,
                        device=device
                    )
                    dummy_labels = torch.randint(
                        0, model.config.vocab_size,
                        (batch_size, seq_len),
                        device=device
                    )
                    output = model_half(
                        input_ids=None,
                        attention_mask=dummy_mask,
                        inputs_embeds=dummy_embeds,
                        labels=dummy_labels
                    )
                    loss = output.loss
                
                loss.backward()
                results["can_backward_pass"] = True
            except Exception as e:
                results["error"] = f"Backward pass failed: {str(e)}"
        
    except Exception as e:
        results["error"] = f"Error checking fp16 support: {str(e)}"
    
    return results


def main():
    """Main function to check fp16 support for both models."""
    print("=" * 70)
    print("Checking FP16 Support for LLaVA Models")
    print("=" * 70)
    print()
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. FP16 training requires GPU.")
        print("   This check will be limited to conversion tests only.")
        print()
    
    # Check CLIP Vision Encoder
    print("1. Checking CLIP Vision Encoder (openai/clip-vit-large-patch14)...")
    clip_results = {}
    try:
        clip_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        clip_results = check_model_fp16_support(
            clip_model,
            "CLIP Vision Encoder"
        )
        print(f"   ✓ Model loaded successfully")
        
        # Print config info
        if clip_results.get('config_info'):
            cfg = clip_results['config_info']
            print(f"   - Model type: {cfg.get('model_type', 'unknown')}")
            print(f"   - Current dtype: {cfg.get('current_dtype', 'unknown')}")
        
        print(f"   - Can convert to half precision: {clip_results['can_convert_to_half']}")
        print(f"   - Supports fp16: {clip_results['supports_fp16']}")
        if torch.cuda.is_available():
            print(f"   - Can perform forward pass: {clip_results['can_forward_pass']}")
            print(f"   - Can perform backward pass: {clip_results['can_backward_pass']}")
        if clip_results['error']:
            print(f"   ⚠️  Error: {clip_results['error']}")
        print()
    except Exception as e:
        print(f"   ❌ Failed to load CLIP model: {e}")
        print()
    
    # Check Qwen 2.5 Language Model
    print("2. Checking Qwen 2.5 Language Model (Qwen/Qwen2.5-1.5B)...")
    qwen_results = {}
    try:
        qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B"
        )
        qwen_results = check_model_fp16_support(
            qwen_model,
            "Qwen 2.5 Language Model"
        )
        print(f"   ✓ Model loaded successfully")
        
        # Print config info
        if qwen_results.get('config_info'):
            cfg = qwen_results['config_info']
            print(f"   - Model type: {cfg.get('model_type', 'unknown')}")
            print(f"   - Current dtype: {cfg.get('current_dtype', 'unknown')}")
        
        print(f"   - Can convert to half precision: {qwen_results['can_convert_to_half']}")
        print(f"   - Supports fp16: {qwen_results['supports_fp16']}")
        if torch.cuda.is_available():
            print(f"   - Can perform forward pass: {qwen_results['can_forward_pass']}")
            print(f"   - Can perform backward pass: {qwen_results['can_backward_pass']}")
        if qwen_results['error']:
            print(f"   ⚠️  Error: {qwen_results['error']}")
        print()
    except Exception as e:
        print(f"   ❌ Failed to load Qwen model: {e}")
        print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    # Check conversion capability (works even without CUDA)
    clip_can_convert = clip_results.get('can_convert_to_half', False)
    qwen_can_convert = qwen_results.get('can_convert_to_half', False)
    
    # Check full training support (requires CUDA)
    if torch.cuda.is_available():
        clip_ok = (
            clip_can_convert and
            clip_results.get('can_forward_pass', False) and
            clip_results.get('can_backward_pass', False)
        )
        qwen_ok = (
            qwen_can_convert and
            qwen_results.get('can_forward_pass', False) and
            qwen_results.get('can_backward_pass', False)
        )
    else:
        # Without CUDA, we can only check conversion
        clip_ok = clip_can_convert
        qwen_ok = qwen_can_convert
    
    print(f"CLIP Vision Encoder: {'✅ SUPPORTS FP16' if clip_ok else '⚠️  PARTIAL/UNKNOWN'}")
    print(f"   - Can convert to fp16: {clip_can_convert}")
    if torch.cuda.is_available():
        print(f"   - Full training test: {clip_ok}")
    else:
        print(f"   - Full training test: ⚠️  Requires CUDA")
    
    print(f"Qwen 2.5 Language Model: {'✅ SUPPORTS FP16' if qwen_ok else '⚠️  PARTIAL/UNKNOWN'}")
    print(f"   - Can convert to fp16: {qwen_can_convert}")
    if torch.cuda.is_available():
        print(f"   - Full training test: {qwen_ok}")
    else:
        print(f"   - Full training test: ⚠️  Requires CUDA")
    print()
    
    # Provide recommendations
    if clip_can_convert and qwen_can_convert:
        print("✅ Both models can be converted to fp16!")
        if torch.cuda.is_available():
            if clip_ok and qwen_ok:
                print("✅ Both models fully support fp16 training!")
                print("   You can enable mixed precision training using:")
                print("   - torch.cuda.amp.autocast() for automatic mixed precision (recommended)")
                print("   - model.half() for full fp16 (less stable, faster)")
            else:
                print("⚠️  Conversion works, but training tests had issues.")
                print("   Consider using torch.cuda.amp.autocast() for safer mixed precision.")
        else:
            print("⚠️  Conversion works, but full training test requires CUDA.")
            print("   On a GPU system, both models should support fp16 training.")
            print("   Recommended: Use torch.cuda.amp.autocast() for mixed precision.")
    elif clip_can_convert or qwen_can_convert:
        print("⚠️  Only one model can be converted to fp16.")
        print("   Mixed precision training may have compatibility issues.")
    else:
        print("❌ Neither model can be converted to fp16.")
        print("   Consider using bf16 (bfloat16) or fp32 training instead.")
    print()
    
    # Additional notes
    print("Note: Most HuggingFace transformers models support fp16 training.")
    print("      CLIP and Qwen models are known to work well with fp16/AMP.")
    print()


if __name__ == "__main__":
    main()

