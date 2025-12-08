"""Diagnostic script to identify NaN gradient causes in Phase 1 training.

This script helps debug why grad_norm becomes NaN when using larger LLMs (4B)
while only training the MLP connector (LLM frozen).

Run with: python scripts/diagnose_nan_grad.py
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlm.models.llava import LLaVAModel
from vlm.configs.model_config import (
    LLaVAConfig,
    VisionEncoderConfig,
    LanguageModelConfig,
    ConnectorConfig,
)


def diagnose_dtype_mismatch(model: LLaVAModel) -> Dict[str, Any]:
    """Check for dtype mismatches between components."""
    results = {
        "vision_encoder_dtype": None,
        "connector_dtype": None,
        "llm_dtype": None,
        "llm_embed_dtype": None,
        "mismatches": [],
    }
    
    # Check vision encoder dtype
    for name, param in model.vision_encoder.named_parameters():
        results["vision_encoder_dtype"] = param.dtype
        break
    
    # Check connector dtype
    for name, param in model.connector.named_parameters():
        results["connector_dtype"] = param.dtype
        break
    
    # Check LLM dtype
    for name, param in model.language_model.named_parameters():
        results["llm_dtype"] = param.dtype
        break
    
    # Check LLM embedding layer dtype specifically
    embed_layer = model.language_model.get_input_embeddings()
    results["llm_embed_dtype"] = embed_layer.weight.dtype
    
    # Identify mismatches
    dtypes = {
        "vision_encoder": results["vision_encoder_dtype"],
        "connector": results["connector_dtype"],
        "llm": results["llm_dtype"],
        "llm_embed": results["llm_embed_dtype"],
    }
    
    unique_dtypes = set(dtypes.values())
    if len(unique_dtypes) > 1:
        results["mismatches"] = [
            f"{k}: {v}" for k, v in dtypes.items()
        ]
    
    return results


def diagnose_embedding_stats(
    model: LLaVAModel,
    device: torch.device,
) -> Dict[str, Any]:
    """Analyze embedding statistics for potential numerical issues."""
    results = {}
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 32
    
    # Get a sample vocabulary token range
    vocab_size = model.language_model.model.config.vocab_size
    input_ids = torch.randint(100, min(vocab_size, 10000), (batch_size, seq_len))
    input_ids = input_ids.to(device)
    
    # Get text embeddings
    with torch.no_grad():
        embed_layer = model.language_model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)
    
    results["text_embed_shape"] = list(text_embeds.shape)
    results["text_embed_dtype"] = str(text_embeds.dtype)
    results["text_embed_mean"] = text_embeds.mean().item()
    results["text_embed_std"] = text_embeds.std().item()
    results["text_embed_min"] = text_embeds.min().item()
    results["text_embed_max"] = text_embeds.max().item()
    results["text_embed_has_nan"] = torch.isnan(text_embeds).any().item()
    results["text_embed_has_inf"] = torch.isinf(text_embeds).any().item()
    
    # Check per-token statistics
    token_stds = text_embeds.std(dim=-1)  # (batch, seq_len)
    results["min_token_std"] = token_stds.min().item()
    results["max_token_std"] = token_stds.max().item()
    results["zero_std_tokens"] = (token_stds == 0).sum().item()
    
    return results


def diagnose_forward_pass(
    model: LLaVAModel,
    device: torch.device,
    precision: str = "bf16",
) -> Dict[str, Any]:
    """Run forward pass and check for numerical issues."""
    results = {"steps": []}
    
    model.to(device)
    model.set_training_stage(1)  # Freeze LLM, train connector
    model.train()
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 32
    image_size = model.vision_encoder.image_size
    
    # Dummy image (3 channels, image_size x image_size)
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    
    # Dummy text tokens
    vocab_size = model.language_model.model.config.vocab_size
    input_ids = torch.randint(100, min(vocab_size, 10000), (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = input_ids.clone()
    labels[:, :5] = -100  # Mask some tokens
    
    # Determine autocast dtype
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    
    # Step 1: Vision encoder output
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            vision_out = model.vision_encoder(images)
    
    results["steps"].append({
        "name": "vision_encoder_output",
        "shape": list(vision_out.shape),
        "dtype": str(vision_out.dtype),
        "mean": vision_out.float().mean().item(),
        "std": vision_out.float().std().item(),
        "has_nan": torch.isnan(vision_out).any().item(),
        "has_inf": torch.isinf(vision_out).any().item(),
    })
    
    # Step 2: Connector output
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        connector_out = model.connector(vision_out)
    
    results["steps"].append({
        "name": "connector_output",
        "shape": list(connector_out.shape),
        "dtype": str(connector_out.dtype),
        "mean": connector_out.float().mean().item(),
        "std": connector_out.float().std().item(),
        "has_nan": torch.isnan(connector_out).any().item(),
        "has_inf": torch.isinf(connector_out).any().item(),
    })
    
    # Step 3: Text embeddings
    with torch.no_grad():
        embed_layer = model.language_model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)
    
    results["steps"].append({
        "name": "text_embeddings",
        "shape": list(text_embeds.shape),
        "dtype": str(text_embeds.dtype),
        "mean": text_embeds.float().mean().item(),
        "std": text_embeds.float().std().item(),
        "has_nan": torch.isnan(text_embeds).any().item(),
        "has_inf": torch.isinf(text_embeds).any().item(),
    })
    
    # Step 4: Full forward pass with loss
    model.zero_grad()
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
    
    results["steps"].append({
        "name": "loss",
        "value": loss.item() if torch.isfinite(loss) else "non-finite",
        "dtype": str(loss.dtype),
        "has_nan": torch.isnan(loss).item(),
        "has_inf": torch.isinf(loss).item(),
    })
    
    # Step 5: Backward pass
    loss.backward()
    
    # Check gradients
    grad_info = []
    total_grad_norm = 0.0
    has_nan_grad = False
    has_inf_grad = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_norm = grad.norm().item()
            total_grad_norm += grad_norm ** 2
            
            is_nan = torch.isnan(grad).any().item()
            is_inf = torch.isinf(grad).any().item()
            
            if is_nan:
                has_nan_grad = True
            if is_inf:
                has_inf_grad = True
            
            # Only log problematic gradients or connector gradients
            if is_nan or is_inf or "connector" in name:
                grad_info.append({
                    "name": name,
                    "shape": list(param.shape),
                    "grad_norm": grad_norm if not (is_nan or is_inf) else "nan/inf",
                    "grad_mean": grad.float().mean().item() if not is_nan else "nan",
                    "grad_std": grad.float().std().item() if not is_nan else "nan",
                    "has_nan": is_nan,
                    "has_inf": is_inf,
                })
    
    total_grad_norm = total_grad_norm ** 0.5
    
    results["steps"].append({
        "name": "gradients",
        "total_grad_norm": total_grad_norm if not (has_nan_grad or has_inf_grad) else "nan/inf",
        "has_nan_grad": has_nan_grad,
        "has_inf_grad": has_inf_grad,
        "problematic_params": grad_info,
    })
    
    return results


def diagnose_normalization_issue(
    model: LLaVAModel,
    device: torch.device,
) -> Dict[str, Any]:
    """Check if the embedding normalization in llava.py could cause issues."""
    results = {}
    
    model.to(device)
    model.set_training_stage(1)
    model.train()
    
    batch_size = 2
    seq_len = 32
    image_size = model.vision_encoder.image_size
    
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    vocab_size = model.language_model.model.config.vocab_size
    input_ids = torch.randint(100, min(vocab_size, 10000), (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        # Get text embeddings
        embed_layer = model.language_model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)
        
        # Get visual embeddings (after connector)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            visual_features = model.vision_encoder(images)
            visual_embeds = model.connector(visual_features)
    
    # Compute statistics as done in llava.py forward()
    text_mean = text_embeds.mean(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
    text_std = text_embeds.std(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
    visual_mean = visual_embeds.mean(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
    visual_std = visual_embeds.std(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
    
    results["text_mean"] = text_mean.item()
    results["text_std"] = text_std.item()
    results["visual_mean"] = visual_mean.item()
    results["visual_std"] = visual_std.item()
    
    # Check the problematic condition
    results["text_std_is_tensor"] = isinstance(text_std, torch.Tensor)
    results["text_std_gt_zero"] = (text_std > 0).item()  # This is how it's used
    results["visual_std_gt_zero"] = (visual_std > 0).item()
    
    # Simulate the normalization
    if text_std > 0 and visual_std > 0:
        normalized = (
            (visual_embeds - visual_mean) / (visual_std + 1e-8)
        ) * (text_std + 1e-8) + text_mean
        
        results["normalized_has_nan"] = torch.isnan(normalized).any().item()
        results["normalized_has_inf"] = torch.isinf(normalized).any().item()
        results["normalized_mean"] = normalized.float().mean().item()
        results["normalized_std"] = normalized.float().std().item()
    
    return results


def main():
    """Run all diagnostics."""
    print("=" * 60)
    print("VLM Phase 1 NaN Gradient Diagnostics")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires GPU.")
        return
    
    device = torch.device("cuda")
    print(f"\nDevice: {device}")
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
    
    # Create model with 4B config
    print("\n" + "-" * 40)
    print("Loading Model (Qwen3-4B-Instruct)...")
    print("-" * 40)
    
    config = LLaVAConfig(
        vision_encoder=VisionEncoderConfig(
            model_name="openai/clip-vit-large-patch14-336",
            freeze=True,
        ),
        language_model=LanguageModelConfig(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            freeze=True,
            torch_dtype=None,  # Let it auto-detect (bf16)
        ),
        connector=ConnectorConfig(
            num_layers=2,
            hidden_dim=1024,
            activation="gelu",
        ),
    )
    
    model = LLaVAModel(config)
    
    # Diagnostic 1: dtype mismatch
    print("\n" + "-" * 40)
    print("1. Checking dtype mismatches...")
    print("-" * 40)
    dtype_results = diagnose_dtype_mismatch(model)
    for key, value in dtype_results.items():
        print(f"  {key}: {value}")
    
    if dtype_results["mismatches"]:
        print("\n  ⚠️  DTYPE MISMATCH DETECTED!")
        print("  This is likely causing NaN gradients.")
        print("  The connector (fp32) and LLM (bf16) have different dtypes.")
    
    # Diagnostic 2: Embedding statistics
    print("\n" + "-" * 40)
    print("2. Checking embedding statistics...")
    print("-" * 40)
    model.to(device)
    embed_results = diagnose_embedding_stats(model, device)
    for key, value in embed_results.items():
        print(f"  {key}: {value}")
    
    # Diagnostic 3: Normalization analysis
    print("\n" + "-" * 40)
    print("3. Checking normalization in forward pass...")
    print("-" * 40)
    norm_results = diagnose_normalization_issue(model, device)
    for key, value in norm_results.items():
        print(f"  {key}: {value}")
    
    # Diagnostic 4: Full forward/backward pass
    print("\n" + "-" * 40)
    print("4. Running forward/backward pass...")
    print("-" * 40)
    
    for precision in ["bf16", "fp16"]:
        print(f"\n  Testing with {precision}:")
        try:
            fwd_results = diagnose_forward_pass(model, device, precision)
            for step in fwd_results["steps"]:
                name = step.pop("name")
                print(f"    {name}:")
                for k, v in step.items():
                    if k == "problematic_params" and v:
                        print(f"      {k}:")
                        for param_info in v:
                            print(f"        - {param_info}")
                    else:
                        print(f"      {k}: {v}")
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Summary and recommendations
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    if dtype_results["mismatches"]:
        print("""
⚠️  ROOT CAUSE: dtype mismatch between connector (fp32) and LLM (bf16)

SOLUTION: Convert connector to the same dtype as the LLM before training.
Add this to LLaVAModel.__init__() or Phase1Trainer:

    # Match connector dtype to LLM dtype
    llm_dtype = next(self.language_model.parameters()).dtype
    self.connector.to(dtype=llm_dtype)
    
Or explicitly set connector dtype in the config.
""")
    
    print("""
Additional checks to perform:
1. Ensure vision encoder output is cast to LLM dtype before connector
2. Consider disabling the embedding normalization temporarily to test
3. Try fp32 training (precision="fp32") to isolate mixed precision issues
4. Check if GradScaler is properly handling scale updates
""")


if __name__ == "__main__":
    main()

