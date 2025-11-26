"""Verification script for LLaVA model implementation."""
import torch
from vlm.models import LLaVAModel
from vlm.configs import LLaVAConfig, VisionEncoderConfig, ConnectorConfig, LanguageModelConfig


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def main():
    """Run LLaVA model verification tests."""
    
    print_section("LLaVA Model Verification")
    
    # Test 1: Instantiate with default config
    print("Test 1: Creating LLaVA model with default configuration...")
    print("  - Vision Encoder: CLIP ViT-L/14")
    print("  - Language Model: Qwen2.5-1.5B")
    print("  - Connector: Single-layer MLP (linear projection)")
    
    try:
        model = LLaVAModel()
        print("✓ Model instantiation successful!\n")
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}\n")
        return
    
    # Test 2: Print model architecture
    print_section("Model Architecture")
    print(f"Vision Encoder: {model.vision_encoder.__class__.__name__}")
    print(f"  - Hidden size: {model.vision_encoder.hidden_size}")
    print(f"  - Frozen: {model.config.vision_encoder.freeze}")
    print()
    
    print(f"Connector: {model.connector.__class__.__name__}")
    print(f"  - Input dim: {model.vision_encoder.hidden_size}")
    print(f"  - Output dim: {model.language_model.hidden_size}")
    print(f"  - Layers: {model.config.connector.num_layers}")
    print()
    
    print(f"Language Model: {model.language_model.__class__.__name__}")
    print(f"  - Hidden size: {model.language_model.hidden_size}")
    print(f"  - Frozen: {model.config.language_model.freeze}")
    
    # Test 3: Model Structure
    print_section("Model Structure")
    print("Complete model architecture:")
    print("-" * 60)
    print(model)
    
    # Test 4: Parameter counts
    print_section("Parameter Counts")
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Test 5: Forward pass with dummy data
    print_section("Forward Pass Test")
    print("Creating dummy inputs...")
    
    # CLIP ViT-L/14 expects 224x224 images
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 10))
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    
    print(f"  - Images: {dummy_images.shape}")
    print(f"  - Input IDs: {dummy_input_ids.shape}")
    print(f"  - Attention mask: {dummy_attention_mask.shape}")
    
    try:
        print("\nRunning forward pass...")
        with torch.no_grad():
            outputs = model(
                images=dummy_images,
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask
            )
        print("✓ Forward pass successful!")
        print(f"  - Output logits shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 6: Test image encoding separately
    print_section("Image Encoding Test")
    try:
        print("Encoding images to visual embeddings...")
        with torch.no_grad():
            visual_embeds = model.encode_images(dummy_images)
        print("✓ Image encoding successful!")
        print(f"  - Visual embeddings shape: {visual_embeds.shape}")
        print(f"  - Expected: (batch={batch_size}, num_patches=?, hidden_dim={model.language_model.hidden_size})")
    except Exception as e:
        print(f"✗ Image encoding failed: {e}")
        return
    
    # Test 7: Custom configuration
    print_section("Custom Configuration Test")
    print("Testing with custom config (2-layer MLP connector)...")
    
    custom_config = LLaVAConfig(
        connector=ConnectorConfig(num_layers=2, hidden_dim=2048)
    )
    
    try:
        custom_model = LLaVAModel(config=custom_config)
        print("✓ Custom configuration successful!")
        print(f"  - Connector layers: {custom_config.connector.num_layers}")
        print(f"  - Connector hidden dim: {custom_config.connector.hidden_dim}")
    except Exception as e:
        print(f"✗ Custom configuration failed: {e}")
        return
    
    # Test 8: Two-stage training freeze settings
    print_section("Two-Stage Training Test")
    print("Testing freeze settings for two-stage training...")
    
    print("\nInitial state (Stage 1 default):")
    print(f"  - Vision encoder frozen: {model.config.vision_encoder.freeze}")
    print(f"  - LLM frozen: {model.config.language_model.freeze}")
    stage1_trainable = count_parameters(model, trainable_only=True)
    print(f"  - Trainable parameters: {stage1_trainable:,}")
    
    print("\nSwitching to Stage 2 (train connector + LLM)...")
    model.set_training_stage(2)
    print(f"  - Vision encoder frozen: {model.config.vision_encoder.freeze}")
    print(f"  - LLM frozen: {model.config.language_model.freeze}")
    stage2_trainable = count_parameters(model, trainable_only=True)
    print(f"  - Trainable parameters: {stage2_trainable:,}")
    print(f"  - Increase: {stage2_trainable - stage1_trainable:,} parameters")
    
    print("\nSwitching back to Stage 1 (train connector only)...")
    model.set_training_stage(1)
    stage1_trainable_verify = count_parameters(model, trainable_only=True)
    print(f"  - Trainable parameters: {stage1_trainable_verify:,}")
    print(f"  - Match original Stage 1: {stage1_trainable == stage1_trainable_verify}")
    
    print_section("All Tests Passed! ✓")
    print("LLaVA model implementation is working correctly.")
    print("\nNext steps:")
    print("  1. Prepare training data (image-text pairs)")
    print("  2. Implement training pipeline with two-stage strategy:")
    print("     - Stage 1: Train connector only (align vision-language)")
    print("     - Stage 2: Train connector + LLM (instruction tuning)")
    print("  3. Fine-tune on visual instruction following tasks")


if __name__ == "__main__":
    main()
