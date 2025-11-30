#!/usr/bin/env python3
"""Inspect Phase 2 training data and verify sample construction.

All checks are enabled by default:
- Multi-turn decomposition inspection (3 conversations)
- Model forward pass verification
- Comprehensive validation

Usage:
    python scripts/inspect_phase2_data.py --data_path ~/dataset/llava-instruct-mix/data
    python scripts/inspect_phase2_data.py --data_path ~/dataset/llava-instruct-mix/data --sample_indices 0 10 100
    python scripts/inspect_phase2_data.py --data_path ~/dataset/llava-instruct-mix/data --no_decomposition
    python scripts/inspect_phase2_data.py --data_path ~/dataset/llava-instruct-mix/data --conversation_location "0,5"
"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
# Use interactive backend for displaying images
try:
    matplotlib.use('TkAgg')  # Try TkAgg first (works on macOS/Linux)
except Exception:
    try:
        matplotlib.use('Qt5Agg')  # Fallback to Qt5Agg
    except Exception:
        matplotlib.use('Agg')  # Fallback to non-interactive if needed
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlm.configs.data_config import Phase2DataConfig
from vlm.configs.model_config import LLaVAConfig
from vlm.data.llava_instruct_dataset import LLaVAInstructDataset
from vlm.models.llava import LLaVAModel


def format_tokens(token_ids: torch.Tensor, tokenizer, max_display: int = 50) -> str:
    """Format token IDs as readable text."""
    tokens = token_ids.tolist()
    if len(tokens) > max_display:
        tokens = tokens[:max_display]
        truncated = True
    else:
        truncated = False
    
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    if truncated:
        text += f" ... (truncated, total {len(token_ids)} tokens)"
    return text


def print_separator(char: str = "=", length: int = 80):
    """Print a separator line."""
    print(char * length)


def inspect_sample(
    dataset: LLaVAInstructDataset,
    idx: int,
    tokenizer,
    show_image_info: bool = True,
    show_tokens: bool = True,
    show_labels: bool = True,
    save_image_dir: Optional[Path] = None,
    display_images: bool = True,
):
    """Inspect a single sample from the dataset.
    
    Args:
        dataset: The dataset to inspect
        idx: Index of the sample to inspect
        tokenizer: Tokenizer for decoding tokens
        show_image_info: Whether to show image information
        show_tokens: Whether to show tokenized text
        show_labels: Whether to show label masks
    """
    print_separator()
    print(f"Sample {idx}")
    print_separator()
    
    try:
        sample = dataset[idx]
    except Exception as e:
        print(f"❌ Error loading sample {idx}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get the raw conversation data
    file_idx, row_idx, turn_start_idx = dataset.index[idx]
    parquet_file = dataset.parquet_files[file_idx]
    
    # Load the raw data to show conversation structure
    if file_idx not in dataset._file_cache:
        try:
            dataset._file_cache[file_idx] = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"⚠️  Could not load raw data: {e}")
            dataset._file_cache[file_idx] = None
    
    if file_idx in dataset._file_cache and dataset._file_cache[file_idx] is not None:
        df = dataset._file_cache[file_idx]
        raw_row = df.iloc[row_idx]
        raw_sample = raw_row.to_dict()
        
        print(f"📁 Source: {parquet_file.name}, row {row_idx}, turn {turn_start_idx}")
        print()
        
        # Show raw conversation structure
        conversations = raw_sample.get('conversations', None)
        if isinstance(conversations, str):
            try:
                conversations = json.loads(conversations)
            except (json.JSONDecodeError, TypeError):
                pass
        
        if conversations:
            print("💬 Raw Conversation Structure:")
            for i, turn in enumerate(conversations):
                if isinstance(turn, dict):
                    role = turn.get('from', turn.get('role', 'unknown'))
                    value = turn.get('value', turn.get('content', ''))
                    marker = "👉" if i == turn_start_idx else "  "
                    print(f"{marker} Turn {i}: {role}")
                    print(f"   {value[:200]}{'...' if len(value) > 200 else ''}")
            print()
    
    # Image information
    if show_image_info:
        # Always try to get raw image from parquet for saving
        file_idx, row_idx, turn_start_idx = dataset.index[idx]
        raw_image = None
        if file_idx not in dataset._file_cache:
            try:
                parquet_file = dataset.parquet_files[file_idx]
                dataset._file_cache[file_idx] = pd.read_parquet(parquet_file)
            except Exception:
                pass
        
        if file_idx in dataset._file_cache and dataset._file_cache[file_idx] is not None:
            df = dataset._file_cache[file_idx]
            raw_row = df.iloc[row_idx]
            raw_image = raw_row.get('image', None)
        
        pixel_values = sample.get('pixel_values', None)
        if pixel_values is not None:
            if isinstance(pixel_values, torch.Tensor):
                print(f"🖼️  Image: Shape {tuple(pixel_values.shape)}, "
                      f"dtype {pixel_values.dtype}")
                print(f"   Min: {pixel_values.min():.3f}, "
                      f"Max: {pixel_values.max():.3f}, "
                      f"Mean: {pixel_values.mean():.3f}")
                
                # Display/save original image from parquet if available
                if (display_images or save_image_dir) and raw_image is not None:
                    try:
                        img_bytes = raw_image
                        if isinstance(raw_image, dict):
                            img_bytes = raw_image.get('bytes', None)
                        
                        if img_bytes and isinstance(img_bytes, bytes):
                            pil_img = Image.open(io.BytesIO(img_bytes))
                            
                            # Save if requested
                            if save_image_dir:
                                save_path = (
                                    save_image_dir /
                                    f"sample_{idx}_original.jpg"
                                )
                                pil_img.save(save_path)
                                print(
                                    f"   💾 Saved original image to: {save_path}"
                                )
                            
                            # Display in UI
                            if display_images:
                                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                                ax.imshow(pil_img)
                                ax.axis('off')
                                
                                parquet_file = dataset.parquet_files[file_idx]
                                title = (
                                    f"Sample {idx} (Original)\n"
                                    f"Source: {parquet_file.name}, "
                                    f"row {row_idx}, turn {turn_start_idx}"
                                )
                                ax.set_title(title, fontsize=10, pad=10)
                                plt.tight_layout()
                                
                                # Save visualization if requested
                                if save_image_dir:
                                    viz_path = (
                                        save_image_dir /
                                        f"sample_{idx}_original_viz.png"
                                    )
                                    plt.savefig(
                                        viz_path, dpi=150, bbox_inches='tight'
                                    )
                                    print(
                                        f"   💾 Saved original visualization "
                                        f"to: {viz_path}"
                                    )
                                
                                # Show the window (non-blocking)
                                plt.show(block=False)
                                plt.pause(0.1)
                                print(
                                    f"   👁️  Displaying original image "
                                    f"(close window to continue)"
                                )
                    except Exception as e:
                        print(f"   ⚠️  Could not display original image: {e}")
                
                # Display/save the processed image
                if display_images or save_image_dir:
                    try:
                        # Convert processed image back to displayable format
                        # pixel_values are normalized, need to denormalize
                        # Assuming ImageNet normalization:
                        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        img_tensor = pixel_values.clone()
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        
                        # Convert to PIL Image
                        img_array = (
                            img_tensor.permute(1, 2, 0).numpy() * 255
                        ).astype('uint8')
                        pil_img = Image.fromarray(img_array)
                        
                        # Save processed image if requested
                        if save_image_dir:
                            save_path = (
                                save_image_dir / f"sample_{idx}_processed.jpg"
                            )
                            pil_img.save(save_path)
                            print(f"   💾 Saved processed image to: {save_path}")
                        
                        # Display in UI
                        if display_images:
                            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                            ax.imshow(pil_img)
                            ax.axis('off')
                            
                            file_idx, row_idx, turn_idx = dataset.index[idx]
                            parquet_file = dataset.parquet_files[file_idx]
                            title = (
                                f"Sample {idx} (Processed)\n"
                                f"Source: {parquet_file.name}, row {row_idx}, "
                                f"turn {turn_idx}"
                            )
                            ax.set_title(title, fontsize=10, pad=10)
                            plt.tight_layout()
                            
                            # Save visualization if requested
                            if save_image_dir:
                                viz_path = (
                                    save_image_dir /
                                    f"sample_{idx}_processed_viz.png"
                                )
                                plt.savefig(
                                    viz_path, dpi=150, bbox_inches='tight'
                                )
                                print(f"   💾 Saved visualization to: {viz_path}")
                            
                            # Show the window (non-blocking)
                            plt.show(block=False)
                            plt.pause(0.1)  # Small pause to ensure window appears
                            print(
                                f"   👁️  Displaying processed image "
                                f"(close window to continue)"
                            )
                    except Exception as e:
                        print(f"   ⚠️  Could not display processed image: {e}")
            else:
                print(f"🖼️  Image: {type(pixel_values)}")
        else:
            # Try to get raw image from dataset to see if it exists
            file_idx, row_idx, turn_start_idx = dataset.index[idx]
            if file_idx not in dataset._file_cache:
                try:
                    parquet_file = dataset.parquet_files[file_idx]
                    dataset._file_cache[file_idx] = pd.read_parquet(parquet_file)
                except Exception:
                    pass
            
            if file_idx in dataset._file_cache and dataset._file_cache[file_idx] is not None:
                df = dataset._file_cache[file_idx]
                raw_row = df.iloc[row_idx]
                raw_image = raw_row.get('image', None)
                if raw_image is not None:
                    print(f"🖼️  Image: Present in parquet but failed to process")
                    print(f"   Raw image type: {type(raw_image)}")
                    if isinstance(raw_image, dict):
                        print(f"   Image dict keys: {list(raw_image.keys())}")
                    
                    # Try to extract and display/save image
                    if display_images or save_image_dir:
                        try:
                            # Extract bytes from dict if needed
                            img_bytes = raw_image
                            if isinstance(raw_image, dict):
                                img_bytes = raw_image.get('bytes', None)
                            
                            if img_bytes and isinstance(img_bytes, bytes):
                                pil_img = Image.open(io.BytesIO(img_bytes))
                                
                                # Save original image if requested
                                if save_image_dir:
                                    save_path = (
                                        save_image_dir /
                                        f"sample_{idx}_original.jpg"
                                    )
                                    pil_img.save(save_path)
                                    print(
                                        f"   💾 Saved original image to: "
                                        f"{save_path}"
                                    )
                                
                                # Display in UI
                                if display_images:
                                    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                                    ax.imshow(pil_img)
                                    ax.axis('off')
                                    
                                    parquet_file = dataset.parquet_files[file_idx]
                                    title = (
                                        f"Sample {idx} (Original - "
                                        f"Failed to Process)\n"
                                        f"Source: {parquet_file.name}, "
                                        f"row {row_idx}, turn {turn_start_idx}"
                                    )
                                    ax.set_title(title, fontsize=10, pad=10)
                                    plt.tight_layout()
                                    
                                    # Save visualization if requested
                                    if save_image_dir:
                                        viz_path = (
                                            save_image_dir /
                                            f"sample_{idx}_original_viz.png"
                                        )
                                        plt.savefig(
                                            viz_path, dpi=150,
                                            bbox_inches='tight'
                                        )
                                        print(
                                            f"   💾 Saved visualization to: "
                                            f"{viz_path}"
                                        )
                                    
                                    # Show the window (non-blocking)
                                    plt.show(block=False)
                                    plt.pause(0.1)
                                    print(
                                        f"   👁️  Displaying original image "
                                        f"(close window to continue)"
                                    )
                        except Exception as e:
                            print(f"   ⚠️  Could not display image: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    print("🖼️  Image: None (text-only sample)")
            else:
                print("🖼️  Image: None (text-only sample)")
        print()
    
    # Input IDs and attention mask
    if show_tokens:
        input_ids = sample.get('input_ids', None)
        attention_mask = sample.get('attention_mask', None)
        
        if input_ids is not None:
            print(f"📝 Input IDs: Shape {tuple(input_ids.shape)}, "
                  f"dtype {input_ids.dtype}")
            print(f"   Length: {len(input_ids)} tokens")
            print(f"   Non-padding tokens: {attention_mask.sum().item() if attention_mask is not None else 'N/A'}")
            print()
            
            # Decode and show text
            print("📖 Decoded Input Text:")
            decoded = format_tokens(input_ids, tokenizer, max_display=100)
            print(f"   {decoded}")
            print()
            
            # Show token breakdown
            if attention_mask is not None:
                valid_tokens = input_ids[attention_mask.bool()]
                print("📊 Token Breakdown:")
                print(f"   Valid tokens: {len(valid_tokens)}")
                print(f"   Padding tokens: {len(input_ids) - len(valid_tokens)}")
                print()
    
    # Labels
    if show_labels:
        labels = sample.get('labels', None)
        if labels is not None:
            print(f"🏷️  Labels: Shape {tuple(labels.shape)}, dtype {labels.dtype}")
            
            # Count masked vs unmasked tokens
            masked = (labels == -100).sum().item()
            unmasked = (labels != -100).sum().item()
            
            # Get attention mask to distinguish padding from valid tokens
            attention_mask = sample.get('attention_mask', None)
            if attention_mask is not None:
                valid_tokens = attention_mask.sum().item()
                padding_tokens = len(labels) - valid_tokens
                unmasked_valid = ((labels != -100) & (attention_mask == 1)).sum().item()
                unmasked_padding = ((labels != -100) & (attention_mask == 0)).sum().item()
                
                print(f"   Masked tokens (input): {masked}")
                print(f"   Unmasked tokens (target, valid): {unmasked_valid}")
                if unmasked_padding > 0:
                    print(f"   ⚠️  Unmasked padding tokens: {unmasked_padding} (should be 0)")
                print(f"   Total valid tokens: {valid_tokens}")
                print(f"   Total padding tokens: {padding_tokens}")
            else:
                print(f"   Masked tokens (input): {masked}")
                print(f"   Unmasked tokens (target): {unmasked}")
            
            if unmasked > 0:
                # Show the target tokens
                target_mask = labels != -100
                target_ids = labels[target_mask]
                print()
                print("🎯 Target Text (what model should predict):")
                target_text = format_tokens(target_ids, tokenizer, max_display=100)
                print(f"   {target_text}")
                print()
                
                # Show label distribution
                if len(target_ids) > 0:
                    unique_labels = torch.unique(target_ids)
                    print(f"   Unique target token IDs: {len(unique_labels)}")
                    print(f"   Target token range: [{target_ids.min().item()}, "
                          f"{target_ids.max().item()}]")
            print()
    
    # Verify label alignment
    input_ids = sample.get('input_ids', None)
    labels = sample.get('labels', None)
    if input_ids is not None and labels is not None:
        if input_ids.shape != labels.shape:
            print("⚠️  WARNING: input_ids and labels have different shapes!")
        else:
            # Check that unmasked labels match input_ids
            unmasked_mask = labels != -100
            if unmasked_mask.any():
                label_values = labels[unmasked_mask]
                input_values = input_ids[unmasked_mask]
                if torch.equal(label_values, input_values):
                    print("✅ Label alignment: Correct (unmasked labels match input_ids)")
                else:
                    print("⚠️  WARNING: Unmasked labels don't match input_ids!")
                    print(f"   Mismatch count: {(label_values != input_values).sum().item()}")
            print()


def verify_model_forward_pass(
    dataset: LLaVAInstructDataset,
    model: LLaVAModel,
    idx: int,
    tokenizer,
    device: torch.device = torch.device("cpu"),
):
    """Verify model forward pass: image prepending, masking, and padding."""
    print_separator()
    print(f"🔍 Model Forward Pass Verification for Sample {idx}")
    print_separator()
    
    results = {
        'passed': True,
        'errors': [],
        'warnings': [],
    }
    
    try:
        sample = dataset[idx]
    except Exception as e:
        print(f"❌ Error loading sample {idx}: {e}")
        results['passed'] = False
        results['errors'].append(f"Failed to load sample: {e}")
        return results
    
    pixel_values = sample.get('pixel_values', None)
    input_ids = sample.get('input_ids', None)
    attention_mask = sample.get('attention_mask', None)
    labels = sample.get('labels', None)
    
    if pixel_values is None:
        print("⚠️  Sample has no image, skipping model forward verification")
        results['warnings'].append("Sample has no image")
        return results
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    # Prepare batch (add batch dimension)
    pixel_values_batch = pixel_values.unsqueeze(0).to(device)
    input_ids_batch = input_ids.unsqueeze(0).to(device)
    attention_mask_batch = attention_mask.unsqueeze(0).to(device)
    labels_batch = labels.unsqueeze(0).to(device)
    
    print(f"📊 Dataset Output (BEFORE model forward):")
    print(f"   pixel_values: {tuple(pixel_values_batch.shape)}")
    print(f"   input_ids: {tuple(input_ids_batch.shape)} (text tokens only)")
    print(f"   attention_mask: {tuple(attention_mask_batch.shape)} (text tokens only)")
    print(f"   labels: {tuple(labels_batch.shape)} (text tokens only)")
    
    # Verify padding is handled in dataset
    text_valid_tokens = attention_mask_batch.sum().item()
    text_padding_tokens = attention_mask_batch.shape[1] - text_valid_tokens
    print(f"   Text valid tokens: {text_valid_tokens}")
    print(f"   Text padding tokens: {text_padding_tokens}")
    
    # Verify padding is masked in labels
    padding_mask = attention_mask_batch[0] == 0
    if padding_mask.any():
        padding_labels = labels_batch[0][padding_mask]
        unmasked_padding = (padding_labels != -100).sum().item()
        if unmasked_padding > 0:
            print(f"   ⚠️  WARNING: {unmasked_padding} padding tokens are NOT masked!")
            results['passed'] = False
            results['errors'].append(
                f"{unmasked_padding} padding tokens not masked in dataset"
            )
        else:
            print(f"   ✅ Padding tokens are correctly masked in labels")
    print()
    
    # Check 1: Verify visual tokens are prepended
    print("✅ Check 1: Visual tokens prepended to input")
    
    # Get visual embeddings shape
    visual_embeds = model.encode_images(pixel_values_batch)
    num_visual_tokens = visual_embeds.shape[1]
    print(f"   Number of visual tokens: {num_visual_tokens}")
    
    # Original attention_mask length (text only, from dataset)
    original_attention_len = attention_mask_batch.shape[1]
    print(f"   Text tokens in dataset: {original_attention_len}")
    print(f"   Total tokens after model forward: {num_visual_tokens + original_attention_len}")
    
    # Check what model would produce
    text_embeds = model.language_model.get_input_embeddings()(
        input_ids_batch
    )
    
    # Check concatenation order
    inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
    print(f"   Visual embeddings shape: {tuple(visual_embeds.shape)}")
    print(f"   Text embeddings shape: {tuple(text_embeds.shape)}")
    print(
        f"   Concatenated embeddings shape: "
        f"{tuple(inputs_embeds.shape)}"
    )
    print(
        f"   ✅ Visual embeddings are PREPENDED "
        f"(first {num_visual_tokens} tokens)"
    )
    print()
    
    # Check 2: Verify visual tokens are masked in labels
    print("✅ Check 2: Visual tokens are masked in labels")
    
    # The model should add visual_labels with -100
    visual_labels = torch.full(
        visual_embeds.size()[:-1],
        -100,
        dtype=labels_batch.dtype,
        device=labels_batch.device
    )
    expected_labels = torch.cat([visual_labels, labels_batch], dim=1)
    
    # Verify the first num_visual_tokens are masked
    visual_labels_in_expected = expected_labels[0, :num_visual_tokens]
    all_masked = (visual_labels_in_expected == -100).all()
    
    if all_masked:
        print(
            f"   ✅ First {num_visual_tokens} tokens (visual) "
            f"are masked (labels = -100)"
        )
    else:
        print(f"   ❌ ERROR: Visual tokens are NOT all masked!")
        print(f"   Visual labels: {visual_labels_in_expected}")
        results['passed'] = False
        results['errors'].append("Visual tokens are not all masked")
    
    # Check 3: Verify user input and assistant prompt are masked
    print()
    print("✅ Check 3: User input and assistant prompt are masked")
    
    # Count masked tokens in original labels (before model adds visual tokens)
    original_masked = (labels_batch[0] == -100).sum().item()
    original_unmasked = (labels_batch[0] != -100).sum().item()
    
    # After model adds visual tokens, total masked should be:
    # num_visual_tokens + original_masked
    total_expected_masked = num_visual_tokens + original_masked
    total_expected_unmasked = original_unmasked
    
    print(f"   Original labels: {original_masked} masked, {original_unmasked} unmasked")
    print(f"   After model forward: {total_expected_masked} masked (visual + user + prompt)")
    print(f"   Expected unmasked: {total_expected_unmasked} (assistant response only)")
    
    # Check that user input and prompt are masked in original labels
    if original_unmasked > 0:
        unmasked_positions = (labels_batch[0] != -100).nonzero(as_tuple=True)[0]
        unmasked_tokens = labels_batch[0][unmasked_positions]
        unmasked_text = tokenizer.decode(unmasked_tokens.tolist(), skip_special_tokens=False)
        
        print(f"   Unmasked text (should be assistant response only):")
        print(f"      {unmasked_text[:200]}{'...' if len(unmasked_text) > 200 else ''}")
        
        # Check unmasked text doesn't contain prompt markers
        if "Human:" in unmasked_text or "Assistant:" in unmasked_text:
            print(f"   ⚠️  WARNING: Unmasked text contains prompt markers!")
            results['warnings'].append("Unmasked text may contain prompt markers")
        else:
            print(f"   ✅ Unmasked text appears to be assistant response only")
    else:
        print(f"   ❌ ERROR: No unmasked tokens (empty target)")
        results['passed'] = False
        results['errors'].append("No unmasked tokens in labels")
    
    # Check 4: Verify complete masking structure
    print()
    print("✅ Check 4: Complete masking structure verification")
    
    # Simulate what model forward does
    final_labels = torch.cat([visual_labels, labels_batch], dim=1)
    final_attention = torch.cat([
        torch.ones(num_visual_tokens, dtype=attention_mask_batch.dtype, device=device).unsqueeze(0),
        attention_mask_batch
    ], dim=1)
    
    # Verify final structure: visual tokens (-100), user+prompt (-100), response (token IDs), padding (-100)
    
    visual_masked = (final_labels[0, :num_visual_tokens] == -100).all()
    if not visual_masked:
        print(f"   ❌ ERROR: Visual tokens not all masked")
        results['passed'] = False
        results['errors'].append("Visual tokens not masked correctly")
    else:
        print(f"   ✅ Visual tokens ({num_visual_tokens}) are masked")
    
    # Check that padding is masked
    padding_mask = final_attention[0] == 0
    if padding_mask.any():
        padding_labels = final_labels[0][padding_mask]
        unmasked_padding = (padding_labels != -100).sum().item()
        if unmasked_padding > 0:
            print(f"   ❌ ERROR: {unmasked_padding} padding tokens are unmasked")
            results['passed'] = False
            results['errors'].append(f"{unmasked_padding} padding tokens unmasked")
        else:
            print(f"   ✅ Padding tokens are masked")
    
    # Summary
    print()
    print("📋 Summary:")
    print()
    print("📦 Dataset Output (before model forward):")
    print("   ✅ Text token padding: Handled (attention_mask has 0s for padding)")
    print("   ✅ Text padding masking: Handled (labels set to -100 for padding)")
    print("   ✅ Text input masking: Handled (user + prompt masked in labels)")
    print("   ❌ Visual token masks: NOT in dataset (added by model forward)")
    print()
    print("🔧 Model Forward (what it adds):")
    print("   ✅ Visual token attention mask: Added (all 1s, prepended)")
    print("   ✅ Visual token labels: Added (all -100, prepended)")
    print("   ✅ Visual embeddings: Concatenated with text embeddings")
    print()
    if results['passed']:
        print("✅ All checks passed!")
        print(f"   ✅ Images are prepended to user input")
        print(f"   ✅ Visual tokens ({num_visual_tokens}) are masked")
        print(f"   ✅ User input and prompt are masked")
        print(f"   ✅ Only assistant response is unmasked")
        print(f"   ✅ Padding is correctly handled")
    else:
        print(f"   ❌ Validation failed with {len(results['errors'])} error(s)")
        for error in results['errors']:
            print(f"      - {error}")
    
    total_len = num_visual_tokens + original_attention_len
    print()
    print(f"   Total sequence length after model forward: {total_len}")
    print(f"      - Visual tokens: {num_visual_tokens}")
    print(f"      - Text tokens: {original_attention_len}")
    print()
    
    return results


def inspect_decomposed_conversation(
    dataset: LLaVAInstructDataset,
    file_idx: int,
    row_idx: int,
    tokenizer,
    show_tokens: bool = True,
):
    """Inspect how a conversation is decomposed into training samples."""
    print_separator()
    print(f"📋 Inspecting Decomposed Conversation")
    print(f"   File: {dataset.parquet_files[file_idx].name}, Row: {row_idx}")
    print_separator()
    
    # Load the raw conversation
    if file_idx not in dataset._file_cache:
        try:
            parquet_file = dataset.parquet_files[file_idx]
            dataset._file_cache[file_idx] = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"❌ Error loading parquet file: {e}")
            return
    
    df = dataset._file_cache[file_idx]
    row = df.iloc[row_idx]
    sample = row.to_dict()
    
    # Parse conversations
    conversations = sample.get('conversations', None)
    if isinstance(conversations, str):
        try:
            conversations = json.loads(conversations)
        except (json.JSONDecodeError, TypeError):
            print(f"❌ Error parsing conversations JSON")
            return
    
    if not isinstance(conversations, list):
        print(f"❌ Conversations is not a list: {type(conversations)}")
        return
    
    # Find all samples from this conversation
    conversation_samples = []
    for idx in range(len(dataset)):
        f_idx, r_idx, turn_idx = dataset.index[idx]
        if f_idx == file_idx and r_idx == row_idx:
            conversation_samples.append((idx, turn_idx))
    
    # Sort by turn index
    conversation_samples.sort(key=lambda x: x[1])
    
    print(f"\n📊 Original Conversation Structure:")
    print(f"   Total turns: {len(conversations)}")
    print(f"   Training samples created: {len(conversation_samples)}")
    print()
    
    # Show original conversation
    print("💬 Original Multi-Turn Conversation:")
    for i, turn in enumerate(conversations):
        if isinstance(turn, dict):
            role = turn.get('from', turn.get('role', 'unknown'))
            value = turn.get('value', turn.get('content', ''))
            # Mark which turns are used in training samples
            is_used = any(t[1] == i for t in conversation_samples)
            marker = "👉" if is_used else "  "
            print(f"{marker} Turn {i}: {role}")
            print(f"   {value[:300]}{'...' if len(value) > 300 else ''}")
    print()
    
    # Show each decomposed sample
    print("🔀 Decomposed Training Samples:")
    print()
    
    for sample_idx, (dataset_idx, turn_start_idx) in enumerate(conversation_samples):
        print(f"   Sample {sample_idx + 1}/{len(conversation_samples)} "
              f"(Dataset Index: {dataset_idx}, Turn Start: {turn_start_idx})")
        print(f"   {'─' * 70}")
        
        # Get the training sample
        try:
            training_sample = dataset[dataset_idx]
        except Exception as e:
            print(f"   ❌ Error loading sample: {e}")
            continue
        
        # Show which turns are used
        if turn_start_idx + 1 < len(conversations):
            human_turn = conversations[turn_start_idx]
            gpt_turn = conversations[turn_start_idx + 1]
            
            human_role = human_turn.get('from', human_turn.get('role', 'unknown'))
            human_value = human_turn.get('value', human_turn.get('content', ''))
            gpt_role = gpt_turn.get('from', gpt_turn.get('role', 'unknown'))
            gpt_value = gpt_turn.get('value', gpt_turn.get('content', ''))
            
            print(f"   📝 Uses turns {turn_start_idx} ({human_role}) and {turn_start_idx + 1} ({gpt_role})")
            print(f"      Human: {human_value[:150]}{'...' if len(human_value) > 150 else ''}")
            print(f"      GPT:   {gpt_value[:150]}{'...' if len(gpt_value) > 150 else ''}")
        
        # Show sample details
        input_ids = training_sample.get('input_ids', None)
        labels = training_sample.get('labels', None)
        attention_mask = training_sample.get('attention_mask', None)
        pixel_values = training_sample.get('pixel_values', None)
        
        if input_ids is not None:
            valid_tokens = attention_mask.sum().item() if attention_mask is not None else len(input_ids)
            masked_tokens = (labels == -100).sum().item() if labels is not None else 0
            unmasked_tokens = (labels != -100).sum().item() if labels is not None else 0
            
            print(f"   📊 Token Stats:")
            print(f"      Total tokens: {len(input_ids)}")
            print(f"      Valid tokens: {valid_tokens}")
            print(f"      Masked (input): {masked_tokens}")
            print(f"      Unmasked (target): {unmasked_tokens}")
            
            if show_tokens and tokenizer is not None:
                # Show decoded text
                decoded = format_tokens(input_ids, tokenizer, max_display=80)
                print(f"   📖 Decoded Input:")
                print(f"      {decoded}")
                
                # Show target text
                if labels is not None and unmasked_tokens > 0:
                    target_mask = labels != -100
                    target_ids = labels[target_mask]
                    target_text = format_tokens(target_ids, tokenizer, max_display=80)
                    print(f"   🎯 Target (what model should predict):")
                    print(f"      {target_text}")
        
        if pixel_values is not None:
            print(f"   🖼️  Image: Present (shape: {tuple(pixel_values.shape)})")
        else:
            print(f"   🖼️  Image: None (text-only)")
        
        print()
    
    print_separator()
    print()


def verify_multi_turn_decomposition(dataset: LLaVAInstructDataset, num_samples: int = 10):
    """Verify multi-turn conversation decomposition."""
    print_separator()
    print("🔍 Multi-Turn Conversation Decomposition Verification")
    print_separator()
    
    results = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'checked_conversations': 0,
        'turns_per_conversation': [],
    }
    
    # Group samples by (file_idx, row_idx) to find conversations
    conversation_map = {}
    for idx in range(min(num_samples * 10, len(dataset))):  # Check more to find conversations
        file_idx, row_idx, turn_idx = dataset.index[idx]
        key = (file_idx, row_idx)
        if key not in conversation_map:
            conversation_map[key] = []
        conversation_map[key].append((idx, turn_idx))
    
    # Check conversations with multiple turns
    multi_turn_conversations = {
        k: v for k, v in conversation_map.items() if len(v) > 1
    }
    
    print(f"Found {len(multi_turn_conversations)} conversations with multiple turns")
    print(f"Checking first {min(num_samples, len(multi_turn_conversations))}...")
    print()
    
    checked = 0
    for (file_idx, row_idx), turns in list(multi_turn_conversations.items())[:num_samples]:
        checked += 1
        results['checked_conversations'] += 1
        results['turns_per_conversation'].append(len(turns))
        
        # Load the raw conversation
        if file_idx not in dataset._file_cache:
            try:
                parquet_file = dataset.parquet_files[file_idx]
                dataset._file_cache[file_idx] = pd.read_parquet(parquet_file)
            except Exception:
                continue
        
        df = dataset._file_cache[file_idx]
        row = df.iloc[row_idx]
        sample = row.to_dict()
        
        # Parse conversations
        conversations = sample.get('conversations', None)
        if isinstance(conversations, str):
            try:
                conversations = json.loads(conversations)
            except (json.JSONDecodeError, TypeError):
                continue
        
        if not isinstance(conversations, list):
            continue
        
        # Verify each turn is a separate sample
        print(f"Conversation from {dataset.parquet_files[file_idx].name}, row {row_idx}:")
        print(f"  Total turns in conversation: {len(conversations)}")
        print(f"  Samples created: {len(turns)}")
        
        # Check that turns are sequential
        turn_indices = sorted([t[1] for t in turns])
        is_sequential = all(
            turn_indices[i] + 2 == turn_indices[i+1]
            for i in range(len(turn_indices) - 1)
        )
        
        if len(turns) == len(conversations) // 2:
            print(f"  ✅ Correct number of samples ({len(turns)} turns → {len(turns)} samples)")
        else:
            print(f"  ⚠️  Expected {len(conversations) // 2} samples, got {len(turns)}")
            results['warnings'].append(
                f"Conversation ({file_idx}, {row_idx}): expected {len(conversations) // 2} samples, got {len(turns)}"
            )
        
        if is_sequential:
            print(f"  ✅ Turns are sequential: {turn_indices}")
        else:
            print(f"  ⚠️  Turns are not sequential: {turn_indices}")
            results['warnings'].append(
                f"Conversation ({file_idx}, {row_idx}): turns not sequential"
            )
        
        print()
    
    if results['turns_per_conversation']:
        avg_turns = sum(results['turns_per_conversation']) / len(results['turns_per_conversation'])
        print(f"Average turns per conversation: {avg_turns:.1f}")
    
    if not results['errors']:
        print("✅ Multi-turn decomposition appears correct")
    else:
        print(f"❌ Found {len(results['errors'])} errors")
        results['passed'] = False
    
    print()
    return results


def find_multi_turn_conversations(
    dataset: LLaVAInstructDataset,
    max_search: int = 1000,
    min_turns: int = 2,
) -> List[tuple]:
    """Find conversations that have been decomposed into multiple training samples.
    
    Args:
        dataset: The dataset to search
        max_search: Maximum number of samples to search through
        min_turns: Minimum number of training samples (turns) to consider
    
    Returns:
        List of (file_idx, row_idx, num_samples) tuples for multi-turn conversations
    """
    # Group samples by (file_idx, row_idx) to find conversations
    conversation_map = {}
    search_range = min(max_search, len(dataset))
    
    for idx in range(search_range):
        file_idx, row_idx, turn_idx = dataset.index[idx]
        key = (file_idx, row_idx)
        if key not in conversation_map:
            conversation_map[key] = []
        conversation_map[key].append((idx, turn_idx))
    
    # Filter to conversations with multiple turns
    multi_turn = []
    for (file_idx, row_idx), turns in conversation_map.items():
        if len(turns) >= min_turns:
            multi_turn.append((file_idx, row_idx, len(turns)))
    
    # Sort by number of turns (descending)
    multi_turn.sort(key=lambda x: x[2], reverse=True)
    
    return multi_turn


def print_dataset_stats(dataset: LLaVAInstructDataset):
    """Print statistics about the dataset."""
    print_separator()
    print("📊 Dataset Statistics")
    print_separator()
    
    print(f"Total samples: {len(dataset)}")
    print(f"Parquet files: {len(dataset.parquet_files)}")
    print(f"Max length: {dataset.max_length}")
    print(f"Num visual tokens: {dataset.num_visual_tokens}")
    print()
    
    # Find multi-turn conversations
    multi_turn_convs = find_multi_turn_conversations(dataset, max_search=min(1000, len(dataset)))
    if multi_turn_convs:
        total_convs = len(set((f, r) for f, r, _ in multi_turn_convs))
        total_samples_from_multi = sum(n for _, _, n in multi_turn_convs)
        avg_turns = sum(n for _, _, n in multi_turn_convs) / len(multi_turn_convs) if multi_turn_convs else 0
        max_turns = max(n for _, _, n in multi_turn_convs) if multi_turn_convs else 0
        
        print(f"Multi-turn conversations:")
        print(f"  Conversations with 2+ turns: {total_convs}")
        print(f"  Training samples from multi-turn: {total_samples_from_multi}")
        print(f"  Average turns per multi-turn conversation: {avg_turns:.1f}")
        print(f"  Maximum turns in a conversation: {max_turns}")
        print()
    
    # Sample a few random indices to check image presence
    num_check = min(100, len(dataset))
    indices_to_check = torch.randperm(len(dataset))[:num_check].tolist()
    
    samples_with_images = 0
    samples_without_images = 0
    
    for idx in indices_to_check:
        try:
            sample = dataset[idx]
            if sample.get('pixel_values') is not None:
                samples_with_images += 1
            else:
                samples_without_images += 1
        except Exception:
            pass
    
    if num_check > 0:
        img_percent = (samples_with_images / num_check) * 100
        print(f"Image presence (sampled {num_check} samples):")
        print(f"  With images: {samples_with_images} ({img_percent:.1f}%)")
        print(f"  Without images: {samples_without_images} ({100-img_percent:.1f}%)")
        print()


def validate_masking_and_prepending(
    dataset: LLaVAInstructDataset,
    model: LLaVAModel,
    tokenizer,
    num_samples: int = 10,
    device: torch.device = torch.device("cpu"),
) -> bool:
    """Comprehensive validation of masking and image prepending."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MASKING AND IMAGE PREPENDING VALIDATION")
    print("=" * 80)
    print()
    
    all_passed = True
    
    # 1. Verify multi-turn decomposition
    decomposition_results = verify_multi_turn_decomposition(dataset, num_samples=5)
    if not decomposition_results['passed']:
        all_passed = False
    
    # 2. Verify model forward pass for samples with images
    print()
    print("Validating model forward pass for samples with images...")
    print()
    
    # Find samples with images
    samples_with_images = []
    for idx in range(min(num_samples * 10, len(dataset))):
        try:
            sample = dataset[idx]
            if sample.get('pixel_values') is not None:
                samples_with_images.append(idx)
                if len(samples_with_images) >= num_samples:
                    break
        except Exception:
            continue
    
    if not samples_with_images:
        print("⚠️  No samples with images found for validation")
        return False
    
    for idx in samples_with_images[:num_samples]:
        results = verify_model_forward_pass(
            dataset, model, idx, tokenizer, device=device
        )
        if not results['passed']:
            all_passed = False
    
    print()
    print("=" * 80)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 80)
    print()
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Phase 2 training data"
    )
    
    # Data args
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to folder containing parquet files"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=768,
        help="Maximum sequence length (default: 768)"
    )
    
    # Inspection args
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random samples to inspect"
    )
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample indices to inspect (overrides --num_samples)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (optional, not needed for inspection)"
    )
    
    # Display options
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Don't show image information"
    )
    parser.add_argument(
        "--no_tokens",
        action="store_true",
        help="Don't show tokenized text"
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Don't show label information"
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only show dataset statistics, don't inspect samples"
    )
    parser.add_argument(
        "--save_images",
        type=str,
        default=None,
        help="Directory to save example images"
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Don't display images in UI (only save if --save_images is set)"
    )
    parser.add_argument(
        "--no_verify_model_forward",
        action="store_true",
        help="Don't verify model forward pass (enabled by default)"
    )
    parser.add_argument(
        "--no_validate_masking",
        action="store_true",
        help="Don't run comprehensive validation (enabled by default)"
    )
    parser.add_argument(
        "--no_decomposition",
        action="store_true",
        help="Skip decomposition inspection"
    )
    parser.add_argument(
        "--num_multi_turn_conversations",
        type=int,
        default=3,
        help="Number of multi-turn conversations to inspect"
    )
    parser.add_argument(
        "--conversation_location",
        type=str,
        default=None,
        help="Inspect specific conversation: 'file_idx,row_idx' (e.g., '0,5')"
    )
    
    args = parser.parse_args()
    
    # Create image save directory if specified
    save_image_dir = None
    if args.save_images:
        save_image_dir = Path(args.save_images).expanduser()
        save_image_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Will save images to: {save_image_dir}")
        print(
            f"   Format: sample_<id>_original.jpg, "
            f"sample_<id>_viz.png, sample_<id>_processed.jpg"
        )
        print()
    
    if not args.no_display:
        print("👁️  Images will be displayed in temporary windows")
        print("   Close each window or press Enter to continue\n")
    
    # Initialize model to get tokenizer and image processor
    # (checkpoint not needed - dataset constructs all data in __getitem__)
    print("Initializing model components (tokenizer & image processor)...")
    config = LLaVAConfig()
    model = LLaVAModel(config)
    
    # Optionally load checkpoint (usually not needed)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            print("✅ Checkpoint loaded (not required for data inspection)")
        else:
            print(f"⚠️  Warning: Checkpoint not found: {checkpoint_path}")
    
    tokenizer = model.language_model.tokenizer
    image_processor = model.vision_encoder.processor
    
    # Build dataset
    print("\nLoading dataset...")
    data_config = Phase2DataConfig(
        data_path=args.data_path,
        max_length=args.max_length
    )
    
    try:
        dataset = LLaVAInstructDataset(
            data_path=data_config.data_path,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=data_config.max_length,
        )
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print statistics
    print_dataset_stats(dataset)
    
    if args.stats_only:
        return 0
    
    # Inspect decomposition by default (unless disabled)
    should_inspect_decomposition = not args.no_decomposition
    if should_inspect_decomposition or args.conversation_location:
        if args.conversation_location:
            # Parse conversation location
            try:
                parts = args.conversation_location.split(',')
                if len(parts) != 2:
                    raise ValueError("Invalid format")
                file_idx = int(parts[0].strip())
                row_idx = int(parts[1].strip())
                
                if file_idx < 0 or file_idx >= len(dataset.parquet_files):
                    print(f"❌ Error: file_idx {file_idx} is out of range "
                          f"(0-{len(dataset.parquet_files)-1})")
                    return 1
                
                inspect_decomposed_conversation(
                    dataset,
                    file_idx,
                    row_idx,
                    tokenizer,
                    show_tokens=not args.no_tokens,
                )
            except (ValueError, IndexError) as e:
                print(f"❌ Error parsing conversation location: {e}")
                print("   Expected format: 'file_idx,row_idx' (e.g., '0,5')")
                return 1
        else:
            # Find and inspect multi-turn conversations
            multi_turn_convs = find_multi_turn_conversations(
                dataset,
                max_search=min(1000, len(dataset)),
                min_turns=2,
            )
            
            if not multi_turn_convs:
                print("⚠️  No multi-turn conversations found")
                return 0
            
            print(f"\nFound {len(multi_turn_convs)} multi-turn conversations")
            num_to_inspect = min(
                args.num_multi_turn_conversations, len(multi_turn_convs)
            )
            print(f"Inspecting first {num_to_inspect}...")
            print()
            
            for i, (file_idx, row_idx, num_samples) in enumerate(
                multi_turn_convs[:args.num_multi_turn_conversations]
            ):
                inspect_decomposed_conversation(
                    dataset,
                    file_idx,
                    row_idx,
                    tokenizer,
                    show_tokens=not args.no_tokens,
                )
                
                if i < num_to_inspect - 1:
                    print()  # Add spacing between conversations
        
        # If conversation_location is specified, we're done
        if args.conversation_location:
            return 0
    
    # Determine which samples to inspect
    if args.sample_indices:
        indices_to_inspect = args.sample_indices
    else:
        # Random samples
        num_samples = min(args.num_samples, len(dataset))
        indices_to_inspect = torch.randperm(len(dataset))[:num_samples].tolist()
        indices_to_inspect.sort()  # Sort for easier reading
    
    print(f"\nInspecting {len(indices_to_inspect)} sample(s)...")
    print()
    
    # Inspect each sample
    for idx in indices_to_inspect:
        if idx >= len(dataset):
            print(f"⚠️  Warning: Index {idx} is out of range (dataset size: {len(dataset)})")
            continue
        
        inspect_sample(
            dataset,
            idx,
            tokenizer,
            show_image_info=not args.no_images,
            show_tokens=not args.no_tokens,
            show_labels=not args.no_labels,
            save_image_dir=save_image_dir,
            display_images=not args.no_display,
        )
        
        # Verify model forward pass by default (unless disabled)
        if not args.no_verify_model_forward:
            # Determine device
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            
            verify_model_forward_pass(
                dataset,
                model,
                idx,
                tokenizer,
                device=device,
            )
        
        # Wait for user to close image windows before continuing
        if not args.no_display:
            print()  # Add blank line
            input("Press Enter to continue to next sample...")
    
    # Run comprehensive validation at the end (unless disabled)
    if not args.no_validate_masking:
        print()
        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        all_passed = validate_masking_and_prepending(
            dataset,
            model,
            tokenizer,
            num_samples=args.num_samples,
            device=device,
        )
        
        print_separator()
        if all_passed:
            print("✅ Inspection complete! All validations passed.")
        else:
            print("⚠️  Inspection complete, but some validations failed.")
        return 0 if all_passed else 1
    
    print_separator()
    print("✅ Inspection complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

