#!/usr/bin/env python3
"""Diagnose Phase 2 data loading issues that could cause volatile loss.

This script checks for common data loading problems:
- Label masking correctness
- Empty or very short target sequences
- Sequence length consistency
- Image processing issues
- Batch construction issues
- Data distribution problems

Usage:
    python scripts/diagnose_phase2_data.py --data_path ~/dataset/llava-instruct-mix/data
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlm.configs.data_config import Phase2DataConfig
from vlm.configs.model_config import LLaVAConfig
from vlm.data.llava_instruct_dataset import LLaVAInstructDataset, collate_fn
from vlm.models.llava import LLaVAModel
from torch.utils.data import DataLoader


class DataDiagnostics:
    """Diagnostic checker for Phase 2 data loading."""
    
    def __init__(
        self,
        dataset: LLaVAInstructDataset,
        tokenizer,
        batch_size: int = 4,
        num_samples: int = 1000,
    ):
        """Initialize diagnostics.
        
        Args:
            dataset: The dataset to diagnose
            tokenizer: Tokenizer for decoding
            batch_size: Batch size for batch checks
            num_samples: Number of samples to check
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_samples = min(num_samples, len(dataset))
        
        self.issues = []
        self.stats = {
            'total_checked': 0,
            'samples_with_images': 0,
            'samples_without_images': 0,
            'empty_targets': 0,
            'very_short_targets': 0,
            'label_mismatches': 0,
            'sequence_lengths': [],
            'target_lengths': [],
            'input_lengths': [],
            'file_distribution': Counter(),
            'batch_issues': [],
        }
    
    def check_sample(self, idx: int) -> Dict:
        """Check a single sample for issues.
        
        Returns:
            Dictionary with issue flags and stats
        """
        result = {
            'idx': idx,
            'has_issues': False,
            'issues': [],
            'stats': {},
        }
        
        try:
            sample = self.dataset[idx]
        except Exception as e:
            result['has_issues'] = True
            result['issues'].append(f"Failed to load sample: {e}")
            return result
        
        # Get file info
        file_idx, row_idx, turn_idx = self.dataset.index[idx]
        parquet_file = self.dataset.parquet_files[file_idx]
        result['stats']['file'] = parquet_file.name
        self.stats['file_distribution'][parquet_file.name] += 1
        
        # Check image
        pixel_values = sample.get('pixel_values', None)
        if pixel_values is not None:
            self.stats['samples_with_images'] += 1
            result['stats']['has_image'] = True
            if not isinstance(pixel_values, torch.Tensor):
                result['has_issues'] = True
                result['issues'].append("pixel_values is not a tensor")
        else:
            self.stats['samples_without_images'] += 1
            result['stats']['has_image'] = False
        
        # Check input_ids
        input_ids = sample.get('input_ids', None)
        attention_mask = sample.get('attention_mask', None)
        labels = sample.get('labels', None)
        
        if input_ids is None:
            result['has_issues'] = True
            result['issues'].append("Missing input_ids")
            return result
        
        if labels is None:
            result['has_issues'] = True
            result['issues'].append("Missing labels")
            return result
        
        if attention_mask is None:
            result['has_issues'] = True
            result['issues'].append("Missing attention_mask")
            return result
        
        # Check shapes
        if input_ids.shape != labels.shape:
            result['has_issues'] = True
            result['issues'].append(
                f"Shape mismatch: input_ids {input_ids.shape} vs "
                f"labels {labels.shape}"
            )
        
        if input_ids.shape != attention_mask.shape:
            result['has_issues'] = True
            result['issues'].append(
                f"Shape mismatch: input_ids {input_ids.shape} vs "
                f"attention_mask {attention_mask.shape}"
            )
        
        # Check sequence length
        seq_len = len(input_ids)
        self.stats['sequence_lengths'].append(seq_len)
        result['stats']['sequence_length'] = seq_len
        
        # Check label masking
        masked = (labels == -100).sum().item()
        unmasked = (labels != -100).sum().item()
        
        # Check that unmasked labels match input_ids
        unmasked_mask = labels != -100
        if unmasked_mask.any():
            label_values = labels[unmasked_mask]
            input_values = input_ids[unmasked_mask]
            if not torch.equal(label_values, input_values):
                result['has_issues'] = True
                result['issues'].append(
                    f"Label mismatch: {((label_values != input_values).sum().item())} "
                    f"tokens don't match"
                )
                self.stats['label_mismatches'] += 1
        
        # Check target length (unmasked tokens)
        valid_mask = attention_mask == 1
        target_mask = (labels != -100) & valid_mask
        target_length = target_mask.sum().item()
        
        self.stats['target_lengths'].append(target_length)
        self.stats['input_lengths'].append(seq_len - target_length)
        result['stats']['target_length'] = target_length
        result['stats']['input_length'] = seq_len - target_length
        
        # Check for empty targets
        if target_length == 0:
            result['has_issues'] = True
            result['issues'].append("Empty target sequence (all masked)")
            self.stats['empty_targets'] += 1
        
        # Check for very short targets (less than 5 tokens)
        if target_length > 0 and target_length < 5:
            result['has_issues'] = True
            result['issues'].append(
                f"Very short target sequence ({target_length} tokens)"
            )
            self.stats['very_short_targets'] += 1
        
        # Check padding masking
        padding_mask = attention_mask == 0
        if padding_mask.any():
            padding_labels = labels[padding_mask]
            unmasked_padding = (padding_labels != -100).sum().item()
            if unmasked_padding > 0:
                result['has_issues'] = True
                result['issues'].append(
                    f"Unmasked padding tokens: {unmasked_padding}"
                )
        
        # Decode to check for tokenization issues
        try:
            if target_length > 0:
                target_ids = labels[target_mask]
                decoded = self.tokenizer.decode(
                    target_ids.tolist(),
                    skip_special_tokens=False
                )
                result['stats']['target_text'] = decoded[:100]  # First 100 chars
        except Exception as e:
            result['has_issues'] = True
            result['issues'].append(f"Failed to decode target: {e}")
        
        return result
    
    def check_batch(self, batch: Dict) -> List[str]:
        """Check a batch for issues.
        
        Returns:
            List of issue messages
        """
        issues = []
        
        # Check batch consistency
        batch_size = len(batch['input_ids'])
        
        # Check all sequences have same length
        input_lengths = [len(ids) for ids in batch['input_ids']]
        if len(set(input_lengths)) > 1:
            issues.append(
                f"Inconsistent sequence lengths in batch: {input_lengths}"
            )
        
        # Check pixel_values
        pixel_values = batch.get('pixel_values', None)
        if pixel_values is not None:
            if pixel_values.shape[0] != batch_size:
                issues.append(
                    f"pixel_values batch size mismatch: "
                    f"{pixel_values.shape[0]} vs {batch_size}"
                )
        
        # Check for NaN or Inf in inputs
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in batch:
                tensor = batch[key]
                if torch.isnan(tensor).any():
                    issues.append(f"NaN values in {key}")
                if torch.isinf(tensor).any():
                    issues.append(f"Inf values in {key}")
        
        return issues
    
    def run_diagnostics(self) -> Dict:
        """Run all diagnostic checks.
        
        Returns:
            Dictionary with all diagnostic results
        """
        print(f"Running diagnostics on {self.num_samples} samples...")
        print()
        
        # Sample indices to check
        if self.num_samples >= len(self.dataset):
            indices = list(range(len(self.dataset)))
        else:
            indices = torch.randperm(len(self.dataset))[:self.num_samples].tolist()
        
        # Check individual samples
        print("Checking individual samples...")
        sample_results = []
        for idx in tqdm(indices, desc="Samples"):
            result = self.check_sample(idx)
            sample_results.append(result)
            self.stats['total_checked'] += 1
            
            if result['has_issues']:
                self.issues.append({
                    'type': 'sample',
                    'idx': idx,
                    'issues': result['issues'],
                })
        
        # Check batches
        print("\nChecking batch construction...")
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
        )
        
        num_batches_to_check = min(50, len(dataloader))
        batch_target_lengths = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches_to_check:
                break
            
            batch_issues = self.check_batch(batch)
            if batch_issues:
                self.issues.append({
                    'type': 'batch',
                    'batch_idx': i,
                    'issues': batch_issues,
                })
                self.stats['batch_issues'].append({
                    'batch_idx': i,
                    'issues': batch_issues,
                })
            
            # Analyze target length variance in batch
            labels = batch.get('labels', None)
            attention_mask = batch.get('attention_mask', None)
            if labels is not None and attention_mask is not None:
                batch_target_lens = []
                for j in range(len(labels)):
                    sample_labels = labels[j]
                    sample_mask = attention_mask[j]
                    target_mask = (sample_labels != -100) & (sample_mask == 1)
                    target_len = target_mask.sum().item()
                    batch_target_lens.append(target_len)
                
                if batch_target_lens:
                    batch_target_lengths.append({
                        'batch_idx': i,
                        'target_lengths': batch_target_lens,
                        'mean': np.mean(batch_target_lens),
                        'std': np.std(batch_target_lens),
                        'min': np.min(batch_target_lens),
                        'max': np.max(batch_target_lens),
                    })
        
        self.stats['batch_target_lengths'] = batch_target_lengths
        
        # Compute statistics
        self._compute_statistics()
        
        return {
            'issues': self.issues,
            'stats': self.stats,
            'sample_results': sample_results,
        }
    
    def _compute_statistics(self):
        """Compute summary statistics."""
        if self.stats['sequence_lengths']:
            self.stats['seq_len_mean'] = np.mean(self.stats['sequence_lengths'])
            self.stats['seq_len_std'] = np.std(self.stats['sequence_lengths'])
            self.stats['seq_len_min'] = np.min(self.stats['sequence_lengths'])
            self.stats['seq_len_max'] = np.max(self.stats['sequence_lengths'])
        
        if self.stats['target_lengths']:
            self.stats['target_len_mean'] = np.mean(self.stats['target_lengths'])
            self.stats['target_len_std'] = np.std(self.stats['target_lengths'])
            self.stats['target_len_min'] = np.min(self.stats['target_lengths'])
            self.stats['target_len_max'] = np.max(self.stats['target_lengths'])
        
        if self.stats['input_lengths']:
            self.stats['input_len_mean'] = np.mean(self.stats['input_lengths'])
            self.stats['input_len_std'] = np.std(self.stats['input_lengths'])
    
    def print_report(self):
        """Print diagnostic report."""
        print("\n" + "=" * 80)
        print("PHASE 2 DATA LOADING DIAGNOSTICS REPORT")
        print("=" * 80)
        print()
        
        # Summary
        print("📊 SUMMARY")
        print("-" * 80)
        print(f"Total samples checked: {self.stats['total_checked']}")
        print(f"Total issues found: {len(self.issues)}")
        print()
        
        # Dataset statistics
        print("📈 DATASET STATISTICS")
        print("-" * 80)
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Parquet files: {len(self.dataset.parquet_files)}")
        print(f"Max length: {self.dataset.max_length}")
        print()
        
        # Image statistics
        print("🖼️  IMAGE STATISTICS")
        print("-" * 80)
        total = self.stats['samples_with_images'] + self.stats['samples_without_images']
        if total > 0:
            img_pct = (self.stats['samples_with_images'] / total) * 100
            print(f"Samples with images: {self.stats['samples_with_images']} ({img_pct:.1f}%)")
            print(f"Samples without images: {self.stats['samples_without_images']} ({100-img_pct:.1f}%)")
        print()
        
        # Sequence length statistics
        if self.stats['sequence_lengths']:
            print("📏 SEQUENCE LENGTH STATISTICS")
            print("-" * 80)
            print(f"Mean: {self.stats.get('seq_len_mean', 0):.1f}")
            print(f"Std: {self.stats.get('seq_len_std', 0):.1f}")
            print(f"Min: {self.stats.get('seq_len_min', 0)}")
            print(f"Max: {self.stats.get('seq_len_max', 0)}")
            print()
        
        # Target length statistics
        if self.stats['target_lengths']:
            print("🎯 TARGET LENGTH STATISTICS")
            print("-" * 80)
            print(f"Mean: {self.stats.get('target_len_mean', 0):.1f}")
            print(f"Std: {self.stats.get('target_len_std', 0):.1f}")
            print(f"Min: {self.stats.get('target_len_min', 0)}")
            print(f"Max: {self.stats.get('target_len_max', 0)}")
            
            # Calculate percentiles
            target_lengths = np.array(self.stats['target_lengths'])
            p25 = np.percentile(target_lengths, 25)
            p50 = np.percentile(target_lengths, 50)
            p75 = np.percentile(target_lengths, 75)
            p95 = np.percentile(target_lengths, 95)
            print(f"Percentiles: P25={p25:.1f}, P50={p50:.1f}, P75={p75:.1f}, P95={p95:.1f}")
            print()
            
            # Check for problematic target lengths
            if self.stats['empty_targets'] > 0:
                print(f"⚠️  WARNING: {self.stats['empty_targets']} samples have empty targets!")
            
            target_std = self.stats.get('target_len_std', 0)
            target_mean = self.stats.get('target_len_mean', 0)
            cv = (target_std / target_mean * 100) if target_mean > 0 else 0
            if cv > 100:  # Coefficient of variation > 100% indicates high variance
                print(f"⚠️  WARNING: High target length variance (CV={cv:.1f}%)!")
                print(f"   This can cause volatile loss when batches have mixed short/long targets.")
            
            if self.stats['very_short_targets'] > 0:
                pct = (self.stats['very_short_targets'] / self.stats['total_checked']) * 100
                print(f"ℹ️  INFO: {self.stats['very_short_targets']} samples ({pct:.1f}%) have very short targets (<5 tokens)")
                print(f"   Short targets are valid, but high variance with long targets can cause issues.")
            print()
        
        # Batch-level target length variance
        if self.stats.get('batch_target_lengths'):
            batch_stds = [b['std'] for b in self.stats['batch_target_lengths']]
            batch_means = [b['mean'] for b in self.stats['batch_target_lengths']]
            if batch_stds:
                print("📦 BATCH-LEVEL TARGET LENGTH VARIANCE")
                print("-" * 80)
                print(f"Mean batch target length: {np.mean(batch_means):.1f}")
                print(f"Mean batch std: {np.mean(batch_stds):.1f}")
                print(f"Max batch std: {np.max(batch_stds):.1f}")
                print(f"Batches with high variance (std > 50): {sum(1 for s in batch_stds if s > 50)}")
                print()
        
        # File distribution
        if self.stats['file_distribution']:
            print("📁 FILE DISTRIBUTION")
            print("-" * 80)
            for file, count in self.stats['file_distribution'].most_common(10):
                pct = (count / self.stats['total_checked']) * 100
                print(f"  {file}: {count} samples ({pct:.1f}%)")
            print()
        
        # Issues
        if self.issues:
            print("⚠️  ISSUES FOUND")
            print("-" * 80)
            
            # Group by type
            sample_issues = [i for i in self.issues if i['type'] == 'sample']
            batch_issues = [i for i in self.issues if i['type'] == 'batch']
            
            if sample_issues:
                print(f"\nSample Issues ({len(sample_issues)}):")
                for issue in sample_issues[:20]:  # Show first 20
                    print(f"  Sample {issue['idx']}:")
                    for msg in issue['issues']:
                        print(f"    - {msg}")
                if len(sample_issues) > 20:
                    print(f"  ... and {len(sample_issues) - 20} more")
            
            if batch_issues:
                print(f"\nBatch Issues ({len(batch_issues)}):")
                for issue in batch_issues:
                    print(f"  Batch {issue['batch_idx']}:")
                    for msg in issue['issues']:
                        print(f"    - {msg}")
        else:
            print("✅ NO ISSUES FOUND")
            print("-" * 80)
        
        # Critical issues summary
        print("\n🔍 CRITICAL ISSUES SUMMARY")
        print("-" * 80)
        critical_issues = []
        
        if self.stats['empty_targets'] > 0:
            pct = (self.stats['empty_targets'] / self.stats['total_checked']) * 100
            critical_issues.append(
                f"❌ {self.stats['empty_targets']} samples ({pct:.1f}%) have empty targets"
            )
        
        if self.stats['very_short_targets'] > 0:
            pct = (self.stats['very_short_targets'] / self.stats['total_checked']) * 100
            critical_issues.append(
                f"⚠️  {self.stats['very_short_targets']} samples ({pct:.1f}%) have very short targets"
            )
        
        if self.stats['label_mismatches'] > 0:
            pct = (self.stats['label_mismatches'] / self.stats['total_checked']) * 100
            critical_issues.append(
                f"❌ {self.stats['label_mismatches']} samples ({pct:.1f}%) have label mismatches"
            )
        
        if not critical_issues:
            print("✅ No critical issues found")
        else:
            for issue in critical_issues:
                print(issue)
        
        print("\n" + "=" * 80)
        
        # Recommendations
        if critical_issues or self.issues:
            print("\n💡 RECOMMENDATIONS")
            print("-" * 80)
            
            if self.stats['empty_targets'] > 0:
                print("1. Empty targets will cause training instability.")
                print("   → Check data preprocessing to ensure assistant responses exist")
            
            target_std = self.stats.get('target_len_std', 0)
            target_mean = self.stats.get('target_len_mean', 0)
            cv = (target_std / target_mean * 100) if target_mean > 0 else 0
            
            if cv > 100 or target_std > 50:
                print("2. High target length variance can cause volatile loss.")
                print("   → The issue is MIXING very short (1-4 tokens) and very long (100+ tokens) targets")
                print("   → Solutions:")
                print("      a) Use loss normalization (divide loss by number of target tokens)")
                print("      b) Filter extreme outliers (e.g., < 3 tokens or > 200 tokens)")
                print("      c) Use curriculum learning (start with medium-length targets)")
                print("      d) Use gradient accumulation with mixed-length batches")
            elif self.stats['very_short_targets'] > 0:
                print("2. Short targets are valid, but monitor batch composition.")
                print("   → If loss is volatile, check if batches have mixed short/long targets")
            
            if self.stats['label_mismatches'] > 0:
                print("3. Label mismatches indicate data corruption.")
                print("   → Check dataset.__getitem__ implementation")
            
            if len(self.stats['file_distribution']) == 1:
                print("4. All samples from single file - may lack diversity.")
                print("   → Consider using multiple parquet files")
            
            target_std = self.stats.get('target_len_std', 0)
            if target_std > 50:
                print("5. High target length variance may cause training instability.")
                print("   → Consider filtering or normalizing target lengths")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Phase 2 data loading issues"
    )
    
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
        help="Maximum sequence length (should match training config)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to check (default: 1000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for batch checks (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Initialize model to get tokenizer and image processor
    print("Initializing model components...")
    config = LLaVAConfig()
    model = LLaVAModel(config)
    tokenizer = model.language_model.tokenizer
    image_processor = model.vision_encoder.processor
    
    # Build dataset
    print("Loading dataset...")
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
    
    # Run diagnostics
    diagnostics = DataDiagnostics(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    
    results = diagnostics.run_diagnostics()
    diagnostics.print_report()
    
    # Return non-zero exit code if critical issues found
    if (diagnostics.stats['empty_targets'] > 0 or
        diagnostics.stats['label_mismatches'] > 0):
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

