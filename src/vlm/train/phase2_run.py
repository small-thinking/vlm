#!/usr/bin/env python3
"""Training script for LLaVA Phase 2 (Instruction Tuning).

Single GPU/CPU training:
python src/vlm/train/phase2_run.py \
    --checkpoint ~/models/llava/checkpoint_phase1_fp16.pt \
    --data_path ~/dataset/llava-instruct-mix/data \
    --max_steps 10 --batch_size 2 --use_cosine_schedule \
    --use_wandb --output_dir ~/models/llava

Distributed training (automatically enabled when using torchrun):
torchrun --nproc_per_node=2 src/vlm/train/phase2_run.py \
    --checkpoint ~/models/llava/checkpoint_phase1_fp16.pt \
    --data_path ~/dataset/llava-instruct-mix/data \
    --max_steps 100000 --batch_size 16 --use_cosine_schedule \
    --gradient_accumulation_steps 4 --precision fp16 \
    --output_dir ~/models/llava --learning_rate 2e-5

Note: --data_path should point to a folder containing parquet files.
All .parquet files in the folder will be loaded and concatenated.
"""

import argparse
import math
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from vlm.configs.data_config import Phase2DataConfig
from vlm.configs.model_config import LLaVAConfig
from vlm.models.llava import LLaVAModel
from vlm.train.phase2_trainer import Phase2Trainer
from vlm.utils.ddp_sync import ddp_synchronized
from vlm.utils.model_logging import log_model_components


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-5,
) -> LambdaLR:
    """Create a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate (default: 1e-5)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale from [min_lr/base_lr, 1.0]
            min_lr_ratio = min_lr / optimizer.param_groups[0]["lr"]
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def setup_ddp(rank: int, world_size: int):
    """Initialize distributed process group.

    Uses MASTER_ADDR and MASTER_PORT from environment (set by torchrun),
    or defaults to localhost:29500 if not set.

    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    # Use environment variables set by torchrun, or defaults
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )


def cleanup_ddp():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    """Run Phase 2 training."""
    # 1. Auto-detect DDP from environment (set by torchrun)
    # Check if we're in a distributed environment
    rank = int(os.environ.get('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))

    # Enable DDP if environment variables are set and world_size > 1
    ddp_enabled = (
        rank >= 0 and
        local_rank >= 0 and
        world_size > 1
    )

    if ddp_enabled:
        # Initialize DDP (uses MASTER_ADDR/MASTER_PORT from env)
        setup_ddp(rank, world_size)

        # Set device based on local rank
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
        else:
            device = torch.device("cpu")
            if rank == 0:
                print("Warning: DDP requires CUDA, falling back to CPU")
    else:
        # Single GPU/CPU training
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if rank == 0:
        print(f"Using device: {device}")
        if ddp_enabled:
            print(
                f"Distributed training enabled: rank={rank}, "
                f"local_rank={local_rank}, world_size={world_size}"
            )

    # Determine model dtype based on precision (before model initialization)
    # Validate precision argument early
    precision = args.precision.lower()
    if precision not in ["fp16", "bf16", "fp32"]:
        if rank == 0:
            print(
                f"Error: Invalid precision '{precision}'. "
                "Must be 'fp16', 'bf16', or 'fp32'."
            )
        if ddp_enabled:
            cleanup_ddp()
        return

    # Wrap entire training setup and execution in DDP synchronization context
    # This ensures all ranks stay synchronized even on errors/early returns
    with ddp_synchronized(ddp_enabled=ddp_enabled):
        _train_impl(
            args, rank, local_rank, world_size, device, ddp_enabled, precision
        )

    # Cleanup DDP after context exits
    if ddp_enabled:
        cleanup_ddp()


def _train_impl(
    args, rank, local_rank, world_size, device, ddp_enabled, precision
):
    """Internal training implementation wrapped in DDP sync context."""

    # Determine model parameter dtype based on precision and device:
    # - FP16 on CUDA: requires GradScaler, so parameters must be fp32
    # - FP16 on MPS: no GradScaler, can use fp16 parameters
    # - BF16: no GradScaler, can use bf16 parameters
    # - FP32: use fp32 parameters
    # Note: When using GradScaler, parameters must be fp32 so gradients
    # are fp32 (GradScaler.unscale_() requires fp32 gradients)
    if precision == "bf16" and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            # BF16 without GradScaler - can use bf16 parameters
            model_dtype = torch.bfloat16
        else:
            # Fall back to fp32 if bf16 not supported
            model_dtype = torch.float32
    elif precision == "fp16" and device.type == "cuda":
        # FP16 on CUDA requires GradScaler, so parameters must be fp32
        model_dtype = torch.float32
    elif precision == "fp16" and device.type == "mps":
        # FP16 on MPS doesn't use GradScaler, can use fp16 parameters
        model_dtype = torch.float16
    else:
        # FP32 or other cases - use fp32 parameters
        model_dtype = torch.float32

    if rank == 0:
        print(
            f"Using precision: {precision} "
            f"(model dtype: {model_dtype}, "
            f"autocast will handle {precision} conversion)"
        )

    # 2. Initialize Model
    if rank == 0:
        print("Initializing LLaVA model...")
    config = LLaVAConfig()
    # Set language model dtype (fp32 for fp16/fp32, bf16 for bf16)
    config.language_model.torch_dtype = model_dtype
    model = LLaVAModel(config)

    # Load checkpoint if provided (Phase 2 fine-tunes from Phase 1 checkpoint)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.exists():
            if rank == 0:
                print(f"Error: Checkpoint not found: {checkpoint_path}")
                print(
                    "Please provide a valid checkpoint path with --checkpoint"
                )
            if ddp_enabled:
                cleanup_ddp()
            return

        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}...")
        # Load checkpoint to CPU first, then move to device
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)

        # Convert model parameters to match model_dtype if needed
        # Key distinction:
        # - FP16 on CUDA: requires fp32 parameters (for GradScaler)
        # - FP16 on MPS: can use fp16 parameters (no GradScaler)
        # - BF16: can use bf16 parameters (no GradScaler)
        checkpoint_dtype = None
        for param in model.parameters():
            if param.is_floating_point():
                checkpoint_dtype = param.dtype
                break

        if checkpoint_dtype != model_dtype:
            # Convert to match desired model_dtype
            model = model.to(dtype=model_dtype)
            if rank == 0:
                print(
                    f"✅ Checkpoint loaded and converted from "
                    f"{checkpoint_dtype} to {model_dtype}"
                )
                if (precision == "fp16" and device.type == "cuda" and
                        model_dtype == torch.float32):
                    print(
                        "   (fp32 required for GradScaler, "
                        "autocast uses fp16 during computation)"
                    )
        else:
            if rank == 0:
                print(
                    f"✅ Checkpoint loaded "
                    f"(parameters: {checkpoint_dtype}, "
                    f"autocast uses {precision})"
                )

    # Wrap model with DDP if enabled
    ddp_model = model  # Default to model if DDP not enabled
    if ddp_enabled:
        model = model.to(device)
        device_ids = [local_rank] if torch.cuda.is_available() else None
        ddp_model = DDP(model, device_ids=device_ids)
        # For accessing the underlying model (e.g., for set_training_stage)
        model = ddp_model.module
    else:
        # Move model to device if not using DDP
        model = model.to(device)

    # Log model components (before training stage is set)
    log_model_components(model, rank=rank, use_wandb=args.use_wandb)

    # 3. Setup Data
    if rank == 0 and not args.use_wandb:
        print("Setting up data...")
    data_config = Phase2DataConfig(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length
    )

    # Get tokenizer and processor from model components
    tokenizer = model.language_model.tokenizer
    image_processor = model.vision_encoder.processor
    num_visual_tokens = model.vision_encoder.num_visual_tokens

    # Build dataset
    try:
        from vlm.data.llava_instruct_dataset import (
            LLaVAInstructDataset,
            collate_fn
        )
        dataset = LLaVAInstructDataset(
            data_path=data_config.data_path,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=data_config.max_length,
            num_visual_tokens=num_visual_tokens,
        )

        # Create DistributedSampler if DDP is enabled
        sampler = None
        if ddp_enabled:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=data_config.shuffle,
                drop_last=data_config.drop_last
            )
            # Disable shuffle in DataLoader when using DistributedSampler
            shuffle = False
        else:
            shuffle = data_config.shuffle

        # Build dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=data_config.drop_last,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        if rank == 0:
            if not args.use_wandb:
                print(
                    f"Dataloader created with {len(dataloader)} "
                    f"batches per process."
                )
            if ddp_enabled:
                effective_bs = args.batch_size * world_size
                print(f"Total effective batch size: {effective_bs}")

        # Validate data if requested (only on rank 0)
        if args.validate_data and rank == 0:
            print("\n" + "=" * 80)
            print("VALIDATING DATA MASKING AND IMAGE PREPENDING")
            print("=" * 80)
            try:
                # Import validation function from scripts
                scripts_path = Path(__file__).parent.parent.parent / "scripts"
                sys.path.insert(0, str(scripts_path))
                from inspect_phase2_data import validate_masking_and_prepending
                # Determine device for validation
                if torch.cuda.is_available():
                    val_device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    val_device = torch.device("mps")
                else:
                    val_device = torch.device("cpu")

                validation_passed = validate_masking_and_prepending(
                    dataset,
                    model,
                    tokenizer,
                    num_samples=args.validation_samples,
                    device=val_device,
                )

                if not validation_passed:
                    print(
                        "\n❌ Data validation failed. "
                        "Please fix issues before training."
                    )
                    # Context manager will handle synchronization
                    return
                else:
                    print(
                        "\n✅ Data validation passed. "
                        "Proceeding with training."
                    )
            except ImportError as e:
                print(
                    f"⚠️  Warning: Could not import validation function: {e}"
                )
                print(
                    "   Validation skipped. "
                    "Install required dependencies if needed."
                )
            except Exception as e:
                print(f"⚠️  Warning: Validation failed with error: {e}")
                print("   Proceeding with training anyway.")
                import traceback
                traceback.print_exc()

        # Cap max_steps to actual dataset size if needed
        # Note: with DDP, each process sees len(dataloader) batches
        actual_max_steps = min(args.max_steps, len(dataloader))
        if rank == 0:
            if not args.use_wandb:
                if args.max_steps > len(dataloader):
                    print(
                        f"Note: max_steps ({args.max_steps}) > "
                        f"dataset batches ({len(dataloader)}). "
                        f"Capping max_steps to {actual_max_steps}."
                    )
            print(
                f"Using actual_max_steps: {actual_max_steps} "
                f"(dataset batches per process: {len(dataloader)}, "
                f"requested max_steps: {args.max_steps})"
            )
    except Exception as e:
        if rank == 0:
            print(f"Error creating dataloader: {e}")
            print("Please ensure dataset is downloaded using ./setup.sh")
        # Context manager will handle synchronization
        return

    # 4. Setup Optimizer
    # We need to set stage 2 BEFORE optimizer to know which params
    # require grad (connector + LLM, vision encoder frozen)
    model.set_training_stage(2)

    # Use DDP model for optimizer if DDP is enabled
    model_for_optimizer = ddp_model if ddp_enabled else model

    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad,
            model_for_optimizer.parameters()
        ),
        lr=args.learning_rate
    )

    # Setup Learning Rate Scheduler (Cosine with Warmup)
    scheduler = None
    if args.use_cosine_schedule:
        min_lr = args.min_lr if args.min_lr is not None else 0.0
        warmup_steps = (
            args.warmup_steps
            if args.warmup_steps is not None
            else int(0.01 * actual_max_steps)
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=actual_max_steps,
            min_lr=min_lr
        )
        if rank == 0:
            print(
                f"Using cosine learning rate schedule: "
                f"T_max={actual_max_steps}, "
                f"min_lr={min_lr}"
            )

    # 5. Initialize Trainer
    # Precision already validated and model dtype set above

    # Calculate effective batch size with gradient accumulation
    effective_batch_size = (
        args.batch_size * world_size * args.gradient_accumulation_steps
        if ddp_enabled
        else args.batch_size * args.gradient_accumulation_steps
    )

    if rank == 0 and args.gradient_accumulation_steps > 1:
        print(
            f"✅ Gradient accumulation enabled: "
            f"{args.gradient_accumulation_steps} steps "
            f"(effective batch size: {effective_batch_size})"
        )

    trainer = Phase2Trainer(
        model=model_for_optimizer,  # Pass DDP model if enabled
        train_dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        output_dir=args.output_dir,
        max_steps=actual_max_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        scheduler=scheduler,
        precision=precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        sampler=sampler,  # Pass sampler for proper epoch synchronization
        hyperparams={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "num_workers": args.num_workers,
            "use_cosine_schedule": args.use_cosine_schedule,
            "min_lr": args.min_lr if args.min_lr is not None else 0.0,
            "warmup_steps": (
                args.warmup_steps
                if args.warmup_steps is not None
                else int(0.01 * actual_max_steps)
            ),
            "ddp_enabled": ddp_enabled,
            "world_size": world_size,
            "precision": precision,
        }
    )

    # 6. Start Training
    # Training is wrapped in context manager,
    # so errors are handled automatically
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLaVA Phase 2 Training (Instruction Tuning)"
    )

    # Data args
    parser.add_argument(
        "--data_path",
        type=str,
        default="~/dataset/llava-instruct",
        help="Path to folder containing parquet files"
    )

    # Training args
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help=(
            "Max sequence length (should accommodate visual tokens, "
            "e.g., 1024 for 336px with 577 visual tokens). "
            "Sequences longer than max_length will be truncated."
        )
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Number of training steps for sketch")
    parser.add_argument("--output_dir", type=str,
                        default="~/checkpoints/llava",
                        help="Output directory")

    # Checkpoint args
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to Phase 1 checkpoint file "
            "(required for Phase 2 fine-tuning)"
        )
    )

    # Learning rate schedule args
    parser.add_argument(
        "--use_cosine_schedule",
        action="store_true",
        default=True,
        help="Use cosine annealing LR schedule with warmup "
        "(default: True)"
    )
    parser.add_argument(
        "--no_cosine_schedule",
        dest="use_cosine_schedule",
        action="store_false",
        help="Disable cosine learning rate schedule"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Number of warmup steps (default: 10%% of max_steps)"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine schedule (default: 0.0)"
    )

    # Wandb args
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str,
                        default="llava-instruct", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

    # Mixed precision args
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help=(
            "Mixed precision mode: 'fp16' (default), 'bf16', or 'fp32'. "
            "fp16: CUDA (with gradient scaling) or MPS. "
            "bf16: CUDA (with bf16 support) or MPS."
        )
    )

    # Gradient accumulation args
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=(
            "Number of gradient accumulation steps "
            "(default: 1, no accumulation)"
        )
    )

    # Validation args
    parser.add_argument(
        "--validate_data",
        action="store_true",
        help="Validate data masking and image prepending before training"
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=10,
        help="Number of samples to validate (default: 10)"
    )

    args = parser.parse_args()

    # Validate checkpoint is provided
    if not args.checkpoint:
        parser.error(
            "Phase 2 training requires a checkpoint from Phase 1. "
            "Please provide --checkpoint <path_to_phase1_checkpoint.pt>"
        )

    train(args)
