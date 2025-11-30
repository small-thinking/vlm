#!/usr/bin/env python3
"""Training script for LLaVA Phase 1 (Pretraining).

Single GPU/CPU training:
python src/vlm/train/phase1_run.py --data_path \
    ~/dataset/llava-pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ~/dataset/llava-pretrain \
    --max_steps 10 --batch_size 8 --use_cosine_schedule \
    --use_wandb --output_dir ~/models/llava

Distributed training (automatically enabled when using torchrun):
torchrun --nproc_per_node=2 src/vlm/train/phase1_run.py --data_path \
    ~/dataset/llava-pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ~/dataset/llava-pretrain \
    --max_steps 10000 --batch_size 64 --use_cosine_schedule \
    --gradient_accumulation_steps 4 --precision fp16 \
    --output_dir ~/models/llava --learning_rate 2e-3
"""

import argparse
import math
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from vlm.configs.data_config import Phase1DataConfig
from vlm.configs.model_config import LLaVAConfig
from vlm.models.llava import LLaVAModel
from vlm.train.phase1_trainer import Phase1Trainer


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
    """Run Phase 1 training."""
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

    # 2. Initialize Model
    if rank == 0:
        print("Initializing LLaVA model...")
    config = LLaVAConfig()
    model = LLaVAModel(config)

    # Wrap model with DDP if enabled
    ddp_model = model  # Default to model if DDP not enabled
    if ddp_enabled:
        model = model.to(device)
        device_ids = [local_rank] if torch.cuda.is_available() else None
        ddp_model = DDP(model, device_ids=device_ids)
        # For accessing the underlying model (e.g., for set_training_stage)
        model = ddp_model.module

    # 3. Setup Data
    if rank == 0 and not args.use_wandb:
        print("Setting up data...")
    data_config = Phase1DataConfig(
        data_path=args.data_path,
        image_folder=args.image_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length
    )

    # Get tokenizer and processor from model components
    tokenizer = model.language_model.tokenizer
    image_processor = model.vision_encoder.processor

    # Build dataset
    try:
        from vlm.data.llava_pretrain_dataset import (
            LLaVAPretrainDataset,
            collate_fn
        )
        dataset = LLaVAPretrainDataset(
            data_path=data_config.data_path,
            image_folder=data_config.image_folder,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=data_config.max_length,
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
        if ddp_enabled:
            cleanup_ddp()
        return

    # 4. Setup Optimizer
    # We need to set stage 1 BEFORE optimizer to know which params
    # require grad
    model.set_training_stage(1)

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
    # Validate precision argument
    precision = args.precision.lower()
    if precision not in ["fp16", "bf16", "fp8", "fp32"]:
        if rank == 0:
            print(
                f"Error: Invalid precision '{precision}'. "
                "Must be 'fp16', 'bf16', 'fp8', or 'fp32'."
            )
        if ddp_enabled:
            cleanup_ddp()
        return
    
    if rank == 0:
        print(f"Using precision: {precision}")

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

    trainer = Phase1Trainer(
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
    try:
        trainer.train()
    finally:
        # Cleanup DDP
        if ddp_enabled:
            cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLaVA Phase 1 Training Sketch"
    )

    # Data args
    parser.add_argument(
        "--data_path",
        type=str,
        default="~/dataset/llava-pretrain/"
        "blip_laion_cc_sbu_558k.json",
        help="Path to dataset JSON"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="~/dataset/llava-pretrain",
        help="Path to image folder"
    )

    # Training args
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Number of training steps for sketch")
    parser.add_argument("--output_dir", type=str,
                        default="~/checkpoints/llava",
                        help="Output directory")

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
                        default="llava-pretrain", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

    # Mixed precision args
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp8", "fp32"],
        help=(
            "Mixed precision mode: 'fp16' (default), 'bf16', 'fp8', or 'fp32'. "
            "fp16: CUDA (with gradient scaling) or MPS. "
            "bf16: CUDA (with bf16 support) or MPS. "
            "fp8: CUDA only, requires accelerate with Transformer Engine/MS-AMP."
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

    args = parser.parse_args()
    train(args)
