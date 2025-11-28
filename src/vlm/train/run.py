#!/usr/bin/env python3
"""Training script for LLaVA Phase 1 (Pretraining).

Dry run:
python src/vlm/train/run.py --data_path \
    ~/dataset/llava-pretrain/blip_laion_cc_sbu_558k.json \
        --image_folder ~/dataset/llava-pretrain \
    --max_steps 10 --batch_size 8 --use_cosine_schedule \
--use_wandb --output_dir ~/models/llava
"""

import argparse
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from vlm.configs.data_config import DataConfig
from vlm.configs.model_config import LLaVAConfig
from vlm.data.llava_pretrain_dataset import build_pretrain_dataloader
from vlm.models.llava import LLaVAModel
from vlm.train.trainer import Phase1Trainer


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


def train(args):
    """Run Phase 1 training."""
    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Initialize Model
    print("Initializing LLaVA model...")
    config = LLaVAConfig()
    model = LLaVAModel(config)

    # Note: Training stage is set by Phase1Trainer

    # Verify trainable parameters (will be set correctly after Trainer init,
    # but we check here for info)
    # We temporarily set it here just to print stats, or we can move stats
    # printing after Trainer init.
    # Let's move stats printing after Trainer init or just rely on Trainer
    # doing it.
    # For now, let's just let Phase1Trainer handle it.

    # 4. Setup Data
    if not args.use_wandb:
        print("Setting up data...")
    data_config = DataConfig(
        data_path=args.data_path,
        image_folder=args.image_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length
    )

    # Get tokenizer and processor from model components
    tokenizer = model.language_model.tokenizer
    image_processor = model.vision_encoder.processor

    # Build dataloader
    try:
        dataloader = build_pretrain_dataloader(
            config=data_config,
            tokenizer=tokenizer,
            image_processor=image_processor
        )
        if not args.use_wandb:
            print(f"Dataloader created with {len(dataloader)} batches.")
        
        # Cap max_steps to actual dataset size if needed
        actual_max_steps = min(args.max_steps, len(dataloader))
        if not args.use_wandb:
            if args.max_steps > len(dataloader):
                print(
                    f"Note: max_steps ({args.max_steps}) > dataset batches "
                    f"({len(dataloader)}). "
                    f"Capping max_steps to {actual_max_steps}."
                )
        print(f"Using actual_max_steps: {actual_max_steps} "
                f"(dataset batches: {len(dataloader)}, "
                f"requested max_steps: {args.max_steps})")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        print("Please ensure dataset is downloaded using ./setup.sh")
        return

    # 5. Setup Optimizer
    # We need to set stage 1 BEFORE optimizer to know which params
    # require grad
    model.set_training_stage(1)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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
        print(
            f"Using cosine learning rate schedule: "
            f"T_max={actual_max_steps}, "
            f"min_lr={min_lr}"
        )
    
    # 6. Initialize Trainer
    trainer = Phase1Trainer(
        model=model,
        train_dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        output_dir=args.output_dir,
        max_steps=actual_max_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        scheduler=scheduler,
        hyperparams={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "num_workers": args.num_workers,
            "use_cosine_schedule": args.use_cosine_schedule,
            "min_lr": args.min_lr if args.min_lr is not None else 0.0,
            "warmup_steps": (
                args.warmup_steps
                if args.warmup_steps is not None
                else int(0.01 * actual_max_steps)
            ),
        }
    )

    # 7. Start Training
    trainer.train()


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

    args = parser.parse_args()
    train(args)
