#!/usr/bin/env python3
"""Training script for LLaVA Phase 1 (Pretraining).

Dry run: 
uv run src/vlm/train/run.py --data_path /workspace/dataset/llava-pretrain/blip_laion_cc_sbu_558k.json --image_folder /workspace/dataset/llava-pretrain --max_steps 20000 --batch_size 64
"""

import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from vlm.configs.data_config import DataConfig
from vlm.configs.model_config import LLaVAConfig
from vlm.data.llava_pretrain_dataset import build_pretrain_dataloader
from vlm.models.llava import LLaVAModel
from vlm.train.trainer import Phase1Trainer


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
    
    # Verify trainable parameters (will be set correctly after Trainer init, but we check here for info)
    # We temporarily set it here just to print stats, or we can move stats printing after Trainer init.
    # Let's move stats printing after Trainer init or just rely on Trainer doing it.
    # For now, let's just let Phase1Trainer handle it.
    
    # 4. Setup Data
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
        print(f"Dataloader created with {len(dataloader)} batches.")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        print("Please ensure dataset is downloaded using ./setup.sh")
        return

    # 5. Setup Optimizer
    # We need to set stage 1 BEFORE optimizer to know which params require grad
    model.set_training_stage(1) 
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    # Setup Learning Rate Scheduler (Cosine Annealing)
    scheduler = None
    if args.use_cosine_schedule:
        min_lr = args.min_lr if args.min_lr is not None else 0.0
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.max_steps,
            eta_min=min_lr
        )
        print(
            f"Using cosine learning rate schedule: "
            f"T_max={args.max_steps}, "
            f"eta_min={scheduler.eta_min}"
        )
    
    # 6. Initialize Trainer
    trainer = Phase1Trainer(
        model=model,
        train_dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
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
        }
    )
    
    # 7. Start Training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA Phase 1 Training Sketch")
    
    # Data args
    parser.add_argument("--data_path", type=str, default="~/dataset/llava-pretrain/blip_laion_cc_sbu_558k.json", help="Path to dataset JSON")
    parser.add_argument("--image_folder", type=str, default="~/dataset/llava-pretrain", help="Path to image folder")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10, help="Number of training steps for sketch")
    parser.add_argument("--output_dir", type=str, default="~/checkpoints/llava", help="Output directory")
    
    # Learning rate schedule args
    parser.add_argument(
        "--use_cosine_schedule",
        action="store_true",
        help="Use cosine annealing LR schedule"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine schedule (default: 0.0)"
    )
    
    # Wandb args
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llava-pretrain", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: auto-generated)")
    
    args = parser.parse_args()
    train(args)
