"""Training logic for LLaVA."""

import os
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm.models.llava import LLaVAModel


class Phase1Trainer:
    """Trainer for LLaVA Phase 1 (Pretraining)."""

    def __init__(
        self,
        model: LLaVAModel,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str,
        max_steps: int,
        log_interval: int = 10,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        hyperparams: Optional[dict] = None,
        save_checkpoint_interval: int = 100,
    ):
        """Initialize Phase 1 Trainer.

        Args:
            model: LLaVA model to train
            train_dataloader: DataLoader for training data
            optimizer: Optimizer
            device: Device to train on
            output_dir: Directory to save checkpoints
            max_steps: Maximum number of training steps
            log_interval: Interval for logging loss
            max_grad_norm: Maximum gradient norm for clipping
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name (required if use_wandb=True)
            wandb_run_name: W&B run name (optional, auto-generated if None)
            scheduler: Learning rate scheduler (optional)
            hyperparams: Dictionary of hyperparameters to log
                (e.g., learning_rate, batch_size)
            save_checkpoint_interval: Interval for saving checkpoints
                (default: 20 steps)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.save_checkpoint_interval = save_checkpoint_interval

        # Phase 1: Freeze VLM/LLM, Train Connector
        if not self.use_wandb:
            print(
                "Phase1Trainer: Setting training stage to 1 (Pretraining)..."
            )
        self.model.set_training_stage(1)
        
        # Initialize wandb if enabled
        self.wandb = None
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                
                # Build config with hyperparameters
                wandb_config = {
                    "max_steps": max_steps,
                    "log_interval": log_interval,
                    "max_grad_norm": max_grad_norm,
                    "device": str(device),
                    "output_dir": output_dir,
                }
                
                # Add hyperparameters if provided
                if hyperparams:
                    wandb_config.update(hyperparams)
                
                # Extract learning rate from optimizer if not in hyperparams
                if hyperparams is None or "learning_rate" not in hyperparams:
                    optimizer_lr = optimizer.param_groups[0].get("lr")
                    if optimizer_lr is not None:
                        wandb_config["learning_rate"] = optimizer_lr
                
                wandb.init(
                    project=wandb_project or "llava-pretrain",
                    name=wandb_run_name,
                    config=wandb_config
                )
                print("Weights & Biases logging enabled.")
            except ImportError:
                print(
                    "Warning: wandb not installed. "
                    "Install with: pip install wandb"
                )
                self.use_wandb = False

    def train(self):
        """Run training loop."""
        print("Starting training...")
        self.model.train()
        self.model.to(self.device)

        step = 0
        total_loss = 0

        progress_bar = tqdm(range(self.max_steps), desc="Training")

        # Infinite iterator over dataloader
        data_iter = iter(self.train_dataloader)

        for _ in progress_bar:
            step += 1

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Handle images (may be None for text-only turns)
            pixel_values = None
            if batch.get('pixel_values') is not None:
                pixel_values = batch['pixel_values'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # For each user turn with images:
            # 1. Images → CLIPImageProcessor → pixel_values
            # 2. pixel_values → CLIP encoder → visual features
            # 3. visual features → connector → visual embeddings
            # 4. visual embeddings concatenated with text embeddings
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=pixel_values
            )
            loss = outputs.loss

            # Check for NaN/Inf
            if not torch.isfinite(loss):
                msg = (f"Warning: Non-finite loss at step {step}: "
                       f"{loss.item()}")
                print(msg)
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            
            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Update stats
            loss_value = loss.item()
            total_loss += loss_value
            avg_loss = total_loss / step
            
            # Compute additional metrics
            # Perplexity: exp(loss), capped to avoid overflow
            perplexity = math.exp(min(loss_value, 10.0))
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Compute parameter norm (L2 norm of all trainable parameters)
            param_norm = 0.0
            for param in self.model.parameters():
                if param.requires_grad:
                    param_norm += param.data.norm(2).item() ** 2
            param_norm = math.sqrt(param_norm)
            
            # Get GPU memory usage if available
            gpu_memory_mb = None
            if self.device.type == "cuda":
                gpu_memory_mb = (
                    torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                )
            elif self.device.type == "mps":
                # MPS doesn't have direct memory query, skip for now
                pass

            # Only print logs if wandb is not enabled
            if not self.use_wandb:
                print(
                    f"Step {step}: Loss: {loss_value:.4f}, "
                    f"Avg Loss: {avg_loss:.4f}, "
                    f"PPL: {perplexity:.2f}, "
                    f"Grad Norm: {grad_norm:.4f}, "
                    f"LR: {current_lr:.2e}"
                )
            progress_bar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "ppl": f"{perplexity:.2f}",
                "grad_norm": f"{grad_norm:.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            # Log to wandb if enabled
            if self.use_wandb and self.wandb is not None:
                grad_norm_val = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
                
                log_dict = {
                    "train/loss": loss_value,
                    "train/avg_loss": avg_loss,
                    "train/perplexity": perplexity,
                    "train/grad_norm": grad_norm_val,
                    "train/param_norm": param_norm,
                    "train/learning_rate": current_lr,
                }
                
                # Add GPU memory if available
                if gpu_memory_mb is not None:
                    log_dict["train/gpu_memory_mb"] = gpu_memory_mb
                
                self.wandb.log(log_dict, step=step)
            
            if step % self.save_checkpoint_interval == 0:
                self.save_checkpoint("checkpoint_phase1.pt")

            # Early stopping if loss explodes
            if (loss_value > 100.0 or math.isnan(loss_value) or
                    math.isinf(loss_value)):
                print(f"\nError: Loss exploded at step {step}: {loss_value}")
                print("Stopping training to prevent further issues.")
                break
            
        self.save_checkpoint("checkpoint_phase1.pt")
        print("Training completed.")
        
        # Finish wandb run if enabled
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Name of the checkpoint file
        """
        if self.output_dir:
            # Expand ~ to home directory if present
            output_dir = Path(self.output_dir).expanduser()
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = output_dir / filename
            torch.save(self.model.state_dict(), str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")
