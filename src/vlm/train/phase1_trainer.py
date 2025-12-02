"""Training logic for LLaVA Phase 1 (Pretraining)."""

import os
import math
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
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
        save_checkpoint_interval: int = 500,
        precision: str = "fp16",
        gradient_accumulation_steps: int = 1,
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
                (default: 500 steps)
            precision: Mixed precision mode: "fp16", "bf16", or "fp32"
                (default: "fp16")
            gradient_accumulation_steps: Number of gradient accumulation steps
                (default: 1, no accumulation)
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.precision = precision

        # Auto-detect rank and world_size from environment or dist
        # (needed early for logging)
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            # Fall back to environment variables or defaults
            self.rank = int(os.environ.get('RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        # Auto-detect DDP: check if model is wrapped with DDP
        # DDP-wrapped models have a 'module' attribute and class name
        self.ddp_enabled = (
            hasattr(model, 'module') and
            type(model).__name__ == 'DistributedDataParallel'
        )

        # Setup mixed precision training (fp16, bf16, or fp32)
        self.scaler = None
        self.amp_dtype = None
        self.device_type = "cuda" if device.type == "cuda" else device.type

        if precision == "fp16":
            # FP16 support: CUDA (with gradient scaling) or MPS/CPU (limited)
            if device.type == "cuda":
                self.amp_dtype = torch.float16
                # FP16 on CUDA requires gradient scaling
                self.scaler = torch.cuda.amp.GradScaler()
                if self.rank == 0:
                    print(
                        "✅ FP16/Mixed precision training enabled "
                        "(using autocast with float16, CUDA)"
                    )
            elif device.type == "mps":
                # MPS supports fp16 but may not be optimal
                self.amp_dtype = torch.float16
                self.scaler = None  # MPS doesn't need gradient scaling
                if self.rank == 0:
                    print(
                        "✅ FP16/Mixed precision training enabled "
                        "(using autocast with float16, MPS)"
                    )
            else:
                # CPU - fp16 not recommended but can work
                if self.rank == 0:
                    print(
                        "⚠️  FP16 on CPU is not recommended. "
                        "Using fp32."
                    )
                self.amp_dtype = None
                self.scaler = None
        elif precision == "bf16":
            # BF16 support: CUDA (with bf16 support) or MPS
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                self.scaler = None
                if self.rank == 0:
                    print(
                        "✅ BF16/Mixed precision training enabled "
                        "(using autocast with bfloat16, CUDA)"
                    )
            elif device.type == "mps":
                # MPS natively supports bf16
                self.amp_dtype = torch.bfloat16
                self.scaler = None
                if self.rank == 0:
                    print(
                        "✅ BF16/Mixed precision training enabled "
                        "(using autocast with bfloat16, MPS)"
                    )
            else:
                if self.rank == 0:
                    if device.type == "cpu":
                        print(
                            "⚠️  BF16 on CPU is not recommended. Using fp32."
                        )
                    else:
                        print(
                            "⚠️  BF16 not supported on this device. "
                            "Using fp32."
                        )
                self.amp_dtype = None
                self.scaler = None
        elif precision == "fp32":
            self.amp_dtype = None
            self.scaler = None
            if self.rank == 0:
                print("Using fp32 precision")
        else:
            raise ValueError(
                f"Invalid precision: {precision}. "
                "Must be 'fp16', 'bf16', or 'fp32'"
            )

        # Get underlying model if wrapped with DDP
        if self.ddp_enabled:
            self.underlying_model = model.module
        else:
            self.underlying_model = model

        # Phase 1: Freeze VLM/LLM, Train Connector
        # Set training stage on underlying model
        if self.rank == 0 and not self.use_wandb:
            print(
                "Phase1Trainer: Setting training stage to 1 (Pretraining)..."
            )
        self.underlying_model.set_training_stage(1)

        # Initialize wandb if enabled (only on rank 0)
        self.wandb = None
        if self.use_wandb and self.rank == 0:
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
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                }

                # Add hyperparameters if provided
                if hyperparams:
                    wandb_config.update(hyperparams)

                # Extract learning rate from optimizer if not in hyperparams
                if hyperparams is None or "learning_rate" not in hyperparams:
                    optimizer_lr = optimizer.param_groups[0].get("lr")
                    if optimizer_lr is not None:
                        wandb_config["learning_rate"] = optimizer_lr

                # Add precision status
                wandb_config["precision"] = self.precision
                wandb_config["amp_dtype"] = (
                    str(self.amp_dtype) if self.amp_dtype else "fp32"
                )

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
        if self.rank == 0:
            print("Starting training...")
        self.model.train()
        # Model should already be on device if DDP is enabled
        if not self.ddp_enabled:
            self.model.to(self.device)

        step = 0
        total_loss = 0
        accumulation_step = 0

        # Only show progress bar on rank 0
        if self.rank == 0:
            progress_bar = tqdm(range(self.max_steps), desc="Training")
        else:
            progress_bar = range(self.max_steps)

        # Infinite iterator over dataloader
        data_iter = iter(self.train_dataloader)

        for _ in progress_bar:
            step += 1
            accumulation_step += 1

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

            # Zero gradients only at the start of accumulation
            if accumulation_step == 1:
                self.optimizer.zero_grad()

            # Forward pass with mixed precision if enabled
            # For each user turn with images:
            # 1. Images → CLIPImageProcessor → pixel_values
            # 2. pixel_values → CLIP encoder → visual features
            # 3. visual features → connector → visual embeddings
            # 4. visual embeddings concatenated with text embeddings
            if self.amp_dtype is not None:
                # Mixed precision training with autocast (fp16 or bf16)
                # Use device-appropriate autocast
                if self.device_type == "cuda":
                    with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            images=pixel_values
                        )
                        loss = outputs.loss
                elif self.device_type == "mps":
                    with torch.amp.autocast(
                        device_type="mps", dtype=self.amp_dtype
                    ):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            images=pixel_values
                        )
                        loss = outputs.loss
                else:
                    # CPU or other - use generic autocast
                    with torch.amp.autocast(
                        device_type="cpu", dtype=self.amp_dtype
                    ):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            images=pixel_values
                        )
                        loss = outputs.loss
            else:
                # FP32 training
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
                if self.rank == 0:
                    print(msg)
                # Reset accumulation when skipping a batch
                # to avoid partial accumulation
                accumulation_step = 0
                continue

            # Scale loss by accumulation steps to get average gradient
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                # FP16 on CUDA requires gradient scaling
                self.scaler.scale(loss).backward()
            else:
                # Standard backward pass (bf16, fp16 on MPS, or fp32)
                loss.backward()

            # Only update optimizer and scheduler after accumulating all steps
            grad_norm = None
            if accumulation_step == self.gradient_accumulation_steps:
                if self.scaler is not None:
                    # FP16 on CUDA - unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm
                    )
                    # Optimizer step with scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping (bf16, fp16 on MPS, or fp32)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm
                    )
                    # Optimizer step
                    self.optimizer.step()

                # Learning rate scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                # Reset accumulation step counter
                accumulation_step = 0

            # Update stats (unscale loss for logging)
            loss_value = loss.item() * self.gradient_accumulation_steps
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

            # Only print logs on rank 0
            # Only log grad_norm when we actually step (after accumulation)
            if self.rank == 0:
                if not self.use_wandb and step % 100 == 0:
                    grad_norm_str = (
                        f"{grad_norm:.4f}" if grad_norm is not None
                        else "accumulating"
                    )
                    print(
                        f"Step {step}: Loss: {loss_value:.4f}, "
                        f"Avg Loss: {avg_loss:.4f}, "
                        f"PPL: {perplexity:.2f}, "
                        f"Grad Norm: {grad_norm_str}, "
                        f"LR: {current_lr:.2e}"
                    )
                if hasattr(progress_bar, 'set_postfix'):
                    grad_norm_str = (
                        f"{grad_norm:.4f}" if grad_norm is not None
                        else "accum"
                    )
                    progress_bar.set_postfix({
                        "loss": f"{loss_value:.4f}",
                        "avg_loss": f"{avg_loss:.4f}",
                        "ppl": f"{perplexity:.2f}",
                        "grad_norm": grad_norm_str,
                        "lr": f"{current_lr:.2e}",
                    })

            # Log to wandb if enabled (only on rank 0)
            # Only log grad_norm when we actually step (after accumulation)
            if (self.use_wandb and self.wandb is not None and
                    self.rank == 0):
                log_dict = {
                    "train/loss": loss_value,
                    "train/avg_loss": avg_loss,
                    "train/perplexity": perplexity,
                    "train/param_norm": param_norm,
                    "train/learning_rate": current_lr,
                }

                # Only log grad_norm when we actually computed it (after step)
                if grad_norm is not None:
                    grad_norm_val = (
                        grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm
                    )
                    log_dict["train/grad_norm"] = grad_norm_val

                # Add GPU memory if available
                if gpu_memory_mb is not None:
                    log_dict["train/gpu_memory_mb"] = gpu_memory_mb

                self.wandb.log(log_dict, step=step)

            # Save checkpoint only on rank 0
            if step % self.save_checkpoint_interval == 0 and self.rank == 0:
                image_size = (
                    self.underlying_model.vision_encoder.image_size
                )
                filename = (
                    f"checkpoint_phase1_{self.precision}_{image_size}px.pt"
                )
                self.save_checkpoint(filename)

            # Early stopping if loss explodes
            if (loss_value > 100.0 or math.isnan(loss_value) or
                    math.isinf(loss_value)):
                if self.rank == 0:
                    print(
                        f"\nError: Loss exploded at step {step}: "
                        f"{loss_value}"
                    )
                    print("Stopping training to prevent further issues.")
                break

        # Handle final optimizer step if we have accumulated gradients
        # but haven't stepped yet
        if accumulation_step > 0:
            if self.scaler is not None:
                # FP16 on CUDA - unscale before clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Gradient clipping (bf16, fp16 on MPS, or fp32)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm
                )
                # Optimizer step
                self.optimizer.step()

            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

        # Final checkpoint save only on rank 0
        if self.rank == 0:
            image_size = (
                self.underlying_model.vision_encoder.image_size
            )
            filename = (
                f"checkpoint_phase1_{self.precision}_{image_size}px.pt"
            )
            self.save_checkpoint(filename)
            print("Training completed.")

        # Finish wandb run if enabled (only on rank 0)
        if self.use_wandb and self.wandb is not None and self.rank == 0:
            self.wandb.finish()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint with the corresponding precision.

        Args:
            filename: Name of the checkpoint file
        """
        if self.output_dir and self.rank == 0:
            # Expand ~ to home directory if present
            output_dir = Path(self.output_dir).expanduser()
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = output_dir / filename

            # Get state dict and convert to training precision if needed
            # (autocast doesn't change parameter dtype, so we convert on save)
            state_dict = self.underlying_model.state_dict()
            if self.amp_dtype is not None:
                # Convert state dict parameters to training precision dtype
                state_dict = {
                    k: v.to(dtype=self.amp_dtype) if v.is_floating_point()
                    else v
                    for k, v in state_dict.items()
                }

            # Save state dict with correct precision
            torch.save(state_dict, str(checkpoint_path))
            print(
                f"Saved checkpoint to {checkpoint_path} "
                f"(precision: {self.precision})"
            )
