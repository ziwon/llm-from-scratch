"""Main trainer class."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Any
import time
from tqdm import tqdm

from ..config import Config
from ..utils.logging import get_logger
from ..utils.helpers import save_checkpoint, load_checkpoint
from .optimizer import create_optimizer
from .scheduler import create_scheduler


class Trainer:
    """Trainer class for GPT model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Configuration object
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader (optional)
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Setup device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            config.optimizer,
            config.training
        )
        
        # Calculate total training steps
        self.steps_per_epoch = len(train_dataloader)
        self.total_steps = self.steps_per_epoch * config.training.epochs
        
        self.scheduler = create_scheduler(
            self.optimizer,
            config.scheduler,
            config.training,
            self.total_steps
        )
        
        # Setup logging
        self.logger = get_logger()
        self.writer = None
        if config.logging.tensorboard:
            self.writer = SummaryWriter(config.tensorboard_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average epoch loss
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}/{self.config.training.epochs}",
            leave=True
        )
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.get_lr():.2e}'
            })
            
            # Log to tensorboard
            if self.writer and self.global_step % self.config.training.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.get_lr(), self.global_step)
                self.writer.add_scalar('train/perplexity', torch.exp(loss).item(), self.global_step)
        
        return epoch_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate model.
        
        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for input_ids, target_ids in tqdm(self.val_dataloader, desc="Validating"):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Run full training loop."""
        self.logger.info(f"Starting training for {self.config.training.epochs} epochs")
        self.logger.info(f"Total steps: {self.total_steps:,}")
        self.logger.info(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate() if self.val_dataloader else float('inf')
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Time: {epoch_time:.1f}s"
            )
            
            if self.writer:
                self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
                self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(
                    f"checkpoint_epoch_{epoch + 1}.pt",
                    train_loss=train_loss,
                    val_loss=val_loss
                )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt", is_best=True)
        
        # Save final model
        self.save_checkpoint("final_model.pt", is_final=True)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes")
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(
        self,
        filename: str,
        **kwargs
    ):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch,
            step=self.global_step,
            loss=kwargs.get('train_loss', 0.0),
            config=self.config,
            **kwargs
        )
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint."""
        checkpoint = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.device
        )
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('step', 0)
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']