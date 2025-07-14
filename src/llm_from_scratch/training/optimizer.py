"""Optimizer creation utilities."""
import torch
from torch.optim import Optimizer
from typing import Dict, Any

from ..config import OptimizerConfig, TrainingConfig


def create_optimizer(
    model: torch.nn.Module,
    optimizer_config: OptimizerConfig,
    training_config: TrainingConfig
) -> Optimizer:
    """
    Create optimizer from config.
    
    Args:
        model: Model instance
        optimizer_config: Optimizer configuration
        training_config: Training configuration
        
    Returns:
        Optimizer instance
    """
    # Get parameters
    params = model.parameters()
    
    # Create optimizer based on type
    optimizer_type = optimizer_config.type.lower()
    
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=training_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=training_config.weight_decay
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=training_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=training_config.weight_decay
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=training_config.learning_rate,
            momentum=0.9,
            weight_decay=training_config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer