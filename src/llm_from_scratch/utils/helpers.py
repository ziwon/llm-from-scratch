"""Helper utilities."""
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """
    Get torch device.
    
    Args:
        device: Device string ("auto", "cuda", "cpu")
        
    Returns:
        torch.device instance
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_parameters(model: torch.nn.Module, non_embedding: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: Model instance
        non_embedding: Exclude embedding parameters if True
        
    Returns:
        Number of parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    if non_embedding and hasattr(model, 'position_embedding'):
        total_params -= model.position_embedding.weight.numel()
    
    return total_params


def save_config(config: Any, path: Path):
    """
    Save configuration to file.
    
    Args:
        config: Configuration object
        path: Save path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = vars(config)
    
    if path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    elif path.suffix in ['.yaml', '.yml']:
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def load_checkpoint(
    checkpoint_path: Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Load checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load state into
        optimizer: Optimizer instance to load state into
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    config: Optional[Any] = None,
    **kwargs
):
    """
    Save checkpoint.
    
    Args:
        checkpoint_path: Path to save checkpoint
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        step: Current step
        loss: Current loss
        config: Configuration object
        **kwargs: Additional items to save
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if config is not None:
        checkpoint['config'] = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
    
    # Add any additional items
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, checkpoint_path)