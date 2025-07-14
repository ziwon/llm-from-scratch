"""Configuration management module."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import torch


@dataclass
class ModelConfig:
    """Model configuration."""
    vocab_size: int = 50257
    context_length: int = 128
    embed_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1
    bias: bool = False


@dataclass
class DataConfig:
    """Data configuration."""
    train_file: str = "data/processed/train.txt"
    val_file: Optional[str] = None
    max_length: int = 32
    stride: int = 4
    cache_dir: str = "data/cache"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 0.0004
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    save_interval: int = 10
    eval_interval: int = 10
    log_interval: int = 100
    gradient_accumulation: int = 1
    mixed_precision: bool = False
    compile_model: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "adamw"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    type: str = "cosine"
    min_lr: float = 1e-5


@dataclass
class GenerationConfig:
    """Generation configuration."""
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    tensorboard: bool = True
    wandb: bool = False
    project_name: str = "llm-from-scratch"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    checkpoint_dir: str = "models/checkpoints"
    log_dir: str = "logs"
    tensorboard_dir: str = "logs/tensorboard"
    
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    
    def __post_init__(self):
        """Post initialization setup."""
        # Setup device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create directories
        for dir_path in [self.checkpoint_dir, self.log_dir, self.tensorboard_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create nested configs
        config = cls()
        
        # Update model config
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        
        # Update data config
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        
        # Update training config
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        
        # Update optimizer config
        if 'optimizer' in data:
            config.optimizer = OptimizerConfig(**data['optimizer'])
        
        # Update scheduler config
        if 'scheduler' in data:
            config.scheduler = SchedulerConfig(**data['scheduler'])
        
        # Update generation config
        if 'generation' in data:
            config.generation = GenerationConfig(**data['generation'])
        
        # Update logging config
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        
        # Update paths and other settings
        if 'paths' in data:
            for key, value in data['paths'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Update remaining top-level settings
        for key in ['seed', 'device', 'num_workers']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'optimizer': self.optimizer.__dict__,
            'scheduler': self.scheduler.__dict__,
            'generation': self.generation.__dict__,
            'logging': self.logging.__dict__,
            'paths': {
                'checkpoint_dir': self.checkpoint_dir,
                'log_dir': self.log_dir,
                'tensorboard_dir': self.tensorboard_dir,
            },
            'seed': self.seed,
            'device': self.device,
            'num_workers': self.num_workers,
        }