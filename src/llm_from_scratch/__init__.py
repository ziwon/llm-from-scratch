"""LLM from Scratch - Building a Large Language Model from the ground up."""

__version__ = "0.1.0"

from .config import Config, ModelConfig, TrainingConfig, GenerationConfig
from .core import GPTModel, GPTDataset, create_dataloader, get_tokenizer
from .training import Trainer
from .generation import generate, generate_text

__all__ = [
    "Config",
    "ModelConfig", 
    "TrainingConfig",
    "GenerationConfig",
    "GPTModel",
    "GPTDataset",
    "create_dataloader",
    "get_tokenizer",
    "Trainer",
    "generate",
    "generate_text",
]