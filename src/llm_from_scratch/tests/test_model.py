"""Test model creation and forward pass."""
import torch
import pytest

from llm_from_scratch.config import ModelConfig
from llm_from_scratch.core import GPTModel


def test_model_creation():
    """Test model can be created."""
    config = ModelConfig(
        vocab_size=1000,
        context_length=128,
        embed_dim=256,
        n_heads=8,
        n_layers=4
    )
    
    model = GPTModel(config)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_forward():
    """Test model forward pass."""
    config = ModelConfig(
        vocab_size=1000,
        context_length=128,
        embed_dim=256,
        n_heads=8,
        n_layers=4
    )
    
    model = GPTModel(config)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    # Check output shape
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_model_param_count():
    """Test parameter counting."""
    config = ModelConfig(
        vocab_size=1000,
        context_length=128,
        embed_dim=256,
        n_heads=8,
        n_layers=4
    )
    
    model = GPTModel(config)
    num_params = model.get_num_params(non_embedding=True)
    
    assert num_params > 0
    assert isinstance(num_params, int)