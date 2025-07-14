"""Test dataset functionality."""
import torch
import pytest
from pathlib import Path
import tempfile

from llm_from_scratch.core import GPTDataset, create_dataloader, TokenizerWrapper
from llm_from_scratch.config import DataConfig


@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test file. " * 100)  # Repeat to have enough text
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink()


def test_dataset_creation(sample_text_file):
    """Test dataset can be created from text file."""
    tokenizer = TokenizerWrapper()
    
    dataset = GPTDataset(
        text_file=str(sample_text_file),
        tokenizer=tokenizer,
        max_length=32,
        stride=4,
        cache_dir=None
    )
    
    assert len(dataset) > 0
    assert isinstance(dataset[0], tuple)
    assert len(dataset[0]) == 2  # input_ids, target_ids


def test_dataset_shapes(sample_text_file):
    """Test dataset returns correct shapes."""
    tokenizer = TokenizerWrapper()
    
    dataset = GPTDataset(
        text_file=str(sample_text_file),
        tokenizer=tokenizer,
        max_length=32,
        stride=4,
        cache_dir=None
    )
    
    input_ids, target_ids = dataset[0]
    
    assert input_ids.shape == (32,)
    assert target_ids.shape == (32,)
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(target_ids, torch.Tensor)


def test_dataset_caching(sample_text_file):
    """Test dataset caching functionality."""
    with tempfile.TemporaryDirectory() as cache_dir:
        tokenizer = TokenizerWrapper()
        
        # Create dataset (should save to cache)
        dataset1 = GPTDataset(
            text_file=str(sample_text_file),
            tokenizer=tokenizer,
            max_length=32,
            stride=4,
            cache_dir=cache_dir
        )
        
        # Create again (should load from cache)
        dataset2 = GPTDataset(
            text_file=str(sample_text_file),
            tokenizer=tokenizer,
            max_length=32,
            stride=4,
            cache_dir=cache_dir
        )
        
        assert len(dataset1) == len(dataset2)
        
        # Check cache file exists
        cache_files = list(Path(cache_dir).glob("*.pkl"))
        assert len(cache_files) == 1


def test_dataloader_creation(sample_text_file):
    """Test dataloader creation from config."""
    config = DataConfig(
        train_file=str(sample_text_file),
        max_length=32,
        stride=4,
        cache_dir=None
    )
    
    dataloader = create_dataloader(
        config=config,
        batch_size=8,
        shuffle=True
    )
    
    assert dataloader is not None
    
    # Get one batch
    batch = next(iter(dataloader))
    input_ids, target_ids = batch
    
    assert input_ids.shape == (8, 32)
    assert target_ids.shape == (8, 32)