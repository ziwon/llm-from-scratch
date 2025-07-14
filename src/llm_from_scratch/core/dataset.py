"""Dataset implementation for GPT training."""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from typing import List, Tuple, Optional
from tqdm import tqdm

from .tokenizer import get_tokenizer, TokenizerWrapper
from ..config import DataConfig


class GPTDataset(Dataset):
    """Dataset for GPT training with caching support."""
    
    def __init__(
        self,
        text_file: str,
        tokenizer: TokenizerWrapper,
        max_length: int,
        stride: int,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            text_file: Path to text file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for sliding window
            cache_dir: Directory for caching processed data
        """
        self.text_file = Path(text_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Try to load from cache
        if self.cache_dir and self._load_from_cache():
            print(f"Loaded dataset from cache: {self._get_cache_path()}")
        else:
            # Process text file
            print(f"Processing text file: {text_file}")
            self._process_text_file()
            
            # Save to cache
            if self.cache_dir:
                self._save_to_cache()
                print(f"Saved dataset to cache: {self._get_cache_path()}")
    
    def _get_cache_path(self) -> Path:
        """Get cache file path based on dataset parameters."""
        cache_name = f"{self.text_file.stem}_ml{self.max_length}_s{self.stride}.pkl"
        return self.cache_dir / cache_name
    
    def _load_from_cache(self) -> bool:
        """Try to load dataset from cache."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.input_ids = data['input_ids']
                    self.target_ids = data['target_ids']
                return True
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return False
    
    def _save_to_cache(self):
        """Save dataset to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_cache_path()
        
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'input_ids': self.input_ids,
                'target_ids': self.target_ids
            }, f)
    
    def _process_text_file(self):
        """Process text file into training sequences."""
        # Read text
        with open(self.text_file, 'r', encoding='utf-8-sig') as f:
            text = f.read()
        
        # Tokenize
        print("Tokenizing text...")
        token_ids = self.tokenizer.encode(text)
        print(f"Total tokens: {len(token_ids):,}")
        
        # Create sequences with sliding window
        self.input_ids = []
        self.target_ids = []
        
        for i in tqdm(range(0, len(token_ids) - self.max_length, self.stride), 
                      desc="Creating sequences"):
            input_chunk = token_ids[i:i + self.max_length]
            target_chunk = token_ids[i + 1:i + self.max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
        print(f"Created {len(self.input_ids):,} training sequences")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    config: DataConfig,
    tokenizer: Optional[TokenizerWrapper] = None,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create dataloader from config.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer instance (creates new if None)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        DataLoader instance
    """
    if tokenizer is None:
        tokenizer = TokenizerWrapper()
    
    dataset = GPTDataset(
        text_file=config.train_file,
        tokenizer=tokenizer,
        max_length=config.max_length,
        stride=config.stride,
        cache_dir=config.cache_dir
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True  # Drop last incomplete batch
    )
    
    return dataloader