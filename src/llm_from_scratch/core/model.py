"""GPT model implementation."""
import torch
import torch.nn as nn
from typing import Optional

from ..config import ModelConfig


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        
        # Query, Key, Value projections
        self.W_query = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.W_key = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.W_value = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        queries = self.W_query(x)  # (batch_size, seq_len, embed_dim)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.n_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention computation: (batch, n_heads, seq_len, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)  # (batch, n_heads, seq_len, seq_len)
        
        # Apply causal mask
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context_vec = attn_weights @ values  # (batch, n_heads, seq_len, head_dim)
        
        # Transpose back and reshape
        context_vec = context_vec.transpose(1, 2)  # (batch, seq_len, n_heads, head_dim)
        context_vec = context_vec.contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(context_vec)
        
        return output


class LayerNorm(nn.Module):
    """Layer normalization."""
    
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation."""
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network."""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = LayerNorm(config.embed_dim)
        self.norm2 = LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Attention block with residual connection
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward block with residual connection
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class GPTModel(nn.Module):
    """GPT model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        
        # Final layer norm and output projection
        self.final_norm = LayerNorm(config.embed_dim)
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings and apply dropout
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Pass through transformer blocks
        x = self.transformer_blocks(x)
        
        # Final norm and output projection
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters.
        
        Args:
            non_embedding: Exclude embedding parameters if True
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
        return n_params
    
    @classmethod
    def from_config(cls, config: ModelConfig) -> "GPTModel":
        """Create model from config."""
        return cls(config)