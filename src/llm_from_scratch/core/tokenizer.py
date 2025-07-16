"""Tokenizer utilities."""

import tiktoken


def get_tokenizer(encoding_name: str = "gpt2"):
    """
    Get tokenizer instance.

    Args:
        encoding_name: Name of the encoding (default: "gpt2")

    Returns:
        Tokenizer instance
    """
    return tiktoken.get_encoding(encoding_name)


class TokenizerWrapper:
    """Wrapper for tiktoken tokenizer with additional utilities."""

    def __init__(self, encoding_name: str = "gpt2"):
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.tokenizer.n_vocab

        # Special tokens
        self.pad_token_id = None  # GPT-2 doesn't have padding token
        self.eos_token_id = 50256  # <|endoftext|>
        self.bos_token_id = 50256  # Same as EOS for GPT-2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]

    def batch_decode(self, token_ids_list: list[list[int]]) -> list[str]:
        """Decode multiple token ID lists."""
        return [self.decode(token_ids) for token_ids in token_ids_list]
