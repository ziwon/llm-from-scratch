"""Text generation functions."""

import torch
from torch import nn

from ..config import GenerationConfig
from ..core.tokenizer import TokenizerWrapper
from .sampling import apply_repetition_penalty, top_k_sampling, top_p_sampling


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """
    Generate tokens autoregressively.

    Args:
        model: Language model
        input_ids: Input token IDs (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty factor
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID

    Returns:
        Generated token IDs
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.shape[0]

    # Get model's context length
    if hasattr(model, "config"):
        context_length = model.config.context_length
    else:
        context_length = model.position_embedding.weight.shape[0]

    # Initialize generated tokens with input
    generated = input_ids

    # Track which sequences are done
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Get model input (truncate to context length)
        model_inputs = generated[:, -context_length:]

        # Get model predictions
        outputs = model(model_inputs)
        next_token_logits = outputs[:, -1, :]

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(batch_size):
                next_token_logits[i] = apply_repetition_penalty(
                    next_token_logits[i], generated[i], repetition_penalty
                )

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k sampling
        if top_k is not None and top_k > 0:
            next_token_logits = top_k_sampling(next_token_logits, top_k)

        # Apply top-p sampling
        if top_p is not None and top_p < 1.0:
            next_token_logits = top_p_sampling(next_token_logits, top_p)

        # Sample next tokens
        probs = torch.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        # Update generated sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Update unfinished sequences
        if eos_token_id is not None:
            unfinished_sequences = (
                unfinished_sequences * (next_tokens != eos_token_id).long().squeeze()
            )

        # Stop if all sequences are finished
        if unfinished_sequences.sum() == 0:
            break

    return generated


def generate_text(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    prompt: str,
    generation_config: GenerationConfig | None = None,
    device: torch.device | None = None,
    **kwargs,
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: Language model
        tokenizer: Tokenizer instance
        prompt: Input prompt
        generation_config: Generation configuration
        device: Device to run on
        **kwargs: Override generation parameters

    Returns:
        Generated text
    """
    if device is None:
        device = next(model.parameters()).device

    # Use default config if not provided
    if generation_config is None:
        generation_config = GenerationConfig()

    # Override config with kwargs
    config_dict = generation_config.__dict__.copy()
    config_dict.update(kwargs)

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    generated_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=config_dict["max_new_tokens"],
        temperature=config_dict["temperature"],
        top_k=config_dict.get("top_k"),
        top_p=config_dict.get("top_p"),
        repetition_penalty=config_dict.get("repetition_penalty", 1.0),
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode generated text
    generated_tokens = generated_ids[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


def generate_batch(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    generation_config: GenerationConfig | None = None,
    device: torch.device | None = None,
    **kwargs,
) -> list[str]:
    """
    Generate text for multiple prompts.

    Args:
        model: Language model
        tokenizer: Tokenizer instance
        prompts: List of input prompts
        generation_config: Generation configuration
        device: Device to run on
        **kwargs: Override generation parameters

    Returns:
        List of generated texts
    """
    if device is None:
        device = next(model.parameters()).device

    # Use default config if not provided
    if generation_config is None:
        generation_config = GenerationConfig()

    # Override config with kwargs
    config_dict = generation_config.__dict__.copy()
    config_dict.update(kwargs)

    # Encode prompts
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    # Pad to same length
    max_length = max(len(tokens) for tokens in encoded_prompts)
    padded_prompts = []
    for tokens in encoded_prompts:
        padded = tokens + [tokenizer.pad_token_id or 0] * (max_length - len(tokens))
        padded_prompts.append(padded)

    input_ids = torch.tensor(padded_prompts, dtype=torch.long, device=device)

    # Generate
    generated_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=config_dict["max_new_tokens"],
        temperature=config_dict["temperature"],
        top_k=config_dict.get("top_k"),
        top_p=config_dict.get("top_p"),
        repetition_penalty=config_dict.get("repetition_penalty", 1.0),
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode generated texts
    generated_texts = []
    for i in range(len(prompts)):
        tokens = generated_ids[i].tolist()
        text = tokenizer.decode(tokens)
        generated_texts.append(text)

    return generated_texts
