"""Sampling strategies for text generation."""

import torch


def top_k_sampling(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply top-k sampling to logits.

    Args:
        logits: Logits tensor
        k: Number of top tokens to consider

    Returns:
        Filtered logits
    """
    if k <= 0:
        return logits

    top_k_values, _ = torch.topk(logits, k)
    min_value = top_k_values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_value, torch.tensor(float("-inf")).to(logits.device), logits
    )


def top_p_sampling(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: Logits tensor
        p: Cumulative probability threshold

    Returns:
        Filtered logits
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Find where cumulative probability exceeds p
    sorted_indices_to_remove = cumulative_probs > p
    # Keep at least one token
    sorted_indices_to_remove[..., 0] = False

    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    return torch.where(
        indices_to_remove, torch.tensor(float("-inf")).to(logits.device), logits
    )


def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float = 1.0
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.

    Args:
        logits: Logits tensor
        generated_ids: Previously generated token IDs
        penalty: Repetition penalty factor

    Returns:
        Modified logits
    """
    if penalty == 1.0:
        return logits

    # Get unique generated tokens
    unique_ids = torch.unique(generated_ids)

    # Apply penalty
    for token_id in unique_ids:
        if logits[token_id] < 0:
            logits[token_id] *= penalty
        else:
            logits[token_id] /= penalty

    return logits
