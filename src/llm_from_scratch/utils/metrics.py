"""Metrics and evaluation utilities."""

from collections import defaultdict

import numpy as np
import torch


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.

    Args:
        loss: Cross entropy loss

    Returns:
        Perplexity value
    """
    return np.exp(loss)


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate token-level accuracy.

    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)

    Returns:
        Accuracy as percentage
    """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).float()
    accuracy = correct.mean().item() * 100
    return accuracy


class MetricsTracker:
    """Track and aggregate metrics during training."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_epoch_metrics = defaultdict(list)

    def update(self, metric_name: str, value: float):
        """Update a metric value."""
        self.current_epoch_metrics[metric_name].append(value)

    def epoch_end(self) -> dict[str, float]:
        """Calculate epoch averages and reset."""
        epoch_averages = {}

        for metric_name, values in self.current_epoch_metrics.items():
            avg_value = np.mean(values)
            epoch_averages[metric_name] = avg_value
            self.metrics[metric_name].append(avg_value)

        # Reset current epoch metrics
        self.current_epoch_metrics.clear()

        return epoch_averages

    def get_history(self, metric_name: str) -> list[float]:
        """Get history for a specific metric."""
        return self.metrics.get(metric_name, [])

    def get_all_histories(self) -> dict[str, list[float]]:
        """Get all metric histories."""
        return dict(self.metrics)

    def get_best(self, metric_name: str, mode: str = "min") -> float | None:
        """
        Get best value for a metric.

        Args:
            metric_name: Name of the metric
            mode: 'min' or 'max'

        Returns:
            Best value or None if metric not found
        """
        history = self.get_history(metric_name)
        if not history:
            return None

        if mode == "min":
            return min(history)
        return max(history)


def calculate_gradient_norm(model: torch.nn.Module) -> float:
    """
    Calculate total gradient norm.

    Args:
        model: Model instance

    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def count_tokens(dataloader: torch.utils.data.DataLoader) -> int:
    """
    Count total tokens in dataloader.

    Args:
        dataloader: DataLoader instance

    Returns:
        Total number of tokens
    """
    total_tokens = 0
    for input_ids, _ in dataloader:
        total_tokens += input_ids.numel()
    return total_tokens
