# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPT language model implementation built from scratch using PyTorch. The project provides a complete training and inference pipeline for a transformer-based language model, following the GPT architecture.

## Core Architecture

The codebase is organized into several key modules:

- **Core (`src/llm_from_scratch/core/`)**: Contains the fundamental components
  - `model.py`: GPT model implementation with transformer blocks, multi-head attention, and feed-forward layers
  - `dataset.py`: Dataset handling with caching support for training sequences
  - `tokenizer.py`: Wrapper around tiktoken for GPT-2 tokenization

- **Training (`src/llm_from_scratch/training/`)**: Training infrastructure
  - `trainer.py`: Main training loop with checkpointing, validation, and TensorBoard logging
  - `optimizer.py`: Optimizer creation (AdamW)
  - `scheduler.py`: Learning rate scheduling (cosine, linear, constant)

- **Generation (`src/llm_from_scratch/generation/`)**: Text generation capabilities
  - `generator.py`: Autoregressive text generation with sampling strategies
  - `sampling.py`: Top-k, top-p, and repetition penalty implementations

- **Configuration (`config.py`)**: Structured configuration system using dataclasses for all aspects (model, training, data, etc.)

## Development Commands

This project uses `just` as the task runner. Key commands:

### Setup and Installation
- `just setup` - Setup development environment with uv
- `just install-dev` - Install with development dependencies

### Development Workflow
- `just format` - Format code with black and ruff
- `just lint` - Run linters (black, ruff, mypy)
- `just test` - Run pytest tests
- `just test-cov` - Run tests with coverage report
- `just pre-commit` - Run format, lint, and test before committing

### Training and Data
- `just prepare-data <file>` - Prepare training data using `llm-prepare`
- `just train` - Train with default config
- `just train-quick` - Quick training for testing (5 epochs, small batch)
- `just train-full` - Full training with default settings
- `just train-with-config <config>` - Train with custom config file

### Generation and Monitoring
- `just generate <model>` - Interactive text generation
- `just tensorboard` - Start TensorBoard for monitoring
- `just notebook` - Start Jupyter notebook server

### Utilities
- `just check-gpu` - Check CUDA availability
- `just clean` - Clean generated files
- `just clean-all` - Clean everything including data and models

## CLI Tools

The project provides three main CLI commands:

1. **`llm-prepare`** (`src/llm_from_scratch/cli/prepare_data.py`): Prepares raw text data for training
2. **`llm-train`** (`src/llm_from_scratch/cli/train.py`): Trains the GPT model with extensive configuration options
3. **`llm-generate`** (`src/llm_from_scratch/cli/generate.py`): Generates text using trained models

## Configuration System

The project uses YAML configuration files with a hierarchical structure:
- `configs/default.yaml` - Default configuration
- Configuration can be overridden via CLI arguments
- Supports model, data, training, optimizer, scheduler, generation, and logging configs

## Data Pipeline

Training data flow:
1. Raw text files in `data/raw/`
2. Processed with `llm-prepare` into `data/processed/`
3. Cached token sequences in `data/cache/` for faster loading
4. Dataset uses sliding window approach with configurable stride

## Model Architecture

GPT implementation features:
- Multi-head attention with causal masking
- Layer normalization (custom implementation)
- GELU activation
- Positional embeddings
- Residual connections
- Weight initialization following GPT standards

## Testing

Run tests with:
- `pytest` or `just test` for basic testing
- `just test-cov` for coverage reports
- Tests are in `src/llm_from_scratch/tests/`

## Key Implementation Notes

- Uses tiktoken for GPT-2 compatible tokenization (vocab_size: 50257)
- Supports gradient clipping, mixed precision, and model compilation (PyTorch 2.0+)
- Checkpoint system saves model, optimizer state, and training progress
- TensorBoard integration for training visualization
- Supports both CPU and CUDA training with automatic device detection
- Generation includes top-k, top-p sampling and repetition penalty
