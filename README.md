# LLM From Scratch

A GPT language model implementation built from scratch using PyTorch.

## Quick Start

```bash
# Setup environment
just setup

# Prepare training data
just prepare-data data/raw/your_text_file.txt

# Train the model
just train

# Generate text
just generate models/checkpoints/your_model.pt
```

## Project Structure

- `src/llm_from_scratch/core/` - Core components (model, dataset, tokenizer)
- `src/llm_from_scratch/training/` - Training infrastructure
- `src/llm_from_scratch/generation/` - Text generation
- `configs/` - Configuration files

## Commands

- `just setup` - Setup development environment
- `just train` - Train with default config
- `just test` - Run tests
- `just format` - Format code
- `just lint` - Run linters

## Features

- Complete GPT transformer implementation
- Configurable training pipeline
- Text generation with sampling strategies
- TensorBoard monitoring
- Checkpoint system

See `CLAUDE.md` for detailed documentation.