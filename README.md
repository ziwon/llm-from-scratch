# LLM From Scratch

A GPT language model implementation built from scratch using PyTorch.

## Quick Start

```
# Setup environment
just setup

# Prepare training data
just download-sample
just combine-sample

# Train the model
just train-quick
llm-train --epochs 5 --batch-size 32 --max-length 64 --log-interval 10
Loading configuration from: configs/default.yaml

Configuration:
  Device: cuda
  Epochs: 5
  Batch size: 32
  Learning rate: 0.0004
  Training data: data/processed/all_train.txt

Initializing tokenizer...

Creating model...
  Total parameters: 162,223,104
  Model size: 618.8 MB (fp32)

Loading data...
Loaded dataset from cache: data/cache/all_train_ml64_s4.pkl

Initializing trainer...

Starting training...
==================================================
[07/17/25 03:56:20] INFO     Starting training for 5 epochs
                    INFO     Total steps: 55,820
                    INFO     Device: cuda
Epoch 1/5:   1%|                                | 83/11164 [00:10<22:23,  8.25it/s, loss=8.2145, lr=3.36e-05]
Epoch 1/5:   7%|                                | 807/11164 [01:38<20:53,  8.26it/s, loss=4.4358, lr=3.23e-04]
```

## Generate text

```
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

## Links

- [LLM From Scratch](https://github.com/rasbt/LLMs-from-scratch)
- [AI Engineering Hub](https://github.com/patchy631/ai-engineering-hub)
- [Hands-On Large Language Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)
- [(Youtube) Let's Build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [(Youtube) Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10)
