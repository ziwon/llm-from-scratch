# LLM From Scratch

A GPT language model implementation built from scratch using PyTorch.

## Prerequisites

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install just into /usr/local/bin
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | sudo bash -s -- --to /usr/local/bin
```

## Quick Start

```
# Setup environment
$ just setup

# Prepare training data
$ just download-sample
$ just combine-sample
$ just prepare-data data/raw/all.txt
$ just prepare-data data/raw/all.txt
llm-prepare data/raw/all.txt
Reading input file: data/raw/all.txt
Original text length: 6,285,438 characters
Cleaning text...
Cleaned text length: 6,244,416 characters
✓ Saved training data: data/processed/all_train.txt (5,619,974 chars)
✓ Saved validation data: data/processed/all_val.txt (624,442 chars)

# Train the model
$ just train-quick
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
Processing text file: data/processed/all_train.txt
Tokenizing text...
Total tokens: 1,429,157
Creating sequences: 100%|████████████████████████████████████████████████████████████████████████████████████████| 357274/357274 [00:04<00:00, 89213.87it/s]
Created 357,274 training sequences
Saved dataset to cache: data/cache/all_train_ml64_s4.pkl

Initializing trainer...

Starting training...
==================================================
[07/31/25 02:13:27] INFO     Starting training for 5 epochs
                    INFO     Total steps: 55,820
                    INFO     Device: cuda
```

## Generation

```bash
$ just generate models/checkpoints/final_model.pt
llm-generate models/checkpoints/final_model.pt --interactive
Loading model from: models/checkpoints/final_model.pt
Using config from checkpoint
Using device: cuda
Initializing model and tokenizer...

Interactive Generation Mode
Type 'quit' or 'exit' to stop
--------------------------------------------------

Enter prompt: Dobby is

Generating...
Dobby is a free houseelf and he can obey anyone he likes and Dobby will do whatever Harry Potter wants him to do!” said Dobby, tears now
streaming down his shriveled little face onto his jumper. “Okay then,” said Harry, and he and Ron both released the elves, who fell right around
the dishy, who fell right in a hicorn, who had gone to his feet and was smiling in an expressionless eyes, who was looking very white

Enter prompt: Dobby is free?

Generating...
Dobby is free?” The elf shivered. He swayed. “Kreacher,” said Harry fiercely, “I order you –“ “Mundungus Fletcher,” croaked the elf, his eyes
still tight shut. “Mundungus Fletcher stole it all for the Mark?” “Kre talking to say, and Snape…” “Yes,” “ The effort,‘Harry,’s got out

Enter prompt: ^C
Interrupted

Goodbye!
```

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
