# LLM from Scratch - Justfile

# Default recipe
default:
    @just --list

# Setup development environment
setup:
    uv venv --python=python3.11 .venv
    uv pip install -e ".[dev]"
    just create-dirs

# Create necessary directories
create-dirs:
    #!/bin/bash
    mkdir -p data/{raw,processed,cache}
    mkdir -p models/checkpoints
    mkdir -p logs/tensorboard
    mkdir -p configs

# Install dependencies
install:
    uv pip install -e .

# Install with dev dependencies
install-dev:
    uv pip install -e ".[dev]"

# Install with experiment tracking
install-exp:
    uv pip install -e ".[dev,experiment]"

# Run tests
test:
    pytest

# Run tests with coverage
test-cov:
    pytest --cov=llm_from_scratch --cov-report=html
    @echo "Coverage report generated in htmlcov/"

# Format code
format:
    black src/ tests/
    ruff --fix src/ tests/

# Lint code
lint:
    black --check src/ tests/
    ruff src/ tests/
    mypy src/

# Clean generated files
clean:
    rm -rf __pycache__ .pytest_cache .coverage htmlcov/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Clean all (including data and models)
clean-all: clean
    rm -rf data/processed/* data/cache/*
    rm -rf models/checkpoints/*
    rm -rf logs/tensorboard/*

# Prepare data
prepare-data file:
    llm-prepare {{file}}

# Train model with default config
train:
    llm-train

# Train with specific config
train-with-config config:
    llm-train --config {{config}}

# Interactive generation
generate model:
    llm-generate --model {{model}}

# Generate with prompt
generate-prompt model prompt:
    llm-generate --model {{model}} --prompt "{{prompt}}"

# Start TensorBoard
tensorboard:
    tensorboard --logdir logs/tensorboard

# Run notebook server
notebook:
    jupyter notebook notebooks/

# Download sample data
download-sample:
    # Download harry potter books
    @echo "Downloading Harry Potter books..."
    @mkdir -p data/raw  
    @curl -L -o data/raw/harry-potter-books.zip https://www.kaggle.com/api/v1/datasets/download/shubhammaindola/harry-potter-books && \
        unzip data/raw/harry-potter-books.zip -d data/raw/ && \
        rm data/raw/harry-potter-books.zip
    @echo "Sample data downloaded to data/raw/"
    @echo "Please add your data to data/raw/"

# Combine all text files into one file
combine-sample:
    @echo "Combining text files into data/raw/all.txt..."
    @mkdir -p data/raw
    @cat data/raw/*.txt > data/raw/all.txt
    @echo "Combined text files saved to data/raw/all.txt"

# Quick train (small dataset, few epochs for testing)
train-quick:
    llm-train \
        --epochs 5 \
        --batch-size 32 \
        --max-length 64 \
        --log-interval 10

# Full train (default settings)
train-full:
    llm-train \
        --epochs 100 \
        --batch-size 128 \
        --save-interval 10

# Profile training
profile:
    python -m torch.utils.bottleneck \
        -c "import sys; sys.argv = ['llm-train', '--epochs', '1']; from llm_from_scratch.cli.train import app; app()"

# Check GPU
check-gpu:
    @python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Test CUDA functionality
test-cuda:
    @python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); x = torch.randn(2, 3).cuda(); y = torch.randn(3, 2).cuda(); z = torch.mm(x, y); print('CUDA test successful:', z.shape)"

# Run all checks before commit
pre-commit: format lint test
    @echo "All checks passed!"

# Create a new experiment
new-exp name:
    mkdir -p experiments/{{name}}
    cp configs/default.yaml experiments/{{name}}/config.yaml
    @echo "Created new experiment: {{name}}"

# Package for distribution
package:
    python -m build

# Help
help:
    @echo "LLM from Scratch - Development Commands"
    @echo "======================================"
    @echo "Setup:"
    @echo "  just setup          - Setup development environment"
    @echo "  just install-dev    - Install with dev dependencies"
    @echo ""
    @echo "Development:"
    @echo "  just format         - Format code"
    @echo "  just lint           - Run linters"
    @echo "  just test           - Run tests"
    @echo "  just clean          - Clean generated files"
    @echo ""
    @echo "Training:"
    @echo "  just train          - Train with default config"
    @echo "  just train-quick    - Quick training for testing"
    @echo "  just tensorboard    - Start TensorBoard"
    @echo ""
    @echo "Generation:"
    @echo "  just generate MODEL - Interactive text generation"