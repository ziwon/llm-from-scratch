"""Training CLI."""
import typer
from pathlib import Path
from typing import Optional
import torch

from ..config import Config
from ..core import GPTModel, create_dataloader, TokenizerWrapper
from ..training import Trainer
from ..utils import setup_logger, set_seed, count_parameters
from ..utils.logging import console


app = typer.Typer(help="Train GPT model")


@app.command()
def train(
    config_file: Optional[Path] = typer.Option(
        Path("configs/default.yaml"),
        "--config", "-c",
        help="Configuration file"
    ),
    data_file: Optional[Path] = typer.Option(
        None,
        "--data", "-d",
        help="Override training data file"
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs", "-e",
        help="Override number of epochs"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Override batch size"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Override learning rate"
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Override device (cuda/cpu)"
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume", "-r",
        help="Resume from checkpoint"
    ),
    compile_model: bool = typer.Option(
        False,
        "--compile",
        help="Compile model with torch.compile (PyTorch 2.0+)"
    ),
    max_length: Optional[int] = typer.Option(
        None,
        "--max-length",
        help="Override maximum sequence length"
    ),
    log_interval: Optional[int] = typer.Option(
        None,
        "--log-interval",
        help="Override logging interval"
    )
):
    """Train GPT model with configuration."""
    # Load configuration
    if not config_file.exists():
        console.print(f"[red]Error: Config file '{config_file}' not found[/red]")
        raise typer.Exit(1)
    
    console.print(f"Loading configuration from: {config_file}")
    config = Config.from_yaml(str(config_file))
    
    # Override config with CLI arguments
    if data_file:
        config.data.train_file = str(data_file)
    if epochs:
        config.training.epochs = epochs
    if batch_size:
        config.training.batch_size = batch_size
    if learning_rate:
        config.training.learning_rate = learning_rate
    if device:
        config.device = device
    if compile_model:
        config.training.compile_model = compile_model
    if max_length:
        config.data.max_length = max_length
    if log_interval:
        config.training.log_interval = log_interval
    
    # Ensure device is properly resolved
    config.__post_init__()
    
    # Setup
    setup_logger(level=config.logging.level)
    set_seed(config.seed)
    
    # Log configuration
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Device: {config.device}")
    console.print(f"  Epochs: {config.training.epochs}")
    console.print(f"  Batch size: {config.training.batch_size}")
    console.print(f"  Learning rate: {config.training.learning_rate}")
    console.print(f"  Training data: {config.data.train_file}")
    
    # Check data file
    if not Path(config.data.train_file).exists():
        console.print(f"\n[red]Error: Training data file '{config.data.train_file}' not found[/red]")
        console.print("[yellow]Please run 'llm-prepare' first to prepare your data[/yellow]")
        raise typer.Exit(1)
    
    # Create tokenizer
    console.print("\n[bold]Initializing tokenizer...[/bold]")
    tokenizer = TokenizerWrapper()
    config.model.vocab_size = tokenizer.vocab_size
    
    # Create model
    console.print("\n[bold]Creating model...[/bold]")
    model = GPTModel(config.model)
    
    # Compile model if requested (PyTorch 2.0+)
    if config.training.compile_model and hasattr(torch, 'compile'):
        console.print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Log model info
    num_params = count_parameters(model)
    console.print(f"  Total parameters: {num_params:,}")
    console.print(f"  Model size: {num_params * 4 / 1024**2:.1f} MB (fp32)")
    
    # Create dataloaders
    console.print("\n[bold]Loading data...[/bold]")
    train_dataloader = create_dataloader(
        config.data,
        tokenizer,
        config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_dataloader = None
    if config.data.val_file and Path(config.data.val_file).exists():
        val_config = config.data
        val_config.train_file = config.data.val_file
        val_dataloader = create_dataloader(
            val_config,
            tokenizer,
            config.training.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
    
    # Create trainer
    console.print("\n[bold]Initializing trainer...[/bold]")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    # Resume from checkpoint if specified
    if resume and resume.exists():
        console.print(f"\nResuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)
    
    # Start training
    console.print("\n[bold green]Starting training...[/bold green]")
    console.print("=" * 50)
    
    try:
        trainer.train()
        console.print("\n[bold green]Training completed successfully![/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        trainer.save_checkpoint("interrupted_checkpoint.pt", interrupted=True)
    except Exception as e:
        console.print(f"\n[red]Training failed with error: {e}[/red]")
        raise


if __name__ == "__main__":
    app()