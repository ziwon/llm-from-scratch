"""Generation CLI."""

from pathlib import Path

import torch
import typer
from rich.prompt import Prompt

from ..config import Config, GenerationConfig, ModelConfig
from ..core import GPTModel, TokenizerWrapper
from ..generation import generate_text
from ..utils import setup_logger
from ..utils.logging import console

app = typer.Typer(help="Generate text with trained model")


@app.command()
def generate(
    model_path: Path = typer.Argument(..., help="Path to model checkpoint"),
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="Input prompt"),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
    max_tokens: int = typer.Option(
        100, "--max-tokens", "-m", help="Maximum tokens to generate"
    ),
    temperature: float = typer.Option(
        0.8, "--temperature", "-t", help="Sampling temperature"
    ),
    top_k: int = typer.Option(50, "--top-k", help="Top-k sampling"),
    top_p: float = typer.Option(0.95, "--top-p", help="Top-p (nucleus) sampling"),
    repetition_penalty: float = typer.Option(
        1.0, "--repetition-penalty", help="Repetition penalty"
    ),
    num_samples: int = typer.Option(
        1, "--num-samples", "-n", help="Number of samples to generate"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive mode"
    ),
    device: str | None = typer.Option(None, "--device", help="Device (cuda/cpu)"),
):
    """Generate text with a trained model."""
    # Check model path
    if not model_path.exists():
        console.print(f"[red]Error: Model file '{model_path}' not found[/red]")
        raise typer.Exit(1)

    # Setup
    setup_logger()

    # Load checkpoint
    console.print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")

    # Get config from checkpoint or file
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        if isinstance(config_dict, dict) and "model" in config_dict:
            model_config_dict = config_dict["model"]
        else:
            model_config_dict = config_dict
        # Convert dict to ModelConfig object
        model_config = ModelConfig(**model_config_dict)
        console.print("Using config from checkpoint")
    elif config_file and config_file.exists():
        config = Config.from_yaml(str(config_file))
        model_config = config.model
        console.print(f"Using config from: {config_file}")
    else:
        console.print(
            "[red]Error: No config found in checkpoint and no config file specified[/red]"
        )
        raise typer.Exit(1)

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    console.print(f"Using device: {device}")

    # Create model and tokenizer
    console.print("Initializing model and tokenizer...")
    model = GPTModel.from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    tokenizer = TokenizerWrapper()

    # Create generation config
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # Interactive mode
    if interactive:
        console.print("\n[bold]Interactive Generation Mode[/bold]")
        console.print("Type 'quit' or 'exit' to stop")
        console.print("-" * 50)

        while True:
            try:
                prompt_text = Prompt.ask("\n[cyan]Enter prompt[/cyan]")

                if prompt_text.lower() in ["quit", "exit"]:
                    break

                if not prompt_text.strip():
                    continue

                console.print("\n[yellow]Generating...[/yellow]")

                for i in range(num_samples):
                    if num_samples > 1:
                        console.print(f"\n[bold]Sample {i + 1}:[/bold]")

                    generated = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt_text,
                        generation_config=gen_config,
                        device=device,
                    )

                    console.print(generated)

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                break

        console.print("\n[green]Goodbye![/green]")

    # Single prompt mode
    else:
        if not prompt:
            console.print(
                "[red]Error: Please provide a prompt with --prompt or use --interactive mode[/red]"
            )
            raise typer.Exit(1)

        console.print(f"\n[bold]Prompt:[/bold] {prompt}")
        console.print("\n[yellow]Generating...[/yellow]\n")

        for i in range(num_samples):
            if num_samples > 1:
                console.print(f"[bold]Sample {i + 1}:[/bold]")

            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                generation_config=gen_config,
                device=device,
            )

            console.print(generated)

            if i < num_samples - 1:
                console.print("-" * 50)


if __name__ == "__main__":
    app()
