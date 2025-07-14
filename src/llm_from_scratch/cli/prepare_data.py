"""Data preparation CLI."""
import typer
from pathlib import Path
import re
from typing import Optional

from ..utils.logging import setup_logger, console


app = typer.Typer(help="Prepare text data for training")
logger = setup_logger()


def clean_text(text: str) -> str:
    """Clean text by normalizing whitespace."""
    # Replace multiple newlines with single space
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


@app.command()
def prepare(
    input_file: Path = typer.Argument(..., help="Input text file"),
    output_dir: Path = typer.Option(Path("data/processed"), help="Output directory"),
    train_split: float = typer.Option(0.9, help="Training data split ratio"),
    encoding: str = typer.Option("utf-8", help="File encoding"),
    clean: bool = typer.Option(True, help="Clean text (normalize whitespace)")
):
    """Prepare text data for training."""
    # Validate input
    if not input_file.exists():
        console.print(f"[red]Error: Input file '{input_file}' not found[/red]")
        raise typer.Exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input file
    console.print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding=encoding) as f:
        text = f.read()
    
    console.print(f"Original text length: {len(text):,} characters")
    
    # Clean text if requested
    if clean:
        console.print("Cleaning text...")
        text = clean_text(text)
        console.print(f"Cleaned text length: {len(text):,} characters")
    
    # Split into train/val if requested
    if 0 < train_split < 1:
        split_point = int(len(text) * train_split)
        train_text = text[:split_point]
        val_text = text[split_point:]
        
        # Save train file
        train_file = output_dir / f"{input_file.stem}_train.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write(train_text)
        console.print(f"[green]✓[/green] Saved training data: {train_file} ({len(train_text):,} chars)")
        
        # Save validation file
        val_file = output_dir / f"{input_file.stem}_val.txt"
        with open(val_file, 'w', encoding='utf-8') as f:
            f.write(val_text)
        console.print(f"[green]✓[/green] Saved validation data: {val_file} ({len(val_text):,} chars)")
    else:
        # Save single file
        output_file = output_dir / f"{input_file.stem}_processed.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        console.print(f"[green]✓[/green] Saved processed data: {output_file}")
    
    console.print("[green]Data preparation complete![/green]")


if __name__ == "__main__":
    app()