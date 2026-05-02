"""Command-line interface for VisionFlow."""
from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from visionflow.config import PipelineConfig
from visionflow.pipeline import Pipeline

app = typer.Typer(
    add_completion=False,
    help="VisionFlow: real-time tracking & traffic analytics.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, console=console, show_path=False)],
    )


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True,
                                help="Path to pipeline YAML config."),
    source: str | None = typer.Option(None, "--source", "-s",
                                      help="Override source (file path, RTSP URL, or webcam index)."),
    show: bool | None = typer.Option(None, "--show/--no-show", help="Override display window."),
    out_video: Path | None = typer.Option(None, "--out-video", help="Override output video path."),
    out_csv: Path | None = typer.Option(None, "--out-csv", help="Override output CSV path."),
) -> None:
    """Run the analytics pipeline against a video source."""
    cfg = PipelineConfig.from_yaml(config)
    if source is not None:
        cfg.source = source
    if show is not None:
        cfg.output.show = show
    if out_video is not None:
        cfg.output.video = str(out_video)
    if out_csv is not None:
        cfg.output.csv = str(out_csv)

    _setup_logging(cfg.log_level)
    console.rule("[bold cyan]VisionFlow")
    console.print(f"Source: [bold]{cfg.source}[/bold]")
    console.print(f"Detector: [bold]{cfg.detector.weights}[/bold] (conf={cfg.detector.conf})")

    pipeline = Pipeline(cfg)
    summary = pipeline.run()

    table = Table(title="Run summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in summary.items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def init_config(
    out: Path = typer.Argument(..., help="Where to write the sample config."),
) -> None:
    """Write a sample pipeline config to disk."""
    cfg = PipelineConfig()
    cfg.to_yaml(out)
    console.print(f"Sample config written to [bold]{out}[/bold]")


@app.command()
def version() -> None:
    """Show the installed VisionFlow version."""
    from visionflow import __version__
    console.print(f"visionflow {__version__}")


if __name__ == "__main__":
    app()
