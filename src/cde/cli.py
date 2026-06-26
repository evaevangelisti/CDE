from pathlib import Path
from typing import Any

import typer

from .annotate import generate_annotated_sentences
from .config import (
    ANNOTATED_SENTENCES_FILENAME,
    CHECKPOINT_FILENAME,
    DEFAULT_BACKEND,
    EVALUATION_FILENAME,
)
from .evaluator import Evaluator
from .generators import GeneratorFactory
from .generators.base import Generator
from .io import Dataset, save_annotated_sentences, save_evaluations
from .models import AnnotatedSentence, Backend

app: typer.Typer = typer.Typer(add_completion=False)


def _parse_backend_options(
    options: list[str],
) -> dict[str, Any]:
    """
    Parse backend options from a list of key=value strings.

    Args:
        options (list[str]): List of backend options in key=value format.

    Returns:
        dict[str, Any]: Dictionary of parsed backend options.

    Raises:
        typer.BadParameter: If any option is not in key=value format.
    """
    backend_options: dict[str, Any] = {}

    for option in options:
        if "=" not in option:
            raise typer.BadParameter(
                f"Backend option '{option}' must be in key=value format."
            )

        key, value = option.split("=", 1)

        match value.lower():
            case "true":
                backend_options[key] = True

            case "false":
                backend_options[key] = False

            case "none":
                backend_options[key] = None

            case _:
                try:
                    backend_options[key] = float(value) if "." in value else int(value)
                except ValueError:
                    backend_options[key] = value

    return backend_options


@app.command()
def main(
    model: str = typer.Argument(
        help="Model",
    ),
    dataset_paths: list[Path] = typer.Argument(
        help="Paths to datasets in JSONL format",
        exists=True,
        dir_okay=False,
    ),
    backend: Backend = typer.Option(
        DEFAULT_BACKEND,
        help="Backend to use for generation",
    ),
    backend_option: list[str] = typer.Option(
        [],
        "--backend-option",
        "-o",
        help="Backend options in key=value format",
    ),
    chunk_size: int | None = typer.Option(
        None,
        help="Chunk size for processing sentences",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        help="Output directory",
        file_okay=False,
    ),
) -> None:
    """
    Continuation-based Disambiguation Evaluator
    """
    try:
        backend_options: dict[str, Any] = _parse_backend_options(backend_option)
    except typer.BadParameter as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    try:
        generator: Generator = GeneratorFactory.create(
            backend,
            model,
            seed=seed,
            **backend_options,
        )
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    for dataset_path in dataset_paths:
        if dataset_path.suffix != ".jsonl":
            typer.echo(f"Dataset '{dataset_path}' is not a JSONL file", err=True)
            continue

        dataset_dir: Path = (
            output_dir / f"{model.replace(':', '-')}" / dataset_path.stem
        )

        try:
            sentences: Dataset = Dataset(dataset_path)
        except FileNotFoundError as e:
            typer.echo(str(e), err=True)
            continue

        if not sentences:
            typer.echo(f"Dataset '{dataset_path.stem}' is empty", err=True)
            continue

        annotated_sentences: list[AnnotatedSentence] = generate_annotated_sentences(
            generator,
            sentences,
            dataset_dir / CHECKPOINT_FILENAME,
            chunk_size=chunk_size,
        )

        save_annotated_sentences(
            annotated_sentences,
            dataset_dir / ANNOTATED_SENTENCES_FILENAME,
        )

        evaluator: Evaluator = Evaluator()
        evaluations: dict[str, Any] = evaluator.evaluate(annotated_sentences)

        save_evaluations(
            evaluations,
            dataset_dir / EVALUATION_FILENAME,
        )


if __name__ == "__main__":
    app()
