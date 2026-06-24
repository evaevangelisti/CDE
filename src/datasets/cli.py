from pathlib import Path

import typer

from cde.models import Language

from .builders import BuilderFactory
from .builders.base import Builder
from .config import DATASETS_DIR, DEFAULT_BUILDER

app: typer.Typer = typer.Typer(add_completion=False)


def get_registered_sources() -> list[str]:
    return list(BuilderFactory._registry.keys())


@app.command()
def main(
    dataset_path: Path = typer.Argument(
        help="Dataset path",
        exists=True,
        dir_okay=False,
    ),
    source: str = typer.Option(
        DEFAULT_BUILDER,
        help="Source",
    ),
    language: str = typer.Option(
        "en",
        help="Language code ('en' and 'it' are currently supported)",
    ),
    output_dir: Path = typer.Option(
        DATASETS_DIR,
        help="Output directory",
        file_okay=False,
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    builder: Builder = BuilderFactory.create(source, language=Language(language))
    builder.build(dataset_path=dataset_path, output_dir=output_dir)


if __name__ == "__main__":
    app()
