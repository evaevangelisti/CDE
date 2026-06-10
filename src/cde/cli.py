from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from .config import CHECKPOINT_INTERVAL, DEFAULT_DEFINITION_SELECTION_OPTIONS
from .generators.base import Generator
from .io import Dataset, save_annotated_sentences
from .models import AnnotatedContinuation, AnnotatedSentence, Sentence
from .utils import extract_predicted_sense_index

app: typer.Typer = typer.Typer(
    help="Continuation-based Disambiguation Evaluator",
    add_completion=False,
)


def _generate_annotated_continuations(
    generator: Generator,
    sentence: Sentence,
    seed: int | None = None,
) -> list[AnnotatedContinuation]:
    """
    Generate annotated continuations for a sentence.

    Args:
        generator (Generator): Generator.
        sentence (Sentence): Sentence.
        seed (int | None): Random seed.

    Returns:
        list[AnnotatedContinuation]: List of annotated continuations.
    """
    from .config import (
        CONTINUATION_DEFINITION_SELECTION_PROMPT_TEMPLATE,
        CONTINUATION_GENERATION_PROMPT_TEMPLATES,
        DEFAULT_CONTINUATION_GENERATION_OPTIONS,
    )
    from .models import Continuation
    from .utils import retrieve_template_fields, validate_continuation

    continuation_generation_options: dict[str, Any] = (
        DEFAULT_CONTINUATION_GENERATION_OPTIONS.copy()
    )

    definition_selection_options: dict[str, Any] = (
        DEFAULT_DEFINITION_SELECTION_OPTIONS.copy()
    )

    if seed is not None:
        continuation_generation_options["seed"] = seed
        definition_selection_options["seed"] = seed

    continuations: list[AnnotatedContinuation] = []

    for (
        condition,
        continuation_generation_prompt_template,
    ) in CONTINUATION_GENERATION_PROMPT_TEMPLATES.items():
        continuation_prompt_template_fields: set[str] = retrieve_template_fields(
            continuation_generation_prompt_template,
        )

        continuation_generation_prompt: str

        if "word" in continuation_prompt_template_fields:
            continuation_generation_prompt = (
                continuation_generation_prompt_template.format(
                    word=sentence.lemma,
                    sentence=sentence.sentence,
                )
            )
        else:
            continuation_generation_prompt = (
                continuation_generation_prompt_template.format(
                    sentence=sentence.sentence,
                )
            )

        continuation: str = generator.generate(
            continuation_generation_prompt,
            options=continuation_generation_options,
        )

        is_continuation_valid: bool = validate_continuation(
            sentence.sentence,
            continuation,
        )

        continuation_predicted_sense_index: int | None = None

        if is_continuation_valid:
            continuation_definition_selection_prompt: str = (
                CONTINUATION_DEFINITION_SELECTION_PROMPT_TEMPLATE.format(
                    word=sentence.lemma,
                    definitions="\n".join(
                        f"{i}) {definition}"
                        for i, definition in enumerate(sentence.definitions, start=1)
                    ),
                    sentence=sentence.sentence,
                    continuation=continuation,
                )
            )

            continuation_predicted_sense_index = extract_predicted_sense_index(
                generator.generate(
                    continuation_definition_selection_prompt,
                    options=definition_selection_options,
                ),
                len(sentence.definitions),
            )

        continuations.append(
            AnnotatedContinuation(
                Continuation(
                    condition=condition,
                    continuation=continuation,
                ),
                is_valid=is_continuation_valid,
                predicted_sense_index=continuation_predicted_sense_index,
            )
        )

    return continuations


def _generate_annotated_sentences(
    generator: Generator,
    dataset: Dataset,
    checkpoint_path: Path,
    seed: int | None = None,
) -> list[AnnotatedSentence]:
    """
    Generate annotated sentences.

    Args:
        generator (Generator): Generator.
        dataset (Dataset): Dataset.
        checkpoint_path (Path): Checkpoint path.
        seed (int | None): Random seed.

    Returns:
        list[AnnotatedSentence]: List of annotated sentences.
    """
    from .config import SENTENCE_DEFINITION_SELECTION_PROMPT_TEMPLATE

    definition_selection_options: dict[str, Any] = (
        DEFAULT_DEFINITION_SELECTION_OPTIONS.copy()
    )

    if seed is not None:
        definition_selection_options["seed"] = seed

    from .io import load_annotated_sentences

    annotated_sentences: list[AnnotatedSentence] = []
    instance_ids: set[str] = set()

    if checkpoint_path.exists():
        annotated_sentences = load_annotated_sentences(checkpoint_path)
        instance_ids = {
            annotated_sentence.sentence.instance_id
            for annotated_sentence in annotated_sentences
        }

    for i, sentence in tqdm(enumerate(dataset), desc="Generating", unit="sentence"):
        try:
            if sentence.instance_id in instance_ids:
                continue

            sentence_definition_selection_prompt: str = (
                SENTENCE_DEFINITION_SELECTION_PROMPT_TEMPLATE.format(
                    word=sentence.lemma,
                    definitions="\n".join(
                        f"{i}) {definition}"
                        for i, definition in enumerate(sentence.definitions, start=1)
                    ),
                    sentence=sentence.sentence,
                )
            )

            sentence_predicted_sense_index: int | None = extract_predicted_sense_index(
                generator.generate(
                    sentence_definition_selection_prompt,
                    options=definition_selection_options,
                ),
                len(sentence.definitions),
            )

            continuations: list[AnnotatedContinuation] = (
                _generate_annotated_continuations(
                    generator,
                    sentence,
                    seed,
                )
            )

            annotated_sentences.append(
                AnnotatedSentence(
                    sentence=sentence,
                    predicted_sense_index=sentence_predicted_sense_index,
                    continuations=continuations,
                )
            )

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_annotated_sentences(annotated_sentences, checkpoint_path)
        except Exception:
            save_annotated_sentences(annotated_sentences, checkpoint_path)

    return annotated_sentences


@app.command()
def main(
    model: str = typer.Argument(
        help="Ollama Model",
    ),
    datasets: str = typer.Option(
        "wiktionary",
        help="Datsets ('wiktionary', 'raganato' or custom path)",
    ),
    host: str | None = typer.Option(
        None,
        help="Ollama host",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        help="Output directory",
    ),
) -> None:
    for dataset in datasets.split():
        from .config import CACHE_DIR, DATASETS
        from .utils import download

        dataset_path: Path

        if dataset in DATASETS:
            dataset = dataset.lower()
            dataset_path = CACHE_DIR / f"{dataset}.jsonl.gz"

            if not dataset_path.exists():
                url = DATASETS[dataset]

                try:
                    download(url, dataset_path)
                except Exception as e:
                    typer.echo(f"Failed to download dataset '{dataset}': {e}", err=True)
                    continue
        else:
            dataset_path = Path(dataset)
            dataset = dataset_path.stem

        try:
            sentences: Dataset = Dataset(dataset_path)
        except FileNotFoundError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1)

        from .generators import OllamaGenerator

        try:
            generator: OllamaGenerator = OllamaGenerator(
                model,
                host=host,
            )
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1)

        from .config import CHECKPOINT_FILENAME

        annotated_sentences: list[AnnotatedSentence] = _generate_annotated_sentences(
            generator,
            sentences,
            output_dir / f"{model.replace(':', '-')}" / dataset / CHECKPOINT_FILENAME,
            seed=seed,
        )

        from .config import ANNOTATED_SENTENCES_FILENAME

        save_annotated_sentences(
            annotated_sentences,
            output_dir
            / f"{model.replace(':', '-')}"
            / dataset
            / ANNOTATED_SENTENCES_FILENAME,
        )

        from .evaluator import Evaluator

        evaluator: Evaluator = Evaluator()
        evaluations: dict[str, Any] = evaluator.evaluate(annotated_sentences)

        from .config import EVALUATION_FILENAME
        from .io import save_evaluations

        save_evaluations(
            evaluations,
            output_dir / f"{model.replace(':', '-')}" / dataset / EVALUATION_FILENAME,
        )


if __name__ == "__main__":
    app()
