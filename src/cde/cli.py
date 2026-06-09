from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from .config import DEFAULT_DEFINITION_SELECTION_OPTIONS
from .generators.base import Generator
from .io import Dataset
from .models import AnnotatedContinuation, AnnotatedSentence, Sentence
from .utils import extract_predicted_sense_index

app: typer.Typer = typer.Typer(
    help="Continuation-based Disambiguation Evaluator",
    add_completion=False,
)


def _generate_annotated_continuations(
    generator: Generator, sentence: Sentence, seed: int | None = None
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
    seed: int | None = None,
) -> list[AnnotatedSentence]:
    """
    Generate annotated sentences.

    Args:
        generator (Generator): Generator.
        dataset (Dataset): Dataset.
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

    annotated_sentences: list[AnnotatedSentence] = []

    for sentence in tqdm(dataset, desc="Generating", unit=" sentence"):
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

        continuations: list[AnnotatedContinuation] = _generate_annotated_continuations(
            generator,
            sentence,
            seed,
        )

        annotated_sentences.append(
            AnnotatedSentence(
                sentence=sentence,
                predicted_sense_index=sentence_predicted_sense_index,
                continuations=continuations,
            )
        )

    return annotated_sentences


@app.command()
def main(
    dataset_path: str = typer.Argument(
        help="Dataset path",
    ),
    model: str = typer.Argument(
        help="Ollama Model",
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
    try:
        dataset: Dataset = Dataset(dataset_path)
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

    annotated_sentences: list[AnnotatedSentence] = _generate_annotated_sentences(
        generator,
        dataset,
        seed=seed,
    )

    from .config import ANNOTATED_SENTENCES_PATH
    from .io import save_annotated_sentences

    save_annotated_sentences(
        annotated_sentences,
        output_dir / ANNOTATED_SENTENCES_PATH,
    )

    from .evaluator import Evaluator

    evaluator: Evaluator = Evaluator()
    evaluations: dict[str, Any] = evaluator.evaluate(annotated_sentences)

    from .config import EVALUATION_PATH
    from .io import save_evaluations

    save_evaluations(
        evaluations,
        output_dir / EVALUATION_PATH,
    )


if __name__ == "__main__":
    app()
