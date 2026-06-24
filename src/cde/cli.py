import re
from difflib import SequenceMatcher
from pathlib import Path
from string import Formatter
from typing import Any

import typer
from tqdm import tqdm

from .config import (
    ANNOTATED_SENTENCES_FILENAME,
    CHECKPOINT_FILENAME,
    CHECKPOINT_INTERVAL,
    CONTINUATION_GENERATION_CONFIGURATIONS,
    DEFAULT_DEFINITION_SELECTION_OPTIONS,
    DEFINITION_SELECTION_PROMPT_TEMPLATES,
    EVALUATION_FILENAME,
)
from .evaluator import Evaluator
from .generators import OllamaGenerator
from .generators.base import Generator
from .io import (
    Dataset,
    load_annotated_sentences,
    save_annotated_sentences,
    save_evaluations,
)
from .models import (
    AnnotatedContinuation,
    AnnotatedSentence,
    Continuation,
    Language,
    Sentence,
)

app: typer.Typer = typer.Typer(add_completion=False)


def _extract_predicted_sense_index(
    response: str,
    definitions: int,
) -> int | None:
    """
    Extract the predicted sense index from the response.

    Args:
        response (str): Response.
        definitions (int): Number of definitions.

    Returns:
        int | None: Predicted sense index (0-based) or None if not found or out of range.
    """
    match: re.Match[str] | None = re.search(r"\d+", response)
    if not match:
        return None

    predicted_sense_index: int = int(match.group()) - 1

    if 0 <= predicted_sense_index < definitions:
        return predicted_sense_index

    return None


def _validate_continuation(
    sentence: str,
    continuation: str,
) -> bool:
    """
    Validate the continuation.

    Args:
        sentence (str): Original sentence.
        continuation (str): Continuation.

    Returns:
        bool: True if the continuation is valid, False otherwise.
    """
    sentence = sentence.strip().lower()
    continuation = continuation.strip().lower()

    if not continuation or len(continuation) <= len(sentence):
        return False

    if sentence in continuation:
        return True

    for start in range(len(continuation) - len(sentence) + 1):
        ratio = SequenceMatcher(
            None,
            sentence,
            continuation[start : start + len(sentence)],
        ).ratio()

        if ratio > 0.90:
            return True

    return False


def _generate_annotated_continuations(
    generator: Generator,
    sentence: Sentence,
    language: Language = Language.ENGLISH,
    think: bool = False,
    seed: int | None = None,
) -> list[AnnotatedContinuation]:
    """
    Generate annotated continuations for a sentence.

    Args:
        generator (Generator): Generator.
        sentence (Sentence): Sentence.
        language (Language): Language.
        think (bool): Whether to use CoT prompting.
        seed (int | None): Random seed.

    Returns:
        list[AnnotatedContinuation]: List of annotated continuations.
    """
    definition_selection_options: dict[str, Any] = (
        DEFAULT_DEFINITION_SELECTION_OPTIONS.copy()
    )

    if isinstance(generator, OllamaGenerator):
        definition_selection_options["think"] = think

    if seed is not None:
        definition_selection_options["seed"] = seed

    continuations: list[AnnotatedContinuation] = []

    for (
        condition,
        continuation_generation_configuration,
    ) in CONTINUATION_GENERATION_CONFIGURATIONS.items():
        continuation_prompt_template_fields: set[str] = {
            field
            for _, field, _, _ in Formatter().parse(
                continuation_generation_configuration["prompts"][language]
            )
            if field
        }

        continuation_generation_prompt: str

        if "word" in continuation_prompt_template_fields:
            continuation_generation_prompt = continuation_generation_configuration[
                "prompts"
            ][language].format(
                word=sentence.sentence[
                    sentence.word_offset[0] : sentence.word_offset[1]
                ],
                sentence=sentence.sentence,
            )
        else:
            continuation_generation_prompt = continuation_generation_configuration[
                "prompts"
            ][language].format(
                sentence=sentence.sentence,
            )

        continuation_generation_options: dict[str, Any] = (
            continuation_generation_configuration["options"]
        )

        if seed is not None:
            continuation_generation_options["seed"] = seed

        continuation: str = generator.generate(
            continuation_generation_prompt,
            options=continuation_generation_options,
        )

        is_continuation_valid: bool = _validate_continuation(
            sentence.sentence,
            continuation,
        )

        continuation_predicted_sense_index: int | None = None

        if is_continuation_valid:
            continuation_definition_selection_prompt: (
                str
            ) = DEFINITION_SELECTION_PROMPT_TEMPLATES[language]["continuation"].format(
                word=sentence.lemma,
                definitions="\n".join(
                    f"{i}) {definition}"
                    for i, definition in enumerate(sentence.definitions, start=1)
                ),
                sentence=sentence.sentence,
                continuation=continuation,
            )

            continuation_predicted_sense_index = _extract_predicted_sense_index(
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
    think: bool = False,
    seed: int | None = None,
) -> list[AnnotatedSentence]:
    """
    Generate annotated sentences.

    Args:
        generator (Generator): Generator.
        dataset (Dataset): Dataset.
        checkpoint_path (Path): Checkpoint path.
        think (bool): Whether to use CoT prompting.
        seed (int | None): Random seed.

    Returns:
        list[AnnotatedSentence]: List of annotated sentences.
    """
    definition_selection_options: dict[str, Any] = (
        DEFAULT_DEFINITION_SELECTION_OPTIONS.copy()
    )

    if isinstance(generator, OllamaGenerator):
        definition_selection_options["think"] = think

    if seed is not None:
        definition_selection_options["seed"] = seed

    annotated_sentences: list[AnnotatedSentence] = []
    instance_ids: set[str] = set()

    if checkpoint_path.exists():
        annotated_sentences = load_annotated_sentences(checkpoint_path)
        instance_ids = {
            annotated_sentence.sentence.instance_id
            for annotated_sentence in annotated_sentences
        }

        typer.echo(
            f"Resuming from checkpoint '{checkpoint_path}' with {len(annotated_sentences)} annotated sentences"
        )

    for i, sentence in tqdm(
        enumerate(dataset),
        desc="Generating",
        total=len(dataset),
        unit="sentence",
    ):
        if sentence.instance_id in instance_ids:
            continue

        try:
            sentence_definition_selection_prompt: (
                str
            ) = DEFINITION_SELECTION_PROMPT_TEMPLATES[dataset.language][
                "sentence"
            ].format(
                word=sentence.lemma,
                definitions="\n".join(
                    f"{i}) {definition}"
                    for i, definition in enumerate(sentence.definitions, start=1)
                ),
                sentence=sentence.sentence,
            )

            sentence_predicted_sense_index: int | None = _extract_predicted_sense_index(
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
                    language=dataset.language,
                    think=think,
                    seed=seed,
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
    datasets: str = typer.Argument(
        help="Datasets",
    ),
    host: str | None = typer.Option(
        None,
        help="Ollama host",
    ),
    think: bool = typer.Option(
        False,
        help="Whether to use CoT prompting",
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
    for dataset in datasets.split():
        dataset_path: Path = Path(dataset)

        if dataset_path.suffix != ".jsonl":
            typer.echo(f"Dataset '{dataset_path}' is not a JSONL file", err=True)
            continue

        output_dir = output_dir / f"{model.replace(':', '-')}" / dataset_path.stem

        try:
            sentences: Dataset = Dataset(dataset_path)
        except FileNotFoundError as e:
            typer.echo(str(e), err=True)
            continue

        if not sentences:
            typer.echo(f"Dataset '{dataset_path.stem}' is empty", err=True)
            continue

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
            sentences,
            output_dir / CHECKPOINT_FILENAME,
            think=think,
            seed=seed,
        )

        save_annotated_sentences(
            annotated_sentences,
            output_dir / ANNOTATED_SENTENCES_FILENAME,
        )

        evaluator: Evaluator = Evaluator()
        evaluations: dict[str, Any] = evaluator.evaluate(annotated_sentences)

        save_evaluations(
            evaluations,
            output_dir / EVALUATION_FILENAME,
        )


if __name__ == "__main__":
    app()
