from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from .config import (
    ANNOTATED_SENTENCES_FILENAME,
    CACHE_DIR,
    CHECKPOINT_FILENAME,
    CHECKPOINT_INTERVAL,
    CONTINUATION_DEFINITION_SELECTION_PROMPT_TEMPLATE,
    CONTINUATION_GENERATION_CONFIGURATIONS,
    DATASETS,
    DEFAULT_DEFINITION_SELECTION_OPTIONS,
    EVALUATION_FILENAME,
    SENTENCE_DEFINITION_SELECTION_PROMPT_TEMPLATE,
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
from .models import AnnotatedContinuation, AnnotatedSentence, Continuation, Sentence
from .utils import (
    download,
    extract_predicted_sense_index,
    retrieve_template_fields,
    validate_continuation,
)

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
    definition_selection_options: dict[str, Any] = (
        DEFAULT_DEFINITION_SELECTION_OPTIONS.copy()
    )

    if seed is not None:
        definition_selection_options["seed"] = seed

    continuations: list[AnnotatedContinuation] = []

    for (
        condition,
        continuation_generation_configuration,
    ) in CONTINUATION_GENERATION_CONFIGURATIONS.items():
        continuation_prompt_template_fields: set[str] = retrieve_template_fields(
            continuation_generation_configuration["prompt"],
        )

        continuation_generation_prompt: str

        if "word" in continuation_prompt_template_fields:
            continuation_generation_prompt = continuation_generation_configuration[
                "prompt"
            ].format(
                word=sentence.sentence[
                    sentence.word_offset[0] : sentence.word_offset[1]
                ],
                sentence=sentence.sentence,
            )
        else:
            continuation_generation_prompt = continuation_generation_configuration[
                "prompt"
            ].format(
                sentence=sentence.sentence,
            )

        options: dict[str, Any] = continuation_generation_configuration["options"]

        if seed is not None:
            options["seed"] = seed

        continuation: str = generator.generate(
            continuation_generation_prompt,
            options=options,
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
    definition_selection_options: dict[str, Any] = (
        DEFAULT_DEFINITION_SELECTION_OPTIONS.copy()
    )

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

    for i, sentence in tqdm(
        enumerate(dataset),
        desc="Generating",
        total=len(dataset),
        unit="sentence",
    ):
        if sentence.instance_id in instance_ids:
            continue

        try:
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
        help="Datsets ('wiktionary', 'wordnet', 'raganato' or custom path)",
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
            output_dir / f"{model.replace(':', '-')}" / dataset / CHECKPOINT_FILENAME,
            seed=seed,
        )

        save_annotated_sentences(
            annotated_sentences,
            output_dir
            / f"{model.replace(':', '-')}"
            / dataset
            / ANNOTATED_SENTENCES_FILENAME,
        )

        evaluator: Evaluator = Evaluator()
        evaluations: dict[str, Any] = evaluator.evaluate(annotated_sentences)

        save_evaluations(
            evaluations,
            output_dir / f"{model.replace(':', '-')}" / dataset / EVALUATION_FILENAME,
        )


if __name__ == "__main__":
    app()
