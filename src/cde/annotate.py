import re
from difflib import SequenceMatcher
from pathlib import Path
from string import Formatter
from typing import Any

import typer
from tqdm import tqdm

from .config import (
    CONTINUATION_GENERATION_CONFIGURATIONS,
    DEFAULT_DEFINITION_SELECTION_OPTIONS,
    DEFINITION_SELECTION_PROMPT_TEMPLATES,
)
from .generators import OllamaGenerator
from .generators.base import Generator
from .io import Dataset, load_annotated_sentences, save_annotated_sentences
from .models import (
    AnnotatedContinuation,
    AnnotatedSentence,
    Continuation,
    Language,
    Sentence,
)


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
    chunk: list[Sentence],
    definition_selection_options: dict[str, Any],
    language: Language = Language.ENGLISH,
    seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Generate annotated continuations for a chunk of sentences.

    Args:
        generator (Generator): Generator.
        sentence (Sentence): Sentence.
        definition_selection_options (dict[str, Any]): Options for definition selection.
        language (Language): Language.
        seed (int | None): Random seed.

    Returns:
        dict[str, dict[str, Any]]: Dictionary mapping sentence instance IDs to their annotated continuations.
    """
    continuations: dict[str, dict[str, Any]] = {
        sentence.instance_id: {} for sentence in chunk
    }

    for (
        condition,
        configuration,
    ) in CONTINUATION_GENERATION_CONFIGURATIONS.items():
        continuation_prompt_template_fields: set[str] = {
            field
            for _, field, _, _ in Formatter().parse(configuration["prompts"][language])
            if field
        }

        continuation_generation_prompts: list[str] = []

        for sentence in chunk:
            prompt: str

            if "word" in continuation_prompt_template_fields:
                prompt = configuration["prompts"][language].format(
                    word=sentence.sentence[
                        sentence.word_offset[0] : sentence.word_offset[1]
                    ],
                    sentence=sentence.sentence,
                )
            else:
                prompt = configuration["prompts"][language].format(
                    sentence=sentence.sentence
                )

            continuation_generation_prompts.append(prompt)

        continuation_generation_options = configuration["options"].copy()

        if seed is not None:
            continuation_generation_options["seed"] = seed

        continuation_generation_responses: list[str] = generator.generate(
            continuation_generation_prompts,
            options=continuation_generation_options,
        )

        for sentence, response in zip(chunk, continuation_generation_responses):
            continuations[sentence.instance_id][condition] = {
                "continuation": response,
                "is_valid": _validate_continuation(
                    sentence.sentence,
                    response,
                ),
                "predicted_sense_index": None,
            }

    continuation_definition_selection_prompts: list[str] = []
    continuation_definition_selection_prompt_metadata: list[tuple[str, str, int]] = []

    for sentence in chunk:
        for condition in CONTINUATION_GENERATION_CONFIGURATIONS.keys():
            continuation: dict[str, Any] = continuations[sentence.instance_id][
                condition
            ]

            if continuation["is_valid"]:
                prompt: str = DEFINITION_SELECTION_PROMPT_TEMPLATES[language][
                    "continuation"
                ].format(
                    word=sentence.lemma,
                    definitions="\n".join(
                        f"{i}) {definition}"
                        for i, definition in enumerate(sentence.definitions, start=1)
                    ),
                    sentence=sentence.sentence,
                    continuation=continuation["continuation"],
                )

                continuation_definition_selection_prompts.append(prompt)
                continuation_definition_selection_prompt_metadata.append(
                    (
                        sentence.instance_id,
                        condition,
                        len(sentence.definitions),
                    )
                )

    if continuation_definition_selection_prompts:
        continuation_definition_selection_responses = generator.generate(
            continuation_definition_selection_prompts,
            options=definition_selection_options,
        )

        for response, (instance_id, condition, definitions) in zip(
            continuation_definition_selection_responses,
            continuation_definition_selection_prompt_metadata,
        ):
            continuations[instance_id][condition]["predicted_sense_index"] = (
                _extract_predicted_sense_index(response, definitions)
            )

    return continuations


def generate_annotated_sentences(
    generator: Generator,
    dataset: Dataset,
    checkpoint_path: Path,
    chunk_size: int | None = None,
    think: bool = False,
    seed: int | None = None,
) -> list[AnnotatedSentence]:
    """
    Generate annotated sentences.

    Args:
        generator (Generator): Generator.
        dataset (Dataset): Dataset.
        checkpoint_path (Path): Checkpoint path.
        chunk_size (int | None): Chunk size. If None, the entire dataset will be processed at once.
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

    sentences: list[Sentence] = [
        sentence for sentence in dataset if sentence.instance_id not in instance_ids
    ]

    if not sentences:
        typer.echo("All sentences have already been annotated. Skipping generation.")
        return annotated_sentences

    if chunk_size is None:
        chunk_size = len(sentences)

    with tqdm(total=len(sentences), desc="Generating", unit="sentence") as pbar:
        for chunk_idx in range(0, len(sentences), chunk_size):
            chunk: list[Sentence] = sentences[chunk_idx : chunk_idx + chunk_size]

            try:
                sentence_definition_selection_prompts: list[str] = []

                for sentence in chunk:
                    prompt: str = DEFINITION_SELECTION_PROMPT_TEMPLATES[
                        dataset.language
                    ]["sentence"].format(
                        word=sentence.lemma,
                        definitions="\n".join(
                            f"{i}) {definition}"
                            for i, definition in enumerate(
                                sentence.definitions, start=1
                            )
                        ),
                        sentence=sentence.sentence,
                    )

                    sentence_definition_selection_prompts.append(prompt)

                sentence_definition_selection_responses: list[str] = generator.generate(
                    sentence_definition_selection_prompts,
                    options=definition_selection_options,
                )

                sentence_predicted_sense_indices: list[int | None] = [
                    _extract_predicted_sense_index(response, len(sentence.definitions))
                    for response, sentence in zip(
                        sentence_definition_selection_responses, chunk
                    )
                ]

                continuations: dict[str, dict[str, Any]] = (
                    _generate_annotated_continuations(
                        generator,
                        chunk,
                        definition_selection_options,
                        language=dataset.language,
                        seed=seed,
                    )
                )

                for i, sentence in enumerate(chunk):
                    annotated_continuations: list[AnnotatedContinuation] = []

                    for condition in CONTINUATION_GENERATION_CONFIGURATIONS.keys():
                        continuation: dict[str, Any] = continuations[
                            sentence.instance_id
                        ][condition]

                        annotated_continuations.append(
                            AnnotatedContinuation(
                                Continuation(
                                    condition=condition,
                                    continuation=continuation["continuation"],
                                ),
                                is_valid=continuation["is_valid"],
                                predicted_sense_index=continuation[
                                    "predicted_sense_index"
                                ],
                            )
                        )

                    annotated_sentences.append(
                        AnnotatedSentence(
                            sentence=sentence,
                            predicted_sense_index=sentence_predicted_sense_indices[i],
                            continuations=annotated_continuations,
                        )
                    )

                save_annotated_sentences(annotated_sentences, checkpoint_path)
                pbar.update(len(chunk))
            except Exception as e:
                save_annotated_sentences(annotated_sentences, checkpoint_path)
                raise e

    return annotated_sentences
