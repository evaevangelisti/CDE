from collections import defaultdict
from typing import Any

from tqdm import tqdm

from .models import AnnotatedSentence


class Evaluator:
    """
    Evaluator for annotated sentences.
    """

    _SETTINGS: list[str] = ["zero_shot", "few_shots"]

    def evaluate(
        self,
        annotated_sentences: list[AnnotatedSentence],
    ) -> dict[str, Any]:
        """
        Evaluate the annotated sentences.

        Args:
            annotated_sentences: List of annotated sentences.

        Returns:
            dict[str, Any]: Evaluations.
        """
        sentences = 0

        correct_sentences: dict[str, int] = {setting: 0 for setting in self._SETTINGS}

        correct_sentences_by_sense: dict[str, defaultdict[str, list[bool]]] = {
            setting: defaultdict(list) for setting in self._SETTINGS
        }

        continuations: dict[str, defaultdict[str, int]] = {
            setting: defaultdict(int) for setting in self._SETTINGS
        }

        correct_continuations: dict[str, defaultdict[str, int]] = {
            setting: defaultdict(int) for setting in self._SETTINGS
        }

        correct_continuations_by_sense: dict[
            str, defaultdict[str, defaultdict[str, list[bool]]]
        ] = {
            setting: defaultdict(lambda: defaultdict(list))
            for setting in self._SETTINGS
        }

        consistent_sentences: dict[str, int] = {
            setting: 0 for setting in self._SETTINGS
        }

        sentences_with_continuations: dict[str, int] = {
            setting: 0 for setting in self._SETTINGS
        }

        for annotated_sentence in tqdm(
            annotated_sentences,
            desc="Evaluating",
            unit=" annotated sentence",
        ):
            sense_id: str = annotated_sentence.sentence.sense_id
            sentences += 1

            for setting in ["zero_shot", "few_shots"]:
                sentence_predicted_sense_index: int | None = getattr(
                    annotated_sentence.predicted_sense_indices,
                    f"{setting}_predicted_sense_index",
                )

                is_sentence_correct: bool = (
                    sentence_predicted_sense_index
                    == annotated_sentence.sentence.sense_index
                )

                correct_sentences[setting] += int(is_sentence_correct)
                correct_sentences_by_sense[setting][sense_id].append(
                    is_sentence_correct
                )

                continuation_predictions: list[int] = []

                for annotated_continuation in annotated_sentence.continuations:
                    continuation_predicted_sense_index: int | None = getattr(
                        annotated_continuation.predicted_sense_indices,
                        f"{setting}_predicted_sense_index",
                    )

                    if (
                        not annotated_continuation.is_valid
                        or continuation_predicted_sense_index is None
                    ):
                        continue

                    condition: str = annotated_continuation.continuation.condition

                    is_continuation_correct: bool = (
                        continuation_predicted_sense_index
                        == annotated_sentence.sentence.sense_index
                    )

                    continuations[setting][condition] += 1

                    correct_continuations[setting][condition] += int(
                        is_continuation_correct
                    )

                    correct_continuations_by_sense[setting][condition][sense_id].append(
                        is_continuation_correct
                    )

                    continuation_predictions.append(continuation_predicted_sense_index)

                if (
                    sentence_predicted_sense_index is not None
                    and continuation_predictions
                ):
                    sentences_with_continuations[setting] += 1
                    consistent_sentences[setting] += int(
                        all(
                            prediction == sentence_predicted_sense_index
                            for prediction in continuation_predictions
                        )
                    )

        evaluations: dict[str, Any] = {"total": sentences}

        for setting in ["zero_shot", "few_shots"]:
            evaluations[f"{setting}_sentence_micro_accuracy"] = (
                round(
                    correct_sentences[setting] / sentences,
                    4,
                )
                if sentences
                else None
            )

            evaluations[f"{setting}_sentence_macro_accuracy"] = (
                round(
                    sum(
                        sum(correct_sentences) / len(correct_sentences)
                        for correct_sentences in correct_sentences_by_sense[
                            setting
                        ].values()
                    )
                    / len(correct_sentences_by_sense[setting]),
                    4,
                )
                if correct_sentences_by_sense[setting]
                else None
            )

            evaluations[f"{setting}_consistency_accuracy"] = (
                round(
                    consistent_sentences[setting]
                    / sentences_with_continuations[setting],
                    4,
                )
                if sentences_with_continuations[setting]
                else None
            )

            evaluations[f"{setting}_continuations"] = {
                condition: {
                    "total": continuations[setting][condition],
                    "micro_accuracy": round(
                        correct_continuations[setting][condition]
                        / continuations[setting][condition],
                        4,
                    ),
                    "macro_accuracy": (
                        round(
                            sum(
                                sum(correct_continuations) / len(correct_continuations)
                                for correct_continuations in correct_continuations_by_sense[
                                    setting
                                ][
                                    condition
                                ].values()
                            )
                            / len(correct_continuations_by_sense[setting][condition]),
                            4,
                        )
                    ),
                }
                for condition in continuations[setting]
            }

        return evaluations
