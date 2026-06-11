from collections import defaultdict
from typing import Any

from tqdm import tqdm

from .models import AnnotatedSentence


class Evaluator:
    """
    Evaluator for annotated sentences.
    """

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
        sentences: int = 0

        correct_sentences: int = 0
        correct_sentences_by_sense: defaultdict[str, list[bool]] = defaultdict(list)

        continuations: defaultdict[str, int] = defaultdict(int)

        correct_continuations: defaultdict[str, int] = defaultdict(int)
        correct_continuations_by_sense: defaultdict[
            str, defaultdict[str, list[bool]]
        ] = defaultdict(lambda: defaultdict(list))

        consistent_sentences = 0
        sentences_with_continuations = 0

        for annotated_sentence in tqdm(
            annotated_sentences,
            desc="Evaluating",
            unit=" annotated sentence",
        ):
            sense_id: str = annotated_sentence.sentence.sense_id

            is_sentence_correct: bool = (
                annotated_sentence.predicted_sense_index
                == annotated_sentence.sentence.sense_index
            )

            sentences += 1

            correct_sentences += int(is_sentence_correct)
            correct_sentences_by_sense[sense_id].append(is_sentence_correct)

            continuation_predictions: list[int] = []

            for annotated_continuation in annotated_sentence.continuations:
                if (
                    not annotated_continuation.is_valid
                    or annotated_continuation.predicted_sense_index is None
                ):
                    continue

                condition: str = annotated_continuation.continuation.condition

                is_continuation_correct: bool = (
                    annotated_continuation.predicted_sense_index
                    == annotated_sentence.sentence.sense_index
                )

                continuations[condition] += 1

                correct_continuations[condition] += int(is_continuation_correct)
                correct_continuations_by_sense[condition][sense_id].append(
                    is_continuation_correct
                )

                continuation_predictions.append(
                    annotated_continuation.predicted_sense_index
                )

            if (
                annotated_sentence.predicted_sense_index is not None
                and continuation_predictions
            ):
                sentences_with_continuations += 1
                consistent_sentences += int(
                    all(
                        prediction == annotated_sentence.predicted_sense_index
                        for prediction in continuation_predictions
                    )
                )

        return {
            "total": sentences,
            "sentence_micro_accuracy": (
                round(
                    correct_sentences / sentences,
                    4,
                )
                if sentences
                else None
            ),
            "sentence_macro_accuracy": (
                round(
                    sum(
                        sum(correct_sentences) / len(correct_sentences)
                        for correct_sentences in correct_sentences_by_sense.values()
                    )
                    / len(correct_sentences_by_sense),
                    4,
                )
                if correct_sentences_by_sense
                else None
            ),
            "consistency_accuracy": (
                round(consistent_sentences / sentences_with_continuations, 4)
                if sentences_with_continuations
                else None
            ),
            "continuations": {
                condition: {
                    "total": continuations[condition],
                    "micro_accuracy": round(
                        correct_continuations[condition] / continuations[condition],
                        4,
                    ),
                    "macro_accuracy": (
                        round(
                            sum(
                                sum(correct_continuations) / len(correct_continuations)
                                for correct_continuations in correct_continuations_by_sense[
                                    condition
                                ].values()
                            )
                            / len(correct_continuations_by_sense[condition]),
                            4,
                        )
                    ),
                }
                for condition in continuations
            },
        }
