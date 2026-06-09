from typing import Any

from tqdm import tqdm

from .models import AnnotatedSentence


class Evaluator:
    """
    Evaluator for annotated sentences.
    """

    @staticmethod
    def _increment(
        dictionary: dict[Any, Any],
        *keys: Any,
    ) -> None:
        """
        Increment a nested dictionary value by 1, creating nested dictionaries as needed.

        Args:
            dictionary (dict[Any, Any]): Dictionary.
            *keys (Any): Keys.
        """
        nested_dictionary: dict[Any, Any] = dictionary

        for key in keys[:-1]:
            nested_dictionary = nested_dictionary.setdefault(key, {})

        nested_dictionary[keys[-1]] = nested_dictionary.get(keys[-1], 0) + 1

    def _evaluate_continuations(
        self,
        annotated_sentence: AnnotatedSentence,
        sentence_features: dict[str, str | int],
        evaluations: dict[str, Any],
    ) -> list[int]:
        """
        Evaluate the continuations for an annotated sentence.

        Args:
            annotated_sentence (AnnotatedSentence): Annotated sentence.
            sentence_features (dict[str, str | int]): Sentence features.
            evaluations (dict[str, Any]): Evaluations.

        Returns:
            list[int]: List of predicted sense indices for valid continuations.
        """
        continuation_predictions: list[int] = []

        for annotated_continuation in annotated_sentence.continuations:
            continuation_features: dict[str, str | int] = {
                "conditions": annotated_continuation.continuation.condition,
                "continuation_lengths": len(
                    annotated_continuation.continuation.continuation
                ),
            }

            continuation_validity: str = (
                "valid" if annotated_continuation.is_valid else "invalid"
            )

            outcomes: list[str] = [continuation_validity]

            if (
                continuation_validity == "valid"
                and annotated_continuation.predicted_sense_index is not None
            ):
                continuation_predictions.append(
                    annotated_continuation.predicted_sense_index
                )

                continuation_correctness = (
                    "correct"
                    if annotated_continuation.predicted_sense_index
                    == annotated_sentence.sentence.sense_index
                    else "incorrect"
                )

                outcomes.append(continuation_correctness)

            for outcome in outcomes:
                self._increment(evaluations, "continuations", outcome)

                for name, feature in sentence_features.items():
                    self._increment(
                        evaluations,
                        name,
                        feature,
                        "continuations",
                        outcome,
                    )

                for name, feature in continuation_features.items():
                    self._increment(
                        evaluations,
                        "continuations",
                        name,
                        feature,
                        outcome,
                    )

        return continuation_predictions

    def _evaluate_consistency(
        self,
        annotated_sentence: AnnotatedSentence,
        continuation_predictions: list[int],
        sentence_features: dict[str, str | int],
        evaluations: dict[str, Any],
    ) -> None:
        """
        Evaluate the consistency of the predicted sense index across the sentence and its valid continuations.

        Args:
            annotated_sentence (AnnotatedSentence): Annotated sentence.
            continuation_predictions (list[int]): List of predicted sense indices for valid continuations.
            sentence_features (dict[str, str | int]): Sentence features.
            evaluations (dict[str, Any]): Evaluations.
        """
        if (
            annotated_sentence.predicted_sense_index is not None
            and continuation_predictions
        ):
            sentence_consistency: str = (
                "consistent"
                if all(
                    continuation_prediction == annotated_sentence.predicted_sense_index
                    for continuation_prediction in continuation_predictions
                )
                else "inconsistent"
            )

            self._increment(evaluations, sentence_consistency)

            for name, feature in sentence_features.items():
                self._increment(
                    evaluations,
                    name,
                    feature,
                    sentence_consistency,
                )

            continuation_consistency: str = (
                "consistent"
                if len(set(continuation_predictions)) == 1
                else "inconsistent"
            )

            self._increment(evaluations, "continuations", continuation_consistency)

            for name, feature in sentence_features.items():
                self._increment(
                    evaluations,
                    name,
                    feature,
                    "continuations",
                    continuation_consistency,
                )

    def _compute_accuracies(
        self,
        evaluations: dict[str, Any] | Any,
    ) -> None:
        """
        Compute accuracies for the evaluations.

        Args:
            evaluations (dict[str, Any] | Any): Evaluations.
        """
        if not isinstance(evaluations, dict):
            return

        correct_sentences = evaluations.get("correct_sentences", 0)
        incorrect_sentences = evaluations.get("incorrect_sentences", 0)

        if (sentences := correct_sentences + incorrect_sentences) > 0:
            evaluations["sentence_accuracy"] = correct_sentences / sentences

        valid_continuations = evaluations.get("valid", 0)
        invalid_continuations = evaluations.get("invalid", 0)

        if (continuations := valid_continuations + invalid_continuations) > 0:
            evaluations["validity_rate"] = valid_continuations / continuations

        correct_continuations = evaluations.get("correct", 0)
        incorrect_continuations = evaluations.get("incorrect", 0)

        if (continuations := correct_continuations + incorrect_continuations) > 0:
            evaluations["accuracy"] = correct_continuations / continuations

        consistent = evaluations.get("consistent", 0)
        inconsistent = evaluations.get("inconsistent", 0)

        if (consistency := consistent + inconsistent) > 0:
            evaluations["consistency_accuracy"] = consistent / consistency

        for evaluation in evaluations.values():
            if isinstance(evaluation, dict):
                self._compute_accuracies(evaluation)

    def evaluate(
        self,
        annotated_sentences: list[AnnotatedSentence],
    ) -> dict[str, Any]:
        """
        Evaluate the annotated sentences.

        Args:
            annotated_sentences (list[AnnotatedSentence]): List of annotated sentences.

        Returns:
            dict[str, Any]: Evaluations.
        """
        evaluations: dict[str, Any] = {}

        for annotated_sentence in tqdm(
            annotated_sentences,
            desc="Evaluating",
            unit=" annotated sentence",
        ):
            sentence_features: dict[str, str | int] = {
                "pos": annotated_sentence.sentence.pos.value,
                "definition_lengths": len(annotated_sentence.sentence.definitions),
                "sentence_lengths": len(annotated_sentence.sentence.sentence),
            }

            sentence_correctness: str = (
                "correct_sentences"
                if annotated_sentence.predicted_sense_index
                == annotated_sentence.sentence.sense_index
                else "incorrect_sentences"
            )

            self._increment(evaluations, sentence_correctness)

            for name, feature in sentence_features.items():
                self._increment(evaluations, name, feature, sentence_correctness)

            continuation_predictions: list[int] = self._evaluate_continuations(
                annotated_sentence,
                sentence_features,
                evaluations,
            )

            self._evaluate_consistency(
                annotated_sentence,
                continuation_predictions,
                sentence_features,
                evaluations,
            )

        self._compute_accuracies(evaluations)
        return evaluations
