from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class POS(Enum):
    """
    Part of speech tags.
    """

    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adj"
    ADVERB = "adv"
    OTHER = "other"

    @classmethod
    def _missing_(
        cls,
        value: object,
    ) -> "POS":
        return cls.OTHER


@dataclass
class Sentence:
    """
    Sentence.
    """

    instance_id: str
    lemma: str
    pos: POS
    definitions: list[str]
    sense_id: str
    sense_index: int
    sentence: str

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            dict[str, Any]: Dictionary.
        """
        return {
            "lemma": self.lemma,
            "pos": self.pos.value,
            "definitions": self.definitions,
            "sense_index": self.sense_index,
            "sentence": self.sentence,
        }


@dataclass
class Continuation:
    """
    Continuation.
    """

    condition: str
    continuation: str

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            dict[str, Any]: Dictionary.
        """
        return {
            "condition": self.condition,
            "continuation": self.continuation,
        }


@dataclass
class AnnotatedContinuation:
    """
    Annotated continuation.
    """

    continuation: Continuation
    is_valid: bool
    predicted_sense_index: int | None = None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            dict[str, Any]: Dictionary.
        """
        return {
            **self.continuation.to_dict(),
            "is_valid": self.is_valid,
            "predicted_sense_index": self.predicted_sense_index,
        }


@dataclass
class AnnotatedSentence:
    """
    Annotated sentence.
    """

    sentence: Sentence
    predicted_sense_index: int | None = None
    continuations: list[AnnotatedContinuation] = field(default_factory=list)

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            dict[str, Any]: Dictionary.
        """
        return {
            **self.sentence.to_dict(),
            "predicted_sense_index": self.predicted_sense_index,
            "continuations": [
                continuation.to_dict() for continuation in self.continuations
            ],
        }
