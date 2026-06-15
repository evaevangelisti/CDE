import gzip
import json
from pathlib import Path
from typing import Any, Iterator, TextIO

from tqdm import tqdm

from .models import (
    POS,
    AnnotatedContinuation,
    AnnotatedSentence,
    Continuation,
    Language,
    Sentence,
)


class Dataset:
    """
    Dataset for word sense disambiguation.
    """

    def __init__(
        self,
        path: str | Path,
    ):
        """
        Initialize the dataset.

        Args:
            path (str | Path): Path to the dataset file.
        """
        self._path: Path = Path(path)

        self._language: Language = Language.ENGLISH
        self._sentences: list[Sentence] = []

        self._load()

    @property
    def language(
        self,
    ) -> Language:
        """
        Get the language of the dataset.

        Returns:
            Language: Language of the dataset.
        """
        return self._language

    def _open(
        self,
    ) -> TextIO:
        """
        Open the dataset file.

        Returns:
            TextIO: Opened file object.
        """
        if self._path.suffix == ".gz":
            return gzip.open(self._path, "rt", encoding="utf-8")

        return self._path.open(encoding="utf-8")

    @staticmethod
    def _parse_sentence(
        record: dict[str, Any],
    ) -> Sentence:
        """
        Parse a sentence record.

        Args:
            record (dict[str, Any]): Sentence record.

        Returns:
            Sentence: Parsed sentence.

        Raises:
            ValueError: If the record format is invalid.
        """
        try:
            return Sentence(
                instance_id=record["instance_id"],
                lemma=record["lemma"],
                pos=POS(record.get("pos")),
                definitions=record["definitions"],
                sense_id=record["sense_id"],
                sense_index=record["sense_index"],
                sentence=record["sentence"],
                word_offset=record["word_offset"],
            )
        except KeyError:
            raise ValueError("Invalid record format")

    def _load(
        self,
    ) -> None:
        """
        Load the dataset from the file.

        Raises:
            FileNotFoundError: If the dataset file is not found.
        """
        if not self._path.exists():
            raise FileNotFoundError("Dataset file not found")

        with self._open() as file:
            lines: int = sum(1 for _ in file)

        with self._open() as file:
            line: str = file.readline()

            if not line.strip():
                return

            try:
                record: dict[str, Any] = json.loads(line)

                if "__metadata__" in record:
                    self._language = Language(record["__metadata__"]["language"])
                else:
                    self._sentences.append(self._parse_sentence(record))
            except (json.JSONDecodeError, ValueError):
                pass

            for line in tqdm(file, desc="Loading Dataset", total=lines, unit=" line"):
                if line.strip():
                    try:
                        record: dict[str, Any] = json.loads(line)
                        self._sentences.append(self._parse_sentence(record))
                    except (json.JSONDecodeError, ValueError):
                        continue

    def __len__(
        self,
    ) -> int:
        """
        Get the number of sentences in the dataset.

        Returns:
            int: Number of sentences.
        """
        return len(self._sentences)

    def __getitem__(
        self,
        index: int,
    ) -> Sentence:
        """
        Get a sentence by index.

        Args:
            index (int): Sentence index.

        Returns:
            Sentence: Sentence at the specified index.
        """
        return self._sentences[index]

    def __iter__(
        self,
    ) -> Iterator[Sentence]:
        """
        Iterate over the sentences in the dataset.

        Returns:
            Iterator[Sentence]: An iterator over the sentences.
        """
        return iter(self._sentences)


def _parse_annotated_sentence(
    record: dict[str, Any],
) -> AnnotatedSentence:
    """
    Parse an annotated sentence.

    Args:
        record (dict[str, Any]): Annotated sentence.

    Returns:
        AnnotatedSentence: Parsed annotated sentence.

    Raises:
        ValueError: If the record format is invalid.
    """
    try:
        return AnnotatedSentence(
            sentence=Dataset._parse_sentence(record),
            predicted_sense_index=record["predicted_sense_index"],
            continuations=[
                AnnotatedContinuation(
                    Continuation(
                        condition=continuation["condition"],
                        continuation=continuation["continuation"],
                    ),
                    is_valid=continuation["is_valid"],
                    predicted_sense_index=continuation["predicted_sense_index"],
                )
                for continuation in record["continuations"]
            ],
        )
    except KeyError:
        raise ValueError("Invalid record format")


def load_annotated_sentences(
    path: Path,
) -> list[AnnotatedSentence]:
    """
    Load annotated sentences.

    Args:
        path (Path): Path to the annotated sentences file.

    Returns:
        list[AnnotatedSentence]: List of annotated sentences.
    """
    annotated_sentences: list[AnnotatedSentence] = []

    if not path.exists():
        return annotated_sentences

    lines: int = sum(1 for _ in path.open())

    with path.open(encoding="utf-8") as file:
        for line in tqdm(
            file,
            desc="Loading checkpoint",
            total=lines,
            unit=" annotated sentence",
        ):
            if line.strip():
                try:
                    record: dict[str, Any] = json.loads(line)
                    annotated_sentences.append(_parse_annotated_sentence(record))
                except (json.JSONDecodeError, ValueError):
                    continue

    return annotated_sentences


def save_annotated_sentences(
    annotated_sentences: list[AnnotatedSentence],
    path: Path,
) -> None:
    """
    Save annotated sentences.

    Args:
        annotated_sentences (list[AnnotatedSentence]): List of annotated sentences.
        path (Path): Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        for annotated_sentence in annotated_sentences:
            file.write(
                json.dumps(
                    annotated_sentence.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_evaluations(
    evaluations: dict[str, Any],
    path: Path,
) -> None:
    """
    Save evaluations.

    Args:
        evaluations (dict[str, Any]): Evaluations.
        path (Path): Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(evaluations, file, ensure_ascii=False, indent=4)
