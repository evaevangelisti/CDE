import json
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from .models import POS, AnnotatedSentence, Sentence


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

        self._sentences: list[Sentence] = []
        self._load()

    def _parse_sentence(
        self,
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
                lemma=record["lemma"],
                pos=POS(record.get("pos")),
                definitions=record["definitions"],
                sense_index=record["sense_index"],
                sentence=record["sentence"],
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

        lines: int = sum(1 for _ in open(self._path))

        with self._path.open(encoding="utf-8") as file:
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
            file.write(json.dumps(annotated_sentence.to_dict()) + "\n")


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
    path.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(evaluations, file)
