import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language as SpacyLanguage
from spacy.tokens import Doc, Token

from cde.models import Language

from ..config import SPACY_MODELS


class Builder(ABC):
    def __init__(
        self,
        language: Language = Language.ENGLISH,
    ):
        self._nlp: SpacyLanguage = spacy.load(
            SPACY_MODELS[language], disable=["ner", "parser"]
        )

    def _load_data(
        self,
        path: Path,
    ) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Error: File {path} not found.")

        data: list[dict[str, Any]] = []

        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                try:
                    record: dict[str, Any] = json.loads(line.strip())
                    data.append(record)
                except json.JSONDecodeError:
                    continue

        return data

    def _find_word_offsets(
        self,
        sentence: str,
        lemma: str,
    ) -> list[tuple[int, int]]:
        word_offsets: list[tuple[int, int]] = []

        sentence_doc: Doc = self._nlp(sentence)
        lemma_doc: Doc = self._nlp(lemma)

        lemma_texts: list[str] = [
            token.text.lower() for token in lemma_doc if token.text not in (" ", "-")
        ]

        lemma_lemmas: list[str] = [
            token.lemma_.lower()
            for token in lemma_doc
            if token.lemma_ not in (" ", "-")
        ]

        for i in range(len(sentence_doc)):
            collected_tokens: list[Token] = []
            j: int = i

            while j < len(sentence_doc) and len(collected_tokens) < len(lemma_texts):
                if sentence_doc[j].text not in (" ", "-"):
                    collected_tokens.append(sentence_doc[j])

                j += 1

            if len(collected_tokens) < len(lemma_texts):
                break

            if [token.text.lower() for token in collected_tokens] == lemma_texts or [
                token.lemma_.lower() for token in collected_tokens
            ] == lemma_lemmas:
                start: int = sentence_doc[i].idx
                end: int = sentence_doc[j - 1].idx + len(sentence_doc[j - 1].text)

                if (start, end) not in word_offsets:
                    word_offsets.append((start, end))
            elif sentence_doc[i].text.lower() == "".join(lemma_texts) or sentence_doc[
                i
            ].lemma_.lower() == "".join(lemma_lemmas):
                start: int = sentence_doc[i].idx
                end: int = sentence_doc[i].idx + len(sentence_doc[i].text)

                if (start, end) not in word_offsets:
                    word_offsets.append((start, end))

        return word_offsets

    @abstractmethod
    def build(
        self,
        dataset_path: Path,
        output_dir: Path,
    ):
        pass
