import json
from pathlib import Path
from typing import Any

from nltk.corpus import wordnet as wn
from tqdm import tqdm

from ..config import (
    WIKTIONARY_SENTENCES_WITH_WIKTIONARY_DEFINITIONS_FILENAME,
    WIKTIONARY_SENTENCES_WITH_WORDNET_DEFINITIONS_FILENAME,
    WORDNET_SENTENCES_WITH_WIKTIONARY_DEFINITIONS_FILENAME,
    WORDNET_SENTENCES_WITH_WORDNET_DEFINITIONS_FILENAME,
)
from .base import Builder
from .factory import BuilderFactory


@BuilderFactory.register("wsc")
class WSCBuilder(Builder):
    @staticmethod
    def _ensure_wordnet() -> None:
        """
        Ensure that the WordNet corpus is downloaded.
        """
        import nltk

        nltk.download("wordnet", quiet=True)

    def build(
        self,
        dataset_path: Path,
        output_dir: Path,
    ) -> None:
        self._ensure_wordnet()

        lemmas: list[dict[str, Any]] = self._load_data(dataset_path)

        wiktionary_sentences_with_wiktionary_senses_path: Path = (
            output_dir / WIKTIONARY_SENTENCES_WITH_WIKTIONARY_DEFINITIONS_FILENAME
        )

        wiktionary_sentences_with_wordnet_senses_path: Path = (
            output_dir / WIKTIONARY_SENTENCES_WITH_WORDNET_DEFINITIONS_FILENAME
        )

        wordnet_sentences_with_wiktionary_senses_path: Path = (
            output_dir / WORDNET_SENTENCES_WITH_WIKTIONARY_DEFINITIONS_FILENAME
        )

        wordnet_sentences_with_wordnet_senses_path: Path = (
            output_dir / WORDNET_SENTENCES_WITH_WORDNET_DEFINITIONS_FILENAME
        )

        with (
            wiktionary_sentences_with_wiktionary_senses_path.open(
                "w",
                encoding="utf-8",
            ) as wiktionary_sentences_with_wiktionary_senses_file,
            wiktionary_sentences_with_wordnet_senses_path.open(
                "w",
                encoding="utf-8",
            ) as wiktionary_sentences_with_wordnet_senses_file,
            wordnet_sentences_with_wiktionary_senses_path.open(
                "w",
                encoding="utf-8",
            ) as wordnet_sentences_with_wiktionary_senses_file,
            wordnet_sentences_with_wordnet_senses_path.open(
                "w",
                encoding="utf-8",
            ) as wordnet_sentences_with_wordnet_senses_file,
        ):
            for file in [
                wiktionary_sentences_with_wiktionary_senses_file,
                wiktionary_sentences_with_wordnet_senses_file,
                wordnet_sentences_with_wiktionary_senses_file,
                wordnet_sentences_with_wordnet_senses_file,
            ]:
                file.write(json.dumps({"__metadata__": {"language": "en"}}) + "\n")

            for lemma in tqdm(lemmas, desc="Building dataset", unit="lemma"):
                wiktionary_definitions: list[str] = [
                    sense["definition"] for sense in lemma["senses"]
                ]

                pos_tags: list[str] | None = {
                    "noun": ["n"],
                    "verb": ["v"],
                    "adj": ["a", "s"],
                    "adv": ["r"],
                }.get(lemma["pos"])

                if not pos_tags:
                    continue

                wordnet_synsets: list = []

                for pos_tag in pos_tags:
                    wordnet_synsets.extend(wn.synsets(lemma["lemma"], pos=pos_tag))

                if not wordnet_synsets:
                    continue

                wordnet_synset_ids: list[str] = [
                    synset.name() for synset in wordnet_synsets
                ]

                wordnet_definitions: list[str] = [
                    synset.definition() for synset in wordnet_synsets
                ]

                for sense in lemma["senses"]:
                    if (
                        not sense.get("wordnet_synset_id")
                        or sense["wordnet_synset_id"] not in wordnet_synset_ids
                    ):
                        continue

                    try:
                        if not (synset := wn.synset(sense["wordnet_synset_id"])):
                            continue
                    except ValueError:
                        continue

                    counters: dict[str, int] = {
                        "wiktionary_sentences_with_wiktionary_senses": 0,
                        "wiktionary_sentences_with_wordnet_senses": 0,
                        "wordnet_sentences_with_wiktionary_senses": 0,
                        "wordnet_sentences_with_wordnet_senses": 0,
                    }

                    wiktionary_sense_index: int = wiktionary_definitions.index(
                        sense["definition"]
                    )

                    wordnet_sense_index: int = wordnet_synset_ids.index(
                        sense["wordnet_synset_id"]
                    )

                    for sentence in sense.get("sentences", []):
                        for word_offset in sentence.get("word_offsets", []):
                            for file, name, definitions, sense_id, sense_index in [
                                (
                                    wiktionary_sentences_with_wiktionary_senses_file,
                                    "wiktionary_sentences_with_wiktionary_senses",
                                    wiktionary_definitions,
                                    f"{lemma['id']}.S{wiktionary_sense_index:02d}",
                                    wiktionary_sense_index,
                                ),
                                (
                                    wiktionary_sentences_with_wordnet_senses_file,
                                    "wiktionary_sentences_with_wordnet_senses",
                                    wordnet_definitions,
                                    sense["wordnet_synset_id"],
                                    wordnet_sense_index,
                                ),
                            ]:
                                file.write(
                                    json.dumps(
                                        {
                                            "instance_id": f"{lemma['id']}.S{sense_index:02d}.C{counters[name]:02d}",
                                            "lemma": lemma["lemma"],
                                            "pos": lemma["pos"],
                                            "definitions": definitions,
                                            "sense_id": sense_id,
                                            "sense_index": sense_index,
                                            "sentence": sentence["sentence"],
                                            "word_offset": word_offset,
                                        }
                                    )
                                    + "\n"
                                )

                                counters[name] += 1

                    for sentence in synset.examples():
                        for word_offset in self._find_word_offsets(
                            sentence,
                            lemma["lemma"],
                        ):
                            for file, name, definitions, sense_id, sense_index in [
                                (
                                    wordnet_sentences_with_wiktionary_senses_file,
                                    "wordnet_sentences_with_wiktionary_senses",
                                    wiktionary_definitions,
                                    f"{lemma['id']}.S{wiktionary_sense_index:02d}",
                                    wiktionary_sense_index,
                                ),
                                (
                                    wordnet_sentences_with_wordnet_senses_file,
                                    "wordnet_sentences_with_wordnet_senses",
                                    wordnet_definitions,
                                    sense["wordnet_synset_id"],
                                    wordnet_sense_index,
                                ),
                            ]:
                                file.write(
                                    json.dumps(
                                        {
                                            "instance_id": f"{lemma['id']}.S{sense_index:02d}.C{counters[name]:02d}",
                                            "lemma": lemma["lemma"],
                                            "pos": lemma["pos"],
                                            "definitions": definitions,
                                            "sense_id": sense_id,
                                            "sense_index": sense_index,
                                            "sentence": sentence,
                                            "word_offset": word_offset,
                                        }
                                    )
                                    + "\n"
                                )

                                counters[name] += 1
