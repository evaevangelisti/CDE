from pathlib import Path

from cde.models import Language

# Constants

DEFAULT_BUILDER = "wsc"

SPACY_MODELS: dict[Language, str] = {
    Language.ENGLISH: "en_core_web_lg",
    Language.ITALIAN: "it_core_news_lg",
}

# Paths

DATA_DIR: Path = Path("data")

RAW_DIR: Path = DATA_DIR / "raw"
DATASETS_DIR: Path = DATA_DIR / "datasets"

WIKTIONARY_SENTENCES_WITH_WIKTIONARY_DEFINITIONS_FILENAME: Path = Path(
    "wiktionary_sentences_with_wiktionary_definitions.jsonl"
)

WIKTIONARY_SENTENCES_WITH_WORDNET_DEFINITIONS_FILENAME: Path = Path(
    "wiktionary_sentences_with_wordnet_definitions.jsonl"
)

WORDNET_SENTENCES_WITH_WIKTIONARY_DEFINITIONS_FILENAME: Path = Path(
    "wordnet_sentences_with_wiktionary_definitions.jsonl"
)

WORDNET_SENTENCES_WITH_WORDNET_DEFINITIONS_FILENAME: Path = Path(
    "wordnet_sentences_with_wordnet_definitions.jsonl"
)
