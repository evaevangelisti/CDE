from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

CHECKPOINT_INTERVAL: int = 50

# Paths

CACHE_DIR: Path = Path(user_cache_dir(appname="CDE"))

CHECKPOINT_FILENAME: Path = Path("checkpoint.jsonl")
ANNOTATED_SENTENCES_FILENAME: Path = Path("annotated_sentences.jsonl")

EVALUATION_FILENAME: Path = Path("evaluation.json")

# Datasets

DATASETS: dict[str, str] = {
    "wiktionary": "wiktionary.jsonl.gz",
    "raganato": "raganato.jsonl.gz",
}

# Prompt templates

SENTENCE_DEFINITION_SELECTION_PROMPT_TEMPLATE: str = """Read the sentence: {sentence}

Choose the correct dictionary definition of the word "{word}":
{definitions}

Provide as output only the number of the correct definition."""

CONTINUATION_DEFINITION_SELECTION_PROMPT_TEMPLATE: str = """Read the sentence: {sentence}
The following continuation implicitly disambiguates the word "{word}": {continuation}

Based on both the sentence and the continuation, choose the correct dictionary definition of the word "{word}":
{definitions}

Provide as output only the number of the correct definition."""

CONTINUATION_GENERATION_PROMPT_TEMPLATES: dict[str, str] = {
    "free": """Read the sentence: {sentence}

Naturally continue the thought.

Provide as output only the complete sentence with the continuation.""",
    "focused": """Read the sentence: {sentence}

Naturally continue the thought, focusing on the word "{word}".

Provide as output only the complete sentence with the continuation.""",
    "grounded": """Read the sentence: {sentence}

Continue the thought by adding a brief clause that grounds the specific scenario of "{word}".

Provide as output only the complete sentence with the continuation.""",
    "constrained": """Read the sentence: {sentence}

Complete the thought by adding only 3 to 5 terminology-dense words that extend the specific aspect of "{word}".

Provide as output only the complete sentence with the continuation.""",
}

# Generation options

DEFAULT_DEFINITION_SELECTION_OPTIONS: dict[str, Any] = {
    "temperature": 0.2,
}

DEFAULT_CONTINUATION_GENERATION_OPTIONS: dict[str, Any] = {
    "temperature": 0.8,
}
