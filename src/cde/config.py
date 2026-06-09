from pathlib import Path
from typing import Any

# Paths

CDE_DIR: Path = Path("cde")

ANNOTATED_SENTENCES_PATH: Path = CDE_DIR / "annotated_sentences.jsonl"
EVALUATION_PATH: Path = CDE_DIR / "evaluation.json"

# Prompt templates

SENTENCE_DEFINITION_SELECTION_PROMPT_TEMPLATE: str = """"""
CONTINUATION_DEFINITION_SELECTION_PROMPT_TEMPLATE: str = """"""

CONTINUATION_GENERATION_PROMPT_TEMPLATES: dict[str, str] = {
    "free": """""",
    "focused": """""",
    "grounded": """""",
    "constrained": """""",
}

# Generation options

DEFAULT_DEFINITION_SELECTION_OPTIONS: dict[str, Any] = {
    "temperature": 0.2,
}

DEFAULT_CONTINUATION_GENERATION_OPTIONS: dict[str, Any] = {
    "temperature": 0.8,
}
