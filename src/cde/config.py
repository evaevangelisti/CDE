from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

from .models import Language

# Constants

CHECKPOINT_INTERVAL: int = 25

# Paths

CACHE_DIR: Path = Path(user_cache_dir(appname="CDE"))

CHECKPOINT_FILENAME: Path = Path("checkpoint.jsonl")
ANNOTATED_SENTENCES_FILENAME: Path = Path("annotated_sentences.jsonl")

EVALUATION_FILENAME: Path = Path("evaluation.json")

# Datasets

DATASETS: dict[str, str] = {
    "wiktionary": "https://github.com/evaevangelisti/CDE/releases/download/v0.1.0/wiktionary.jsonl.gz",
    "wordnet": "https://github.com/evaevangelisti/CDE/releases/download/v0.1.0/wordnet.jsonl.gz",
    "raganato": "https://github.com/evaevangelisti/CDE/releases/download/v0.1.0/raganato.jsonl.gz",
}

# Prompt templates

DEFINITION_SELECTION_PROMPT_TEMPLATES: dict[Language, dict[str, str]] = {
    Language.ENGLISH: {
        "sentence": """Read the sentence: {sentence}

Choose the correct dictionary definition of the word "{word}":
{definitions}

Provide as output only the number of the correct definition.""",
        "continuation": """Read the sentence: {sentence}
The following continuation implicitly disambiguates the word "{word}": {continuation}

Based on both the sentence and the continuation, choose the correct dictionary definition of the word "{word}":
{definitions}

Provide as output only the number of the correct definition.""",
    },
    Language.ITALIAN: {
        "sentence": """Leggi la frase: {sentence}

Scegli la definizione corretta del dizionario per la parola "{word}":
{definitions}

Fornisci come output solo il numero della definizione corretta.""",
        "continuation": """Leggi la frase: {sentence}
La seguente continuazione disambigua implicitamente la parola "{word}": {continuation}

In base sia alla frase che alla continuazione, scegli la definizione corretta del dizionario per la parola "{word}":
{definitions}

Fornisci come output solo il numero della definizione corretta.""",
    },
}

CONTINUATION_GENERATION_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "free": {
        "prompts": {
            Language.ENGLISH: """Given the following sentence, naturally continue the thought: {sentence}
Provide the complete sentence with the continuation, without explanations or formatting.""",
            Language.ITALIAN: """Data la seguente frase, continua il pensiero in modo naturale: {sentence}
Fornisci la frase completa di continuazione, senza spiegazioni o formattazione.""",
        },
        "options": {
            "temperature": 0.4,
        },
    },
    "focused": {
        "prompts": {
            Language.ENGLISH: """Given the following sentence, focus on the word "{word}" and naturally continue the thought: {sentence}
Provide the complete sentence with the continuation, without explanations or formatting.""",
            Language.ITALIAN: """Data la seguente frase, concentrati sulla parola "{word}" e continua il pensiero in modo naturale: {sentence}
Fornisci la frase completa di continuazione, senza spiegazioni o formattazione.""",
        },
        "options": {
            "temperature": 0.4,
        },
    },
    "grounded": {
        "prompts": {
            Language.ENGLISH: """Given the following sentence, focus on the word "{word}" and continue the thought by adding a brief clause that grounds that specific scenario: {sentence}
Provide the complete sentence with the continuation, without explanations or formatting.""",
            Language.ITALIAN: """Data la seguente frase, concentrati sulla parola "{word}" e continua il pensiero aggiungendo una breve proposizione che contestualizzi quello specifico scenario: {sentence}
Fornisci la frase completa di continuazione, senza spiegazioni o formattazione.""",
        },
        "options": {
            "temperature": 0.2,
        },
    },
    "constrained": {
        "prompts": {
            Language.ENGLISH: """Given the following sentence, focus on the word "{word}" and complete the thought by adding only 3 to 5 terminology-dense words that naturally extend that specific aspect: {sentence}
Provide the complete sentence with the continuation, without explanations or formatting.""",
            Language.ITALIAN: """Data la seguente frase, concentrati sulla parola "{word}" e completa il pensiero aggiungendo solo da 3 a 5 parole dense di terminologia specifica che estendano in modo naturale quel determinato aspetto: {sentence}
Fornisci la frase completa di continuazione, senza spiegazioni o formattazione.""",
        },
        "options": {
            "temperature": 0.2,
        },
    },
}

# Generation options

TIMEOUT: float = 120.0

DEFAULT_DEFINITION_SELECTION_OPTIONS: dict[str, Any] = {
    "temperature": 0.0,
}
