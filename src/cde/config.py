from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

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

SENTENCE_DEFINITION_SELECTION_EXAMPLES: str = """Example 1:
Read the sentence: the river bank was covered in mud after the heavy storm

Choose the correct dictionary definition of the word "bank":
0) sloping land (especially the slope beside a body of water)
1) a financial institution that accepts deposits and channels the money into lending activities

Provide as output only the number of the correct definition.
Output: 0

Example 2:
Read the sentence: The automobile company decided to build a massive new plant in the industrial area.

Choose the correct dictionary definition of the word "plant":
0) (botany) An organism that is not an animal, especially an organism capable of photosynthesis, and typically a small or herbaceous organism of this kind, rather than a tree.
1) (countable) A factory or other industrial or institutional building or facility.

Provide as output only the number of the correct definition.
Output: 1

"""

SENTENCE_DEFINITION_SELECTION_PROMPT_TEMPLATE: str = """{examples}Read the sentence: {sentence}

Choose the correct dictionary definition of the word "{word}":
{definitions}

Provide as output only the number of the correct definition."""

CONTINUATION_DEFINITION_SELECTION_EXAMPLES: str = """Example 1:
Read the sentence: the river bank was covered in mud after the heavy storm
The following continuation implicitly disambiguates the word "bank": the river bank was covered in mud after the heavy storm, making it difficult for fishermen to find a safe spot along the rushing water

Based on both the sentence and the continuation, choose the correct dictionary definition of the word "bank":
0) sloping land (especially the slope beside a body of water)
1) a financial institution that accepts deposits and channels the money into lending activities

Provide as output only the number of the correct definition.
Output: 0

Example 2:
Read the sentence: The automobile company decided to build a massive new plant in the industrial area.
The following continuation implicitly disambiguates the word "plant": The automobile company decided to build a massive new plant in the industrial area, to streamline the assembly line and accelerate electric vehicle manufacturing

Based on both the sentence and the continuation, choose the correct dictionary definition of the word "plant":
0) (botany) An organism that is not an animal, especially an organism capable of photosynthesis, and typically a small or herbaceous organism of this kind, rather than a tree.
1) (countable) A factory or other industrial or institutional building or facility.

Provide as output only the number of the correct definition.
Output: 1

"""

CONTINUATION_DEFINITION_SELECTION_PROMPT_TEMPLATE: str = """{examples}Read the sentence: {sentence}
The following continuation implicitly disambiguates the word "{word}": {continuation}

Based on both the sentence and the continuation, choose the correct dictionary definition of the word "{word}":
{definitions}

Provide as output only the number of the correct definition."""

CONTINUATION_GENERATION_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "free": {
        "prompt": """Given the following sentence, naturally continue the thought: {sentence}
Provide the complete sentence with the continuation, without explanations or formatting.""",
        "options": {
            "temperature": 0.4,
        },
    },
    "focused": {
        "prompt": """Given the following sentence, focus on the word "{word}" and naturally continue the thought: {sentence}
Provide the complete sentence with the continuation, without explanations or formatting.""",
        "options": {
            "temperature": 0.4,
        },
    },
    "grounded": {
        "prompt": """Given the following sentence, focus on the word "{word}" and continue the thought by adding a brief clause that grounds that specific scenario: {sentence}
Provide the complete sentence with the continuation, without explanations or formatting.""",
        "options": {
            "temperature": 0.2,
        },
    },
    "constrained": {
        "prompt": """Given the following sentence, focus on the word "{word}" and complete the thought by adding only 3 to 5 terminology-dense words: {sentence}
Provide the complete sentence with the continuation, without explanations or formatting.""",
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
