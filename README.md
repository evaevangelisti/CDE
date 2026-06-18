# CDE

**Continuation-based Disambiguation Evaluator** — A toolkit for evaluating the implicit word sense disambiguation capabilities of large language models (LLMs).

## Installation

### Prerequisites 

- Python 3.10+

### Install

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev,docs]"
```

## Usage

Run the pipeline with the following command:

```bash
cde [options] <model> <datasets>
```

where `<datasets>` is one or more paths to dataset files.

## Dataset Format

Each dataset must be a JSONL file (optionally gzip-compressed), where each line contains the following fields:

| Field | Description |
| --- | --- |
| `instance_id` | Unique identifier for the instance |
| `lemma` | The target lemma to disambiguate |
| `pos` | Part-of-speech tag |
| `definitions` | List of candidate definitions |
| `sense_id` | Gold sense identifier |
| `sense_index` | Index of the correct sense |
| `sentence` | The context sentence |
| `word_offset` | Position of the target word in the sentence |
