import re
from difflib import SequenceMatcher
from pathlib import Path
from string import Formatter

import requests
from tqdm import tqdm


def download(
    url: str,
    path: Path,
    chunk_size: int = 8192,
) -> None:
    """
    Download a file from a URL.

    Args:
        url (str): URL.
        path (Path): Path.
        chunk_size (int, optional): Chunk size.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    response: requests.Response = requests.get(url, stream=True)
    response.raise_for_status()

    with (
        path.open("wb") as file,
        tqdm(
            desc=path.name,
            total=int(response.headers.get("content-length", 0)),
            unit="B",
            unit_scale=True,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            file.write(chunk)
            pbar.update(len(chunk))


def retrieve_template_fields(
    template: str,
) -> set[str]:
    """
    Retrieve the fields from a template.

    Args:
        template (str): Template.

    Returns:
        set[str]: Set of fields in the template.
    """
    return {field for _, field, _, _ in Formatter().parse(template) if field}


def extract_predicted_sense_index(
    response: str,
    definitions: int,
) -> int | None:
    """
    Extract the predicted sense index from the response.

    Args:
        response (str): Response.
        definitions (int): Number of definitions.

    Returns:
        int | None: Predicted sense index (0-based) or None if not found or out of range.
    """
    match: re.Match[str] | None = re.search(r"\d+", response)
    if not match:
        return None

    predicted_sense_index: int = int(match.group()) - 1

    if 0 <= predicted_sense_index < definitions:
        return predicted_sense_index

    return None


def validate_continuation(
    sentence: str,
    continuation: str,
) -> bool:
    """
    Validate the continuation.

    Args:
        sentence (str): Original sentence.
        continuation (str): Continuation.

    Returns:
        bool: True if the continuation is valid, False otherwise.
    """
    sentence = sentence.strip().lower()
    continuation = continuation.strip().lower()

    if not continuation or len(continuation) <= len(sentence):
        return False

    if sentence in continuation:
        return True

    for start in range(len(continuation) - len(sentence) + 1):
        ratio = SequenceMatcher(
            None,
            sentence,
            continuation[start : start + len(sentence)],
        ).ratio()

        if ratio > 0.90:
            return True

    return False
