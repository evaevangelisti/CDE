from typing import Any

import ollama

from ..config import TIMEOUT
from .base import Generator


class OllamaGenerator(Generator):
    """
    Ollama generator.
    """

    def __init__(
        self,
        model: str,
        host: str | None = None,
        timeout: float = TIMEOUT,
    ):
        """
        Initialize the Ollama generator.

        Args:
            model (str): Ollama model.
            host (str | None): Ollama host. If None, the default host will be used.
            timeout (float): Timeout for Ollama API requests in seconds.
        """
        self._model: str = model

        self._client: ollama.Client = ollama.Client(host=host, timeout=timeout)
        self._ensure_model()

    def _ensure_model(
        self,
    ) -> None:
        """
        Ensure that the specified model is available in Ollama. If not, attempt to pull it.

        Raises:
            ValueError: If the model is not available in Ollama and cannot be pulled, or if there is an issue connecting to the Ollama API.
        """
        try:
            if self._model not in [model.model for model in self._client.list().models]:
                try:
                    self._client.pull(self._model)
                except Exception:
                    raise ValueError(
                        f"Error: model '{self._model}' not available in Ollama"
                    )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Error: unable to connect to Ollama API. Details: {str(e)}"
            )

    def generate(
        self,
        prompt: str,
        options: dict[str, Any] = {},
    ) -> str:
        """
        Generate text using the Ollama API.

        Args:
             prompt (str): Prompt.
             options (dict[str, Any]): Options.

        Returns:
            str: Generated text.
        """
        return self._client.generate(
            model=self._model,
            prompt=prompt,
            think=False,
            options=options,
        )["response"].strip()
