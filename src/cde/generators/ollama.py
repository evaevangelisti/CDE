from typing import Any

import ollama

from ..models import Backend
from .base import Generator
from .factory import GeneratorFactory


@GeneratorFactory.register(Backend.OLLAMA)
class OllamaGenerator(Generator):
    """
    Ollama generator.
    """

    def __init__(
        self,
        model: str,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the Ollama generator.

        Args:
            model (str): Ollama model.
            seed (int | None): Random seed for generation.
            **kwargs: Additional keyword arguments to pass to the Ollama client.
        """
        super().__init__(model=model, seed=seed)

        self._client: ollama.Client = ollama.Client(**kwargs)
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
                    print("Pulling model from Ollama...")
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
        prompts: list[str],
        options: dict[str, Any] = {},
    ) -> list[str]:
        """
        Generate text using the Ollama API.

        Args:
             prompts (list[str]): Prompts.
             options (dict[str, Any]): Generation options.

        Returns:
            list[str]: Generated text for each prompt.
        """
        return [
            self._client.generate(
                model=self._model,
                prompt=prompt,
                think=True,
                options={
                    **options,
                    **({"seed": self._seed} if self._seed is not None else {}),
                },
            )["response"].strip()
            for prompt in prompts
        ]
