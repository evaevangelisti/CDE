from abc import ABC, abstractmethod
from typing import Any


class Generator(ABC):
    """
    Base generator.
    """

    def __init__(
        self,
        model: str,
        seed: int | None = None,
    ):
        """
        Initialize the generator.

        Args:
            model (str): Model name.
            seed (int | None): Random seed for generation.
        """
        self._model: str = model
        self._seed: int | None = seed

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        options: dict[str, Any] = {},
    ) -> list[str]:
        """
        Generate text based on the given prompts and options.

        Args:
            prompt (list[str]): Prompts.
            options (dict[str, Any]): Generation options.

        Returns:
            list[str]: Generated text for each prompt.
        """
        pass
