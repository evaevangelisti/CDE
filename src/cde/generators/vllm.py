from typing import Any

from vllm import LLM, SamplingParams

from ..models import Backend
from .base import Generator
from .factory import GeneratorFactory


@GeneratorFactory.register(Backend.VLLM)
class VLLMGenerator(Generator):
    def __init__(
        self,
        model: str,
        **kwargs: Any,
    ):
        """
        Initialize the vLLM generator.

        Args:
            model (str): vLLM model.
            **kwargs: Additional keyword arguments to pass to the vLLM client.

        Raises:
            RuntimeError: If the vLLM client fails to initialize.
        """
        super().__init__(model=model)

        try:
            self._llm: LLM = LLM(
                model,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize vLLM client for model '{model}': {e}"
            ) from e

    def generate(
        self,
        prompts: list[str],
        options: dict[str, Any] = {},
    ) -> list[str]:
        """
        Generate text using the vLLM API.

        Args:
            prompts (list[str]): Prompts.
            options (dict[str, Any]): Generation options.

        Returns:
            list[str]: Generated text for each prompt.
        """
        if not prompts:
            return []

        return [
            output.outputs[0].text
            for output in self._llm.generate(
                prompts,
                SamplingParams(
                    temperature=options.get("temperature", 0.8),
                    seed=options.get("seed"),
                ),
                use_tqdm=False,
            )
        ]
