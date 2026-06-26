import os
from typing import Any

from dotenv import load_dotenv
from vllm import LLM, SamplingParams

from ..config import MAX_TOKENS
from ..models import Backend
from .base import Generator
from .factory import GeneratorFactory


@GeneratorFactory.register(Backend.VLLM)
class VLLMGenerator(Generator):
    def __init__(
        self,
        model: str,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the vLLM generator.

        Args:
            model (str): vLLM model.
            seed (int | None): Random seed for generation.
            **kwargs: Additional keyword arguments to pass to the vLLM client.

        Raises:
            RuntimeError: If the vLLM client fails to initialize.
        """
        super().__init__(model=model, seed=seed)
        load_dotenv()

        try:
            self._llm: LLM = LLM(
                model,
                hf_token=os.getenv("HF_TOKEN"),
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

        outputs = [
            output.outputs[0].text.split("</think>")[-1].strip()
            for output in self._llm.chat(
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                SamplingParams(
                    temperature=options.get("temperature", 0.8),
                    max_tokens=MAX_TOKENS,
                    seed=self._seed,
                ),
                use_tqdm=False,
                chat_template_kwargs={
                    "thinking": True,
                    "enable_thinking": True
                },
            )
        ]

        print(outputs)
        return outputs
