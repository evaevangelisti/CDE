from typing import Any, Callable, Type

from ..config import DEFAULT_BACKEND
from ..models import Backend
from .base import Generator


class GeneratorFactory:
    _registry: dict[str, Type[Generator]] = {}

    @classmethod
    def register(
        cls,
        *backends: Backend,
    ) -> Callable[[Type[Generator]], Type[Generator]]:
        def decorator(
            generator: Type[Generator],
        ) -> Type[Generator]:
            for backend in backends:
                cls._registry[backend] = generator

            return generator

        return decorator

    @classmethod
    def create(
        cls,
        backend: Backend,
        model: str,
        **kwargs: Any,
    ) -> Generator:
        generator: Type[Generator] | None = cls._registry.get(backend)

        if generator is None:
            generator = cls._registry.get(DEFAULT_BACKEND)

            if generator is None:
                raise ValueError("error: No default builder registered")

        return generator(model, **kwargs)
