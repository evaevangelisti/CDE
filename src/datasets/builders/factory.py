from typing import Any, Callable, Type

from ..config import DEFAULT_BUILDER
from .base import Builder


class BuilderFactory:
    _registry: dict[str, Type[Builder]] = {}

    @classmethod
    def register(
        cls,
        *names: str,
    ) -> Callable[[Type[Builder]], Type[Builder]]:
        def decorator(
            builder: Type[Builder],
        ) -> Type[Builder]:
            for name in names:
                cls._registry[name] = builder

            return builder

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,
    ) -> Builder:
        builder: Type[Builder] | None = cls._registry.get(name)

        if builder is None:
            builder = cls._registry.get(DEFAULT_BUILDER)

            if builder is None:
                raise ValueError("error: No default builder registered")

        return builder(**kwargs)
