"""Generic typed component registry for modular extension in Complex Flow Matching."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar, cast, overload

T = TypeVar("T")


class Registry(Generic[T]):
    """A generic typed registry for plug-and-play architectural components.

    Provides a clean decorator-based registration system, name-based lookup,
    factory instantiation, and discovery introspection.

    Example:
        >>> MODELS = Registry[nn.Module]("models")
        >>> @MODELS.register("my_model")
        ... class MyModel(nn.Module):
        ...     pass
        >>> model = MODELS.build("my_model", ...)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[T] | Callable[..., T]] = {}

    @property
    def name(self) -> str:
        """Name of the registry."""
        return self._name

    @overload
    def register(
        self, name: str | None = None, *, force: bool = False
    ) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]]: ...

    @overload
    def register(
        self, name: type[T] | Callable[..., T], *, force: bool = False
    ) -> type[T] | Callable[..., T]: ...

    def register(
        self,
        name: str | type[T] | Callable[..., T] | None = None,
        *,
        force: bool = False,
    ) -> Any:
        """Register a class or factory callable under a given name.

        Can be used as:
            @REGISTRY.register("custom_name")
            class MyClass: ...

            @REGISTRY.register
            class MyClass: ...

            REGISTRY.register("custom_name")(MyClass)

        Args:
            name: Optional registration key string or the decorated class itself.
            force: If True, allows overwriting an existing entry.

        Returns:
            The decorated class/callable or a decorator function.
        """
        # Case 1: Used as bare decorator @REGISTRY.register without arguments
        if callable(name) and not isinstance(name, str):
            fn_or_cls = name
            key = fn_or_cls.__name__
            self._do_register(key, fn_or_cls, force=force)
            return fn_or_cls

        # Case 2: Used with name (or no args with parens @REGISTRY.register())
        def decorator(target: type[T] | Callable[..., T]) -> type[T] | Callable[..., T]:
            key = cast(str, name) if isinstance(name, str) else target.__name__
            self._do_register(key, target, force=force)
            return target

        return decorator

    def _do_register(
        self,
        key: str,
        target: type[T] | Callable[..., T],
        force: bool = False,
    ) -> None:
        if key in self._registry and not force:
            raise ValueError(
                f"Cannot register '{key}' into registry '{self._name}': "
                f"key already registered by {self._registry[key]}. Use force=True to overwrite."
            )
        self._registry[key] = target

    def get(self, name: str) -> type[T] | Callable[..., T]:
        """Retrieve a registered class or factory by name.

        Args:
            name: Registration key.

        Returns:
            The registered class or factory.

        Raises:
            KeyError: If name is not registered.
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in registry '{self._name}'. " f"Available: {self.list()}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> T:
        """Instantiate a registered component by name with keyword arguments.

        Args:
            name: Registration key.
            **kwargs: Arguments to pass to the constructor / factory.

        Returns:
            Instantiated object of type T.

        Raises:
            KeyError: If name is not registered.
        """
        cls_or_fn = self.get(name)
        return cls_or_fn(**kwargs)

    def list(self) -> list[str]:
        """Return a sorted list of registered keys."""
        return sorted(self._registry.keys())

    def contains(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry

    def __contains__(self, name: str) -> bool:
        return self.contains(name)

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, items={self.list()})"


# Pre-instantiated core registries
MANIFOLDS: Registry[Any] = Registry("manifolds")
MODELS: Registry[Any] = Registry("models")
SOLVERS: Registry[Any] = Registry("solvers")
LOSSES: Registry[Any] = Registry("losses")
DATASETS: Registry[Any] = Registry("datasets")
RECONSTRUCTORS: Registry[Any] = Registry("reconstructors")
