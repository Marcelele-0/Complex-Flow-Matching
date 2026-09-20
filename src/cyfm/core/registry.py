"""Generic typed component registry for modular extension in Complex Flow Matching."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar, cast, overload

from torch import nn

from cyfm.core.dataset import BaseComplexDataset
from cyfm.core.manifold import BaseManifold
from cyfm.core.protocols import Coupling, Sampler

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
                f"'{name}' not found in registry '{self._name}'. Available: {self.list()}"
            )
        return self._registry[name]

    def get_class(self, name: str) -> type[T]:
        """Retrieve a registered entry, insisting it is a class.

        :meth:`get` may hand back a factory function, which is fine for
        :meth:`build` but not for a caller that needs a classmethod -- a
        geometry's ``from_config``, say. Narrowing here keeps that ``isinstance``
        out of every such call site.

        Args:
            name: Registration key.

        Returns:
            The registered class.

        Raises:
            KeyError: If name is not registered.
            TypeError: If the entry is a factory callable rather than a class.
        """
        entry = self.get(name)
        if not isinstance(entry, type):
            raise TypeError(
                f"'{name}' in registry '{self._name}' is a factory callable, not a class; "
                f"it cannot be built from a classmethod. Register the class instead."
            )
        return entry

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


# The package's registries, each typed to what it actually holds. They were all
# Registry[Any], which made `build` return Any and pushed a cast onto every
# consumer -- cyfm.data and the registry's own tests both carried one. Typing
# them here deletes those casts and lets mypy reject a class registered into the
# wrong registry at the decorator.
MANIFOLDS: Registry[BaseManifold] = Registry("manifolds")
MODELS: Registry[nn.Module] = Registry("models")
SOLVERS: Registry[Sampler] = Registry("solvers")
DATASETS: Registry[BaseComplexDataset] = Registry("datasets")
COUPLINGS: Registry[Coupling] = Registry("couplings")

# The decorator spelling used at the definition sites. `@register_dataset("knee")`
# reads as what it does where it is applied; `DATASETS.register` names the table
# being mutated, which is the detail the reader at that line cares least about.
# Both are the same object, so there is no second mechanism to keep in step.
register_manifold = MANIFOLDS.register
register_model = MODELS.register
register_solver = SOLVERS.register
register_dataset = DATASETS.register
register_coupling = COUPLINGS.register
