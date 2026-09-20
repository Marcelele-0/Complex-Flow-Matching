"""Every script must at least import.

``scripts/`` holds the paper's network-free probes -- Table 1, Table 4 and the
Section 5.3 cost curve -- and they are reached only by name from
``conf/experiment/*.yaml``, so nothing in the suite executes them. That is how
``scripts/coupling_gate.py``, which measures the Factorised Coupling Trap, went on
importing a function that had been moved to another module: nothing looked.

Importing is a weak check and a cheap one. It catches exactly the failure this
was: a module-level import that no longer resolves.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MODULES = sorted(path for path in SCRIPTS.rglob("*.py") if path.name != "__init__.py")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(SCRIPTS)))
def test_script_imports(path: Path) -> None:
    """Import the module without running its ``main``.

    Every script here guards execution behind ``if __name__ == "__main__"``, so
    importing runs the module body -- imports, constants, argument-parser
    construction -- and nothing else.
    """
    # Modules under scripts/paper/ import each other by bare name, which works
    # because running one directly puts its directory first on sys.path. Reproduced
    # here rather than "fixed": those files are reached by path from
    # conf/experiment/*.yaml, and renaming their imports would change the contract
    # tests/test_experiments.py asserts.
    name = f"_script_{path.parent.name}_{path.stem}"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Registered before execution: a @dataclass declared at module level
        # resolves its own module out of sys.modules while being built, and gets
        # an AttributeError if it is not there yet.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
    finally:
        sys.path.remove(str(path.parent))


def test_the_sweep_is_not_empty() -> None:
    """A glob that matched nothing would make every assertion above vacuous."""
    assert len(MODULES) >= 10
