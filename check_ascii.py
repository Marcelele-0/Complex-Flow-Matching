"""Fail on non-ASCII characters in the source and configuration tree.

The repository keeps its code and configs ASCII-only on purpose: the docstrings
carry a lot of prose, and a stray en-dash or curly quote pasted from a paper or a
chat window is invisible in review and shows up as mojibake in a terminal on a
cluster login node.

This used to be a hardcoded list of fifteen paths. Three of them
(``core/reconstructor.py``, ``models/varnet.py``, ``suite.py``) had been deleted
long before, and because a read error was caught and printed without setting the
failure flag, the script still exited 0 -- it reported "All clean!" while checking
almost nothing. It now walks the tree, so it cannot drift from it, and a file it
cannot read is an error rather than a shrug.

The manuscript is no longer in this repository, so nothing here is exempt: every
file the walk reaches is expected to be ASCII.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOTS = ("src", "conf", "tests", "scripts")
SUFFIXES = {".py", ".yaml", ".yml"}


def offending_lines(path: Path) -> list[tuple[int, str]]:
    """Return the ``(line number, offending characters)`` pairs in one file.

    Args:
        path: File to scan.

    Returns:
        One entry per line that carries at least one character above U+007F.

    Raises:
        OSError: If the file cannot be read.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    found: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            non_ascii = "".join(character for character in line if ord(character) > 127)
            if non_ascii:
                found.append((number, non_ascii))
    return found


def main() -> int:
    """Scan every tracked source file and report non-ASCII characters.

    Returns:
        1 if any file carries non-ASCII characters or could not be read, else 0.
    """
    failed = False
    scanned = 0

    for root in ROOTS:
        base = Path(root)
        if not base.is_dir():
            print(f"{root}: missing directory")
            failed = True
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SUFFIXES or "__pycache__" in path.parts:
                continue
            scanned += 1
            try:
                for number, characters in offending_lines(path):
                    print(f"{path}:{number}: non-ASCII: {characters}")
                    failed = True
            except (OSError, UnicodeDecodeError) as error:
                print(f"{path}: cannot read: {error}")
                failed = True

    print(f"scanned {scanned} files")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
