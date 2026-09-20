#!/usr/bin/env python3
"""Chapter 0-8 research entry point.

The package lives under ``src/``, so it is importable only when ``src`` is on
``sys.path``.  ``PYTHONPATH=src python3 run_research.py`` does that from a
shell, but an IDE run/debug button usually does not, which is why this script
puts ``src`` on the path itself.  Running it from any directory, from an IDE,
or from an editable install all work the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_package_on_path() -> None:
    source_root = Path(__file__).resolve().parent / "src"
    if (source_root / "cpp_search" / "__init__.py").is_file():
        path = str(source_root)
        if path not in sys.path:
            sys.path.insert(0, path)


_ensure_package_on_path()

from cpp_search.runner import main  # noqa: E402


if __name__ == "__main__":
    main()
