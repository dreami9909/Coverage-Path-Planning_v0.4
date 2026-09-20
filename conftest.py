"""pytest bootstrap: put the ``src`` layout on ``sys.path``.

Only needed for a source checkout that has not been installed with
``pip install -e .``; harmless otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if (_SOURCE_ROOT / "cpp_search" / "__init__.py").is_file():
    _path = str(_SOURCE_ROOT)
    if _path not in sys.path:
        sys.path.insert(0, _path)
