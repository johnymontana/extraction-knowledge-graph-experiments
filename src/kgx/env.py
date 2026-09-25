"""Settings from the environment, with the repo's ``.env`` as a fallback.

Keys belong out of notebooks and out of shell history. Copy ``.env.example`` to
``.env`` at the repo root (gitignored) and fill it in; :func:`getenv` reads the
process environment first and the file second, so an exported variable always
wins over the file.

A deliberately small parser rather than a ``python-dotenv`` dependency:
``KEY=value`` lines, an optional ``export`` prefix, ``#`` comments, and single
or double quotes around a value. Nothing is expanded or interpolated, and
nothing is written into ``os.environ``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = ["DOTENV", "read_dotenv", "getenv"]

#: The repo-root ``.env``. Tests point this somewhere empty.
DOTENV = Path(__file__).resolve().parents[2] / ".env"

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def read_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Parse a ``.env`` file into a dict. A missing file is an empty dict."""
    path = Path(path) if path is not None else DOTENV
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[key] = value
    return values


def getenv(name: str, default: str | None = None, *, path: str | Path | None = None) -> str | None:
    """``os.environ[name]`` if set and non-empty, else the ``.env`` value, else ``default``."""
    return os.environ.get(name) or read_dotenv(path).get(name) or default
