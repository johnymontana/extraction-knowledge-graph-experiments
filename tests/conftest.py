"""Shared test setup.

The repo-root ``.env`` may hold a real ``TYPESAFE_API_KEY``. Every test sees an
empty one instead, so no test can pick the key up and make a live, billed call
-- the offline guarantee ``tests/test_typesafe.py`` states holds whatever is in
the developer's ``.env``.
"""

import pytest

import kgx.env


@pytest.fixture(autouse=True)
def _no_dotenv(tmp_path, monkeypatch):
    monkeypatch.setattr(kgx.env, "DOTENV", tmp_path / "no-such.env")
