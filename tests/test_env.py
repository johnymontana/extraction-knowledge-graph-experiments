"""``kgx.env``: the environment first, the repo's ``.env`` second."""

import kgx.env as ke


def test_read_dotenv_parses_the_common_forms(tmp_path):
    f = tmp_path / ".env"
    f.write_text("\n".join([
        "# a comment",
        "",
        "PLAIN=abc",
        "export EXPORTED=def",
        "DOUBLE=\"with # hash\"",
        "SINGLE='x y'",
        "TRAILING=value  # comment",
        "EMPTY=",
        "not a line",
    ]))
    assert ke.read_dotenv(f) == {"PLAIN": "abc", "EXPORTED": "def", "DOUBLE": "with # hash",
                                 "SINGLE": "x y", "TRAILING": "value", "EMPTY": ""}


def test_a_missing_file_is_empty(tmp_path):
    assert ke.read_dotenv(tmp_path / "absent") == {}


def test_getenv_prefers_the_environment(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("KGX_TEST_KEY=from-file\n")
    monkeypatch.delenv("KGX_TEST_KEY", raising=False)
    assert ke.getenv("KGX_TEST_KEY", path=f) == "from-file"
    monkeypatch.setenv("KGX_TEST_KEY", "from-env")
    assert ke.getenv("KGX_TEST_KEY", path=f) == "from-env"
    assert ke.getenv("KGX_ABSENT", "fallback", path=f) == "fallback"


def test_the_default_file_is_neutralised_in_tests(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert ke.getenv("TYPESAFE_API_KEY") is None


def test_cached_typesafe_reads_the_key_from_dotenv(tmp_path, monkeypatch):
    from kgx.typesafe import CachedTypeSafe

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=ts-test\n")
    monkeypatch.setattr(ke, "DOTENV", tmp_path / ".env")
    assert CachedTypeSafe(cache_dir=tmp_path / "cache").available
