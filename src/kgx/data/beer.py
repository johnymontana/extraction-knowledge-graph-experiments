"""The Magellan Beer benchmark, as TypeSafe's entity-alignment cookbook uses it.

Two beer catalogues, BeerAdvocate (``tableA``, 4,345 rows) and RateBeer
(``tableB``, 3,000 rows), and 450 labelled candidate pairs between them that a
blocking pass already picked out, 68 of them the same beer. This is the
DeepMatcher release of the Magellan data, with its train/valid/test split
(268/91/91 pairs).

The cookbook (https://docs.typesafe.ai/cookbooks/entity_alignment) ships these
450 pairs as ``candidate_pairs.json`` with ids ``c000``..``c449``. That file is
the three splits concatenated in the order train, valid, test. Its first pair
and all four of its worked examples (``c100``, ``c427``, ``c428``, ``c446``)
land on those ids here, so results can be compared pair by pair.

Like the cookbook, the text is left exactly as published: HTML entities never
decoded (``&#40; Ohio &#41;``), apostrophes split off as separate words,
mis-decoded UTF-8 (``TrÃ ¶ egs``). Nothing is normalised on load.

The data is not vendored. :func:`load_pairs` downloads the 160 KB zip once
into ``output/`` (gitignored) and reads it from there after that.
"""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Mapping

__all__ = ["URL", "SPLITS", "FIELDS", "load_pairs", "pairs_from_tables"]

URL = ("https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/"
       "Structured/Beer/beer_exp_data.zip")

#: The order the cookbook's ids follow.
SPLITS: tuple[str, ...] = ("train", "valid", "test")

#: Column in the published tables -> field name in the cookbook's state.
FIELDS: dict[str, str] = {
    "Beer_Name": "name",
    "Brew_Factory_Name": "brewery",
    "Style": "style",
    "ABV": "abv",
}

_DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "output" / "beer"


def pairs_from_tables(
    table_a: Iterable[Mapping[str, str]],
    table_b: Iterable[Mapping[str, str]],
    labelled: Iterable[tuple[str, Iterable[Mapping[str, str]]]],
) -> list[dict]:
    """Join the labelled id pairs onto the two tables, in the cookbook's order.

    ``labelled`` is ``(split name, rows)`` in order, each row carrying
    ``ltable_id``, ``rtable_id`` and ``label``. Ids are assigned by position
    across all splits, so the order of ``labelled`` is the order of the ids.
    """
    def entity(row: Mapping[str, str]) -> dict[str, str]:
        return {field: row[column] for column, field in FIELDS.items()}

    a = {row["id"]: entity(row) for row in table_a}
    b = {row["id"]: entity(row) for row in table_b}
    pairs: list[dict] = []
    for split, rows in labelled:
        for row in rows:
            pairs.append({
                "id": f"c{len(pairs):03d}",
                "entity_a": dict(a[row["ltable_id"]]),
                "entity_b": dict(b[row["rtable_id"]]),
                "known_same_as": row["label"].strip() == "1",
                "split": split,
            })
    return pairs


def load_pairs(cache_dir: str | Path | None = None) -> list[dict]:
    """The 450 candidate pairs, as the cookbook's ``candidate_pairs.json`` has them.

    Each pair is ``{"id", "entity_a", "entity_b", "known_same_as", "split"}``,
    where each entity is ``{"name", "brewery", "style", "abv"}``. ``split`` is
    not in the cookbook's file; it is here so a threshold can be fitted on
    ``train`` and checked on the rest.
    """
    cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    archive = cache / "beer_exp_data.zip"
    if not archive.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        archive.write_bytes(payload)

    with zipfile.ZipFile(archive) as zf:
        def rows(name: str) -> list[dict[str, str]]:
            with zf.open(f"exp_data/{name}.csv") as fh:
                return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", newline="")))

        return pairs_from_tables(rows("tableA"), rows("tableB"),
                                 [(split, rows(split)) for split in SPLITS])
