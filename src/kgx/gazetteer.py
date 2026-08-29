"""Dictionary-based entity linking: resolution against a controlled vocabulary.

Notebooks 01 and 05 resolve entities by *comparing mentions to each other* --
normalise, block, score, cluster. That is the right approach when you do not know
in advance what exists. It has a hard failure mode, measured in notebook 05: no
string metric links ``LHR`` to ``London Heathrow`` (Jaro-Winkler 0.481) or ``SEC``
to ``Securities and Exchange Commission`` (0.696), because they share almost no
characters. Splink needed hand-written SQL to reach those pairs; embeddings need
the surrounding sentence to carry the signal.

But in a great many domains you *do* know what exists. Airports have IATA codes.
Airlines have carrier codes. Products have SKUs. Currencies, countries, ICD
codes, ticker symbols, warehouse locations -- all closed vocabularies, published,
and exact.

Against a closed vocabulary, similarity is the wrong instrument. A dictionary
lookup is exact, instant, explainable, and it links a mention to a *stable
identifier* rather than to a cluster of other mentions -- which means two
documents that never share a spelling still land on the same node, and a node
that appears in no document at all can still be referenced.

This module is deliberately small. The interesting decisions are which surfaces
to index and how to handle a miss, not the lookup itself.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .extract import Mention

__all__ = ["GazetteerEntry", "Gazetteer", "GazetteerMatch"]


def _fold(text: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class GazetteerMatch:
    """One resolved mention."""

    mention_id: str
    key: str                 # the controlled-vocabulary identifier, e.g. "LHR"
    canonical: str           # the preferred display name
    matched_surface: str     # the surface that matched
    how: str                 # "code" | "name" | "alias" | "fuzzy"
    score: float = 1.0

    def as_record(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id, "key": self.key, "canonical": self.canonical,
            "surface": self.matched_surface, "how": self.how, "score": round(self.score, 3),
        }


@dataclass
class GazetteerEntry:
    key: str
    canonical: str
    kind: str = ""
    aliases: tuple[str, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)


class Gazetteer:
    """A controlled vocabulary with an index over every surface that denotes it.

    Built from ``{key: {"name": ..., "kind": ..., "aliases": [...], ...}}``. The
    key itself is indexed too, which is the whole point: ``LHR`` resolves because
    ``LHR`` is the identifier, not because it looks like ``London Heathrow``.

    Matching is tried in decreasing order of confidence -- exact code, exact
    name, exact alias, then optionally fuzzy. The ``how`` field on every match
    records which rule fired, so a graph built this way can be audited by
    resolution mechanism rather than by score alone.
    """

    def __init__(
        self,
        entries: Mapping[str, Mapping[str, Any]] | Iterable[GazetteerEntry],
        *,
        name_key: str = "name",
        kind_key: str = "kind",
        alias_key: str = "aliases",
        case_sensitive_codes: bool = True,
    ) -> None:
        self.entries: dict[str, GazetteerEntry] = {}
        if isinstance(entries, Mapping):
            for key, payload in entries.items():
                self.entries[key] = GazetteerEntry(
                    key=key,
                    canonical=str(payload.get(name_key, key)),
                    kind=str(payload.get(kind_key, "")),
                    aliases=tuple(payload.get(alias_key, ()) or ()),
                    attrs={k: v for k, v in payload.items()
                           if k not in {name_key, kind_key, alias_key}},
                )
        else:
            for entry in entries:
                self.entries[entry.key] = entry

        self.case_sensitive_codes = case_sensitive_codes
        self._codes: dict[str, str] = {}
        self._names: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        for entry in self.entries.values():
            self._codes[entry.key if case_sensitive_codes else entry.key.casefold()] = entry.key
            self._names[_fold(entry.canonical)] = entry.key
            for alias in entry.aliases:
                self._aliases[_fold(alias)] = entry.key

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kinds = sorted({e.kind for e in self.entries.values() if e.kind})
        return f"<Gazetteer {len(self.entries)} entries, kinds={kinds}>"

    def of_kind(self, kind: str) -> list[GazetteerEntry]:
        return [e for e in self.entries.values() if e.kind == kind]

    # -- lookup ----------------------------------------------------------

    def lookup(self, surface: str, *, fuzzy: float | None = None) -> tuple[str, str, float] | None:
        """Resolve one surface to ``(key, how, score)``, or ``None``.

        ``fuzzy`` enables a last-resort similarity pass over canonical names and
        aliases above the given threshold. It is off by default: the value of a
        gazetteer is that a miss is *informative* -- an airport code you have
        never seen is news, and silently snapping it to the nearest known one
        destroys exactly the signal you wanted.
        """
        raw = str(surface).strip()
        if not raw:
            return None

        code = raw if self.case_sensitive_codes else raw.casefold()
        if code in self._codes:
            return (self._codes[code], "code", 1.0)

        folded = _fold(raw)
        if folded in self._names:
            return (self._names[folded], "name", 1.0)
        if folded in self._aliases:
            return (self._aliases[folded], "alias", 1.0)

        # A parenthetical code is the convention half of these corpora use:
        # "London Heathrow (LHR)". Try the inner token before giving up.
        inner = re.search(r"\(([^)]{2,8})\)", raw)
        if inner:
            candidate = inner.group(1).strip()
            key = self._codes.get(candidate if self.case_sensitive_codes else candidate.casefold())
            if key:
                return (key, "code", 1.0)

        if fuzzy is not None:
            from rapidfuzz import fuzz, process

            table = {**self._names, **self._aliases}
            best = process.extractOne(folded, table.keys(), scorer=fuzz.WRatio)
            if best and best[1] / 100 >= fuzzy:
                return (table[best[0]], "fuzzy", best[1] / 100)
        return None

    # -- mention resolution ----------------------------------------------

    def resolve(
        self,
        mentions: Sequence[Mention],
        *,
        types: Sequence[str] | None = None,
        kinds: Mapping[str, str] | None = None,
        fuzzy: float | None = None,
    ) -> tuple[list[GazetteerMatch], list[Mention]]:
        """Split mentions into ``(matched, unmatched)``.

        ``types`` restricts which ontology entity types are looked up at all --
        there is no sense checking a ``disruption`` span against an airport
        table. ``kinds`` optionally maps an entity type to the gazetteer ``kind``
        it must resolve to, so a ``city`` mention cannot silently link to an
        airline that happens to share a name.
        """
        matched: list[GazetteerMatch] = []
        unmatched: list[Mention] = []
        for mention in mentions:
            if types is not None and mention.type not in types:
                unmatched.append(mention)
                continue
            hit = self.lookup(mention.text, fuzzy=fuzzy)
            if hit is None:
                unmatched.append(mention)
                continue
            key, how, score = hit
            entry = self.entries[key]
            required = (kinds or {}).get(mention.type)
            if required is not None and entry.kind != required:
                unmatched.append(mention)
                continue
            matched.append(GazetteerMatch(mention.mention_id, key, entry.canonical,
                                          mention.text, how, score))
        return matched, unmatched

    def coverage(self, mentions: Sequence[Mention], **kwargs: Any) -> dict[str, Any]:
        """How much of a mention set the vocabulary accounts for."""
        matched, unmatched = self.resolve(mentions, **kwargs)
        by_how: dict[str, int] = {}
        for m in matched:
            by_how[m.how] = by_how.get(m.how, 0) + 1
        # Count only the mentions the vocabulary was asked about. `resolve` puts
        # type-excluded mentions in `unmatched` too, and counting those as misses
        # reports the gazetteer failing at a job it was never given.
        types = kwargs.get("types")
        eligible = [m for m in unmatched if types is None or m.type in types]
        considered = len(matched) + len(eligible)
        return {
            "considered": considered,
            "linked": len(matched),
            "coverage": round(len(matched) / considered, 3) if considered else 0.0,
            "by_rule": by_how,
            "distinct_keys": len({m.key for m in matched}),
            "unlinked_surfaces": sorted({m.text for m in eligible})[:12],
        }

    def canonical_map(self, matches: Sequence[GazetteerMatch], prefix: str = "") -> dict[str, str]:
        """``mention_id -> canonical id``, ready for :func:`kgx.graph.build_graph` pinning.

        The id is built from the vocabulary key, not from a cluster, so it is
        stable across corpora and reruns: ``airport:LHR`` means the same node
        next month and in a different document set.
        """
        return {m.mention_id: f"{prefix}{m.key}" for m in matches}
