"""Scoring extracted graphs against gold facts.

Everything in this repo up to now has been demonstrated rather than measured.
Five gold datasets ship with the sample corpora and only one of them
(``ALIAS_GROUPS``, via B-cubed) was ever scored. This module is what the rest
get scored with.

The hard part is not the arithmetic, it is deciding when two triples are the
same triple. ``("Northwind Logistics Inc.", "acquires", "Cascade Freight
Systems")`` and ``("Northwind Logistics Inc", "acquires", "Cascade")`` refer to
the same fact, and an evaluator that calls them different is measuring string
formatting rather than extraction. The matching policy here is deliberately
explicit and deliberately conservative:

1. **Relations must match exactly.** They come from a closed vocabulary the
   ontology defines; a fuzzy relation match would be measuring nothing.
2. **Entity names match after normalisation** -- the same
   :func:`kgx.resolve.normalize` used during resolution, so legal suffixes,
   determiners and case are ignored.
3. **Optionally, names match by alias or by fuzzy similarity** above a stated
   threshold, and every such match is *recorded* so the loosening can be audited
   rather than assumed harmless.
4. **Matching is one-to-one.** A single predicted triple cannot satisfy two gold
   triples, which is the difference between precision and wishful thinking.

Report both the strict and the lenient number when they differ. If a comparison
between two extractors flips depending on which policy you use, that is a
finding about the comparison, not a detail to bury.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .resolve import normalize

__all__ = [
    "Triple",
    "MatchPolicy",
    "TripleScore",
    "score_triples",
    "score_entities",
    "graph_triples",
    "comparison_table",
]

Triple = tuple[str, str, str]


@dataclass(frozen=True)
class MatchPolicy:
    """How forgiving triple matching should be, stated rather than implied."""

    normalize_names: bool = True
    use_aliases: bool = True
    fuzzy_threshold: float | None = None   # e.g. 0.90; None disables fuzzy matching
    directed: bool = True                  # False also accepts a reversed triple

    @property
    def label(self) -> str:
        if self.fuzzy_threshold is not None:
            return f"lenient (fuzzy>={self.fuzzy_threshold:g})"
        if self.use_aliases:
            return "normalised + aliases"
        return "strict"


def _key(name: str, type_hint: str = "") -> str:
    """Normalise a triple endpoint for matching.

    A triple is three strings; unlike a :class:`~kgx.extract.Mention` it carries
    no type, so the type-specific rules in :func:`kgx.resolve.normalize` -- legal
    suffixes for organisations, titles for people -- would never fire. For
    scoring we want both: ``"Acme Corp."`` should match ``"Acme"`` and
    ``"Dr. Elena Vasquez"`` should match ``"Elena Vasquez"``.

    So normalise under both hypotheses and take the shortest result. Relation
    matching is already exact, which is what keeps this from being too generous:
    collapsing two endpoint spellings can only merge triples that agree on the
    relation and on the other endpoint.
    """
    if type_hint:
        return normalize(name, type_hint).key
    candidates = [normalize(name, hint).key for hint in ("", "company", "person")]
    return min(candidates, key=lambda k: (len(k), k))


@dataclass
class TripleScore:
    """Precision / recall / F1 plus the triples behind each number."""

    matched: list[tuple[Triple, Triple]] = field(default_factory=list)
    spurious: list[Triple] = field(default_factory=list)   # predicted, no gold match
    missed: list[Triple] = field(default_factory=list)     # gold, never predicted
    loosened: list[tuple[Triple, Triple, str]] = field(default_factory=list)
    policy: MatchPolicy = field(default_factory=MatchPolicy)
    n_predicted: int = 0
    n_gold: int = 0

    @property
    def precision(self) -> float:
        return len(self.matched) / self.n_predicted if self.n_predicted else 0.0

    @property
    def recall(self) -> float:
        return len(self.matched) / self.n_gold if self.n_gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_record(self) -> dict[str, Any]:
        return {
            "policy": self.policy.label,
            "predicted": self.n_predicted,
            "gold": self.n_gold,
            "matched": len(self.matched),
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
        }

    def report(self, limit: int = 8) -> str:
        lines = [
            f"precision {self.precision:.3f}  recall {self.recall:.3f}  f1 {self.f1:.3f}"
            f"   ({len(self.matched)}/{self.n_gold} gold, {self.n_predicted} predicted"
            f", policy: {self.policy.label})"
        ]
        if self.missed:
            lines.append(f"\n  missed ({len(self.missed)}):")
            lines += [f"    {h} -{r}-> {t}" for h, r, t in self.missed[:limit]]
        if self.spurious:
            lines.append(f"\n  spurious ({len(self.spurious)}):")
            lines += [f"    {h} -{r}-> {t}" for h, r, t in self.spurious[:limit]]
        if self.loosened:
            lines.append(f"\n  matched only by loosening ({len(self.loosened)}):")
            lines += [f"    {p} ~ {g}  [{why}]" for p, g, why in self.loosened[:limit]]
        return "\n".join(lines)


def _alias_index(aliases: Mapping[str, Sequence[str]] | None) -> dict[str, str]:
    """surface-key -> canonical-key, from a {canonical: [aliases]} mapping."""
    index: dict[str, str] = {}
    for canonical, variants in (aliases or {}).items():
        canon_key = _key(canonical)
        index[canon_key] = canon_key
        for variant in variants:
            index[_key(variant)] = canon_key
    return index


def score_triples(
    predicted: Iterable[Triple],
    gold: Iterable[Triple],
    *,
    policy: MatchPolicy | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> TripleScore:
    """Score predicted triples against gold, one-to-one.

    ``aliases`` maps a canonical name to its known surface variants (the corpora
    ship this as ``ALIAS_GROUPS``); supplying it lets a prediction that used a
    different-but-known name still count.
    """
    policy = policy or MatchPolicy()
    predicted = [tuple(t) for t in predicted]
    gold = [tuple(t) for t in gold]
    index = _alias_index(aliases) if policy.use_aliases else {}

    def canon(name: str) -> str:
        key = _key(name) if policy.normalize_names else name
        return index.get(key, key)

    score = TripleScore(policy=policy, n_predicted=len(predicted), n_gold=len(gold))
    remaining = list(range(len(gold)))
    gold_keys = [(canon(h), r, canon(t)) for h, r, t in gold]

    from rapidfuzz import fuzz

    for p in predicted:
        ph, pr, pt = canon(p[0]), p[1], canon(p[2])
        hit: int | None = None
        why = ""

        for i in remaining:                                   # exact, after canonicalisation
            gh, gr, gt = gold_keys[i]
            if pr != gr:
                continue
            if (ph, pt) == (gh, gt) or (not policy.directed and (ph, pt) == (gt, gh)):
                hit = i
                break

        if hit is None and policy.fuzzy_threshold is not None:
            best, best_score = None, 0.0
            for i in remaining:
                gh, gr, gt = gold_keys[i]
                if pr != gr:
                    continue
                sim = min(fuzz.token_set_ratio(ph, gh), fuzz.token_set_ratio(pt, gt)) / 100
                if sim >= policy.fuzzy_threshold and sim > best_score:
                    best, best_score = i, sim
            if best is not None:
                hit, why = best, f"fuzzy {best_score:.2f}"

        if hit is None:
            score.spurious.append(p)
            continue

        remaining.remove(hit)
        score.matched.append((p, gold[hit]))
        if why:
            score.loosened.append((p, gold[hit], why))

    score.missed = [gold[i] for i in remaining]
    return score


def score_entities(
    predicted: Iterable[tuple[str, str]],
    gold: Iterable[tuple[str, str]],
    *,
    typed: bool = True,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> TripleScore:
    """Score ``(name, type)`` pairs. With ``typed=False`` the type is ignored.

    Useful for comparing a closed-vocabulary tagger (spaCy's 18 OntoNotes labels)
    against an open-vocabulary one, where getting the *span* right and the *type*
    right are worth separating.
    """
    p = [(n, t if typed else "_", "") for n, t in predicted]
    g = [(n, t if typed else "_", "") for n, t in gold]
    return score_triples(p, g, policy=MatchPolicy(), aliases=aliases)


def graph_triples(
    graph: Any, *, min_confidence: float = 0.0, min_support: int = 1,
    drop_relations: Sequence[str] = (),
) -> list[Triple]:
    """Canonical-name triples out of a :class:`~kgx.graph.KnowledgeGraph`."""
    drop = set(drop_relations)
    return [
        (graph.name(e.head), e.type, graph.name(e.tail))
        for e in graph.edges
        if e.confidence >= min_confidence and e.support >= min_support
        and e.type not in drop
    ]


def comparison_table(scores: Mapping[str, TripleScore], sort_by: str = "f1"):
    """One row per system, for a head-to-head table."""
    import pandas as pd

    rows = []
    for name, score in scores.items():
        rows.append({"system": name, **score.as_record()})
    frame = pd.DataFrame(rows).set_index("system")
    return frame.sort_values(sort_by, ascending=False)
