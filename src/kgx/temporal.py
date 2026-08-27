"""Bi-temporal fact storage: what the agent believes, and when it believed it.

GLiNER2.5 has no temporal model. It reads one text and reports what that text
says. That is the correct division of labour -- but it means a memory graph built
naively from a stream of conversations accumulates contradictions and never
notices. "The user prefers npm" and "the user prefers pnpm" both sit in the graph
as equally valid edges, and the assistant picks one at random on the next turn.

This module is the store layer that fixes it. Every asserted fact carries two
independent timelines, which is the model Graphiti/Zep converge on:

``valid_from`` / ``valid_until``
    When the fact was true *in the world*.
``recorded_at`` / ``superseded_at``
    When the system learned it, and when it learned otherwise.

Keeping both matters because they disagree constantly: an episode on 2026-06-18
can tell you something that was true from 2026-04-01. Collapsing them into one
timestamp makes "what did we believe last March?" unanswerable, which is exactly
the question you need when an agent acted on a fact that later turned out wrong.

Facts are **superseded, never deleted**. An invalidated fact is still the reason
the assistant said what it said last month.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Fact",
    "TemporalGraph",
    "SUPERSEDING_RELATIONS",
    "graph_alternatives",
    "explicit_alternatives",
    "embedding_alternatives",
]


# How a new assertion interacts with what is already believed. Three policies:
#
# "single"       one value per subject. A person is in one place, holds one job
#                title, has one assignee. A new value supersedes the old.
# "alternatives" multi-valued in general, but mutually exclusive within a choice
#                dimension. Preferring pnpm supersedes preferring npm; preferring
#                dark mode supersedes neither. Requires `alternative_fn` to say
#                whether two objects compete -- see `embedding_alternatives`.
# "accumulate"   facts pile up and never conflict. Constraints, skills,
#                collaborators, artifacts.
#
# Getting this table wrong is worse than having no temporal logic at all. Marking
# `prefers` as "single" produces a chain where every new tool preference wipes
# out the last, and the memory ends up holding one arbitrary fact per relation.
SUPERSEDING_RELATIONS: dict[str, str] = {
    "located_in": "single",
    "works_at": "single",
    "assigned_to": "single",
    "part_of": "single",
    "officer_of": "single",
    "subsidiary_of": "single",
    "prefers": "alternatives",
    "avoids": "alternatives",
    "uses_tool": "alternatives",
    "has_constraint": "accumulate",
    "has_skill": "accumulate",
    "collaborates_with": "accumulate",
    "produced": "accumulate",
    "attended": "accumulate",
    "works_on": "accumulate",
    "pursues": "accumulate",
    "blocked_by": "accumulate",
    "replaces": "accumulate",
}


def graph_alternatives(graph: Any, *, min_confidence: float = 0.5):
    """Build an ``alternative_fn`` from ``replaces`` edges the model extracted.

    This is the approach that actually works, and it works because the
    information was in the text all along. "I've switched to pnpm", "we
    standardised on Go instead of Python", "we moved off Jenkins" -- these state
    the supersession explicitly, and a ``replaces(tool, tool)`` relation in the
    ontology lets joint decoding pick it up at the same confidence as any other
    edge (0.95+ on this repo's corpus).

    Two things this avoids. It needs no world knowledge, so it does not care that
    a 194M encoder has never heard of pnpm. And it is evidence-backed: every
    supersession traces to a sentence, which a similarity threshold never can.

    The limit is equally clear -- it only fires when the user *says* they
    switched. A silent drift from one tool to another leaves no ``replaces``
    edge, and those pairs still land in :attr:`TemporalGraph.candidates`.
    """
    pairs: set[tuple[str, str]] = set()
    for edge in getattr(graph, "edges", []):
        if edge.type != "replaces" or edge.confidence < min_confidence:
            continue
        head = graph.name(edge.head).casefold()
        tail = graph.name(edge.tail).casefold()
        pairs.add((head, tail))
        pairs.add((tail, head))

    def is_alternative(a: str, b: str) -> bool:
        return (a.casefold(), b.casefold()) in pairs

    return is_alternative


def explicit_alternatives(groups: Sequence[Sequence[str]]):
    """Build an ``alternative_fn`` from a curated equivalence table.

    Deterministic, auditable, and what you would actually ship for a domain you
    know::

        explicit_alternatives([
            ["npm", "pnpm", "yarn", "bun"],
            ["Python", "Go", "Rust", "Java"],
        ])
    """
    lookup: dict[str, int] = {}
    for i, group in enumerate(groups):
        for name in group:
            lookup[name.casefold()] = i

    def is_alternative(a: str, b: str) -> bool:
        ga, gb = lookup.get(a.casefold()), lookup.get(b.casefold())
        return ga is not None and ga == gb

    return is_alternative


def embedding_alternatives(encoder: Any = None, threshold: float = 0.55):
    """Similarity-based ``alternative_fn``. **Measured here as not good enough.**

    The idea is reasonable: npm and pnpm are both package managers and should sit
    close in embedding space, while npm and Berlin should not. Included because
    it is the first thing most people reach for, and because the measurement is
    worth seeing.

    On this repo's vocabulary with ``all-MiniLM-L6-v2``, true alternatives and
    unrelated pairs **overlap** -- the lowest-scoring genuine alternative pair
    falls below the highest-scoring unrelated pair, under every prompt template
    tried. There is no threshold that separates them. Bare product names carry
    too little signal, and a switch from npm to pnpm looks much like a mention of
    npm and Dev Shah.

    Prefer :func:`graph_alternatives` (the model extracted the switch from the
    text) or :func:`explicit_alternatives` (you wrote the table down). Reach for
    an LLM adjudicator when neither applies -- deciding that pnpm competes with
    npm is a world-knowledge question, and a 194M encoder does not hold that
    knowledge.
    """
    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cache: dict[str, Any] = {}

    def similar(a: str, b: str) -> bool:
        for name in (a, b):
            if name not in cache:
                cache[name] = encoder.encode(name, normalize_embeddings=True)
        return float(cache[a] @ cache[b]) >= threshold

    return similar


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Fact:
    """One assertion, with both timelines and a pointer to its source."""

    head: str                       # canon_id
    relation: str
    tail: str                       # canon_id
    head_name: str = ""
    tail_name: str = ""
    tail_type: str = ""
    confidence: float = 1.0
    support: int = 1
    episode_id: str = ""            # which conversation/document asserted it
    valid_from: str | None = None   # true in the world from
    valid_until: str | None = None  # true in the world until
    recorded_at: str = field(default_factory=_now)
    superseded_at: str | None = None
    superseded_by: str | None = None
    evidence: list[str] = field(default_factory=list)

    @property
    def fact_id(self) -> str:
        return f"{self.head}|{self.relation}|{self.tail}|{self.episode_id}"

    @property
    def is_current(self) -> bool:
        return self.superseded_at is None

    def as_record(self) -> dict[str, Any]:
        return {
            "head": self.head_name or self.head,
            "relation": self.relation,
            "tail": self.tail_name or self.tail,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "episode": self.episode_id,
            "confidence": round(self.confidence, 3),
            "current": self.is_current,
            "superseded_by": self.superseded_by,
        }


@dataclass
class TemporalGraph:
    """An append-only fact log with contradiction detection on write."""

    facts: list[Fact] = field(default_factory=list)
    superseding: Mapping[str, str] = field(default_factory=lambda: dict(SUPERSEDING_RELATIONS))
    alternative_fn: Any = None
    candidates: list[tuple[Fact, Fact]] = field(default_factory=list)
    _seen_candidates: set = field(default_factory=set, repr=False, compare=False)

    # -- write path ------------------------------------------------------

    def _conflicts(self, existing: Fact, fact: Fact) -> bool:
        """Does ``fact`` invalidate ``existing``?"""
        if existing.head != fact.head or existing.relation != fact.relation:
            return False
        if existing.tail == fact.tail:
            return False
        policy = self.superseding.get(fact.relation, "accumulate")
        if policy == "single":
            return True
        if policy != "alternatives":
            return False
        if existing.tail_type != fact.tail_type:
            return False
        if self.alternative_fn is None:
            return False  # no basis to judge; recorded as a candidate instead
        return bool(self.alternative_fn(existing.tail_name or existing.tail,
                                        fact.tail_name or fact.tail))

    def assert_fact(self, fact: Fact) -> list[Fact]:
        """Record a fact, superseding any it contradicts. Returns what it replaced.

        A repeat of something already believed is corroboration, not a new fact:
        it bumps ``support`` and refreshes ``valid_until`` instead of appending.

        When a relation is "alternatives" but no ``alternative_fn`` is configured,
        the competing pair is logged to :attr:`candidates` rather than acted on.
        Surfacing an unresolved conflict beats silently picking one.
        """
        superseded: list[Fact] = []

        for existing in self.facts:
            if not existing.is_current:
                continue
            if (existing.head, existing.relation, existing.tail) == (
                fact.head, fact.relation, fact.tail
            ):
                # Corroboration, not a new fact. Crucially this must NOT set
                # valid_until -- closing the window on a fact the graph still
                # believes would make as_of() drop it from every later date.
                existing.support += fact.support
                existing.confidence = max(existing.confidence, fact.confidence)
                existing.last_seen = fact.valid_from or fact.recorded_at
                existing.evidence = list(dict.fromkeys([*existing.evidence, *fact.evidence]))
                return superseded
            if self._conflicts(existing, fact):
                # Two timelines, two different stamps. `valid_until` is world
                # time (when the old fact stopped being true); `superseded_at`
                # is transaction time (when we learned that). Using world time
                # for both inverts the transaction timeline whenever an episode
                # reports something backdated.
                existing.valid_until = fact.valid_from or existing.valid_until
                existing.superseded_at = fact.recorded_at
                existing.superseded_by = fact.fact_id
                superseded.append(existing)
            elif (
                existing.head == fact.head
                and existing.relation == fact.relation
                and existing.tail != fact.tail
                and self.superseding.get(fact.relation) == "alternatives"
                and self.alternative_fn is None
            ):
                key = (existing.head, existing.relation, existing.tail, fact.tail)
                if key not in self._seen_candidates:
                    self._seen_candidates.add(key)
                    self.candidates.append((existing, fact))

        self.facts.append(fact)
        return superseded

    def ingest_graph(
        self,
        graph: Any,
        *,
        episode_id: str = "",
        valid_from: str | None = None,
        min_confidence: float = 0.0,
    ) -> dict[str, int]:
        """Fold a :class:`~kgx.graph.KnowledgeGraph` into the fact log.

        Call once per episode, in chronological order -- supersession is
        order-dependent by construction.
        """
        added = superseded = corroborated = 0
        for edge in graph.edges:
            if edge.confidence < min_confidence:
                continue
            head = graph.entities.get(edge.head)
            tail = graph.entities.get(edge.tail)
            if head is None or tail is None:
                continue
            fact = Fact(
                head=edge.head,
                relation=edge.type,
                tail=edge.tail,
                head_name=head.canonical,
                tail_name=tail.canonical,
                tail_type=tail.type,
                confidence=edge.confidence,
                support=edge.support,
                episode_id=episode_id,
                valid_from=valid_from,
                evidence=[ev.snippet for ev in edge.evidence[:3]],
            )
            before = len(self.facts)
            killed = self.assert_fact(fact)
            if len(self.facts) == before:
                corroborated += 1
            else:
                added += 1
            superseded += len(killed)
        return {"added": added, "superseded": superseded, "corroborated": corroborated}

    # -- read path -------------------------------------------------------

    def current(self, *, head: str | None = None, relation: str | None = None) -> list[Fact]:
        """What the agent believes right now."""
        return [
            f for f in self.facts
            if f.is_current
            and (head is None or f.head == head or f.head_name == head)
            and (relation is None or f.relation == relation)
        ]

    def as_of(
        self, when: str, *, head: str | None = None, timeline: str = "valid"
    ) -> list[Fact]:
        """State at a point in time, on either timeline.

        ``timeline="valid"``
            What was **true in the world** then (``valid_from`` / ``valid_until``).
            Use this to reconstruct the user's situation at a past date.
        ``timeline="transaction"``
            What the system **believed** then (``recorded_at`` / ``superseded_at``).
            Use this to explain a decision the agent made at the time -- it may
            have acted on a fact that has since been corrected.

        The two answers differ whenever an episode reports something that was
        already true before it was mentioned, which is most of the time. That
        divergence is the reason to keep both.
        """
        if timeline not in {"valid", "transaction"}:
            raise ValueError("timeline must be 'valid' or 'transaction'")

        def in_window(f: Fact) -> bool:
            if timeline == "transaction":
                start, end = f.recorded_at, f.superseded_at
            else:
                start, end = (f.valid_from or f.recorded_at), f.valid_until
            return start <= when and (end is None or end > when)

        return [
            f for f in self.facts
            if in_window(f) and (head is None or f.head == head or f.head_name == head)
        ]

    def history(self, head: str, relation: str) -> list[Fact]:
        """Every assertion for a subject/relation, oldest first."""
        return sorted(
            (f for f in self.facts if (f.head == head or f.head_name == head) and f.relation == relation),
            key=lambda f: (f.valid_from or f.recorded_at, f.recorded_at, f.tail),
        )

    def contradictions(self) -> list[tuple[Fact, Fact]]:
        """Superseded/superseding pairs -- what the agent changed its mind about."""
        by_id = {f.fact_id: f for f in self.facts}
        return [
            (old, by_id[old.superseded_by])
            for old in self.facts
            if old.superseded_by and old.superseded_by in by_id
        ]

    def candidate_conflicts(self):
        """Competing assertions no rule was confident enough to resolve.

        This is the review queue -- the place an LLM adjudicator or a human
        belongs. It is populated only when a relation is "alternatives" and no
        ``alternative_fn`` was configured.
        """
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "subject": old.head_name or old.head,
                    "relation": old.relation,
                    "existing": old.tail_name or old.tail,
                    "since": old.valid_from,
                    "new": new.tail_name or new.tail,
                    "asserted": new.valid_from,
                    "same_type": old.tail_type == new.tail_type,
                }
                for old, new in self.candidates
            ]
        )

    def frame(self, *, current_only: bool = False):
        import pandas as pd

        rows = [f.as_record() for f in self.facts if f.is_current or not current_only]
        return pd.DataFrame(rows)

    def memory_prompt(self, subject: str, *, limit: int = 40) -> str:
        """The graph rendered as prompt context, which is the whole point.

        Current facts only, grouped by relation, with superseded beliefs left out
        -- the agent should not be told the user prefers npm when they switched to
        pnpm three months ago.
        """
        facts = [f for f in self.current() if f.head_name == subject or f.head == subject]
        if not facts:
            return f"No stored facts about {subject}."
        groups: dict[str, list[Fact]] = {}
        for f in sorted(facts, key=lambda f: -f.confidence)[:limit]:
            groups.setdefault(f.relation, []).append(f)
        lines = [f"Known facts about {subject}:"]
        for relation, items in sorted(groups.items()):
            values = ", ".join(dict.fromkeys(i.tail_name or i.tail for i in items))
            lines.append(f"- {relation.replace('_', ' ')}: {values}")
        return "\n".join(lines)
