"""Entity resolution: many document-local mentions -> one canonical node set.

Every :class:`~kgx.extract.DocGraph` is document-local. ``d01:e3`` and ``d07:e1``
may be the same real-world company under two spellings, and inside a single
document the same company usually appears several times, once per sentence. That
duplication is not noise to be tolerated -- it is the difference between a graph
with 400 nodes and a graph with 60, and between an edge count that means
something and one that just counts sentences.

The pipeline is the conventional five stages, kept separate so each can be
measured on its own:

    normalize -> block -> score -> cluster -> canonicalize

Two properties are worth defending:

*Every scored pair is retained*, with its component scores, before any
clustering decision. Re-clustering at a different threshold is then free, which
is what makes a threshold sweep possible at all. Re-scoring is the expensive
part.

*Blocking recall is the hard ceiling on the whole system.* A pair that never
enters the candidate set can never be matched, and no downstream tuning recovers
it. :meth:`EntityResolver.blocking_report` measures it separately from
end-to-end quality for exactly that reason.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .extract import Mention

__all__ = [
    "NormalizedName",
    "normalize",
    "ScoredPair",
    "CanonicalEntity",
    "Resolution",
    "EntityResolver",
    "CanonicalRegistry",
    "bcubed",
    "detect_aliases",
    "is_acronym_of",
    "EMBED_MODEL",
]

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 90MB, CPU-fine

# Stripped into a separate field rather than deleted, so "Acme Inc." can still be
# preferred over "Acme" as the canonical display form.
LEGAL_SUFFIXES = {
    "inc", "inc.", "incorporated", "corp", "corp.", "corporation", "co", "co.",
    "company", "llc", "l.l.c.", "ltd", "ltd.", "limited", "plc", "lp", "llp",
    "nv", "n.v.", "sa", "s.a.", "ag", "gmbh", "bv", "b.v.", "ab", "as", "oy",
    "pte", "pty", "holdings", "group", "partners",
}
PERSON_TITLES = {
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "dr", "dr.", "prof", "prof.",
    "sir", "dame", "rev", "rev.", "hon", "hon.",
}
PERSON_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "phd", "ph.d.", "md", "m.d."}
DETERMINERS = {"the", "a", "an", "this", "that", "our", "their", "its", "his", "her", "my"}

ORG_TYPES = {"organization", "company", "regulator", "team"}
PERSON_TYPES = {"person", "user"}


@dataclass(frozen=True)
class NormalizedName:
    """The normalised form of one mention surface, plus the parts ER needs."""

    key: str                  # casefolded, punctuation-light matching key
    display: str              # cleaned but case-preserving surface
    tokens: tuple[str, ...]
    suffix: str = ""          # legal suffix stripped from an org name
    first: str = ""           # person given name
    last: str = ""            # person family name
    is_acronym: bool = False
    entropy: float = 0.0      # character entropy; low => string similarity is unreliable

    @property
    def initials(self) -> str:
        return "".join(t[0] for t in self.tokens if t)


def _entropy(text: str) -> float:
    """Shannon entropy over characters, in bits.

    Short, repetitive, or acronym-like names produce spuriously high fuzzy
    similarity to each other ("NWL" vs "NWT"). Below roughly 2.5 bits, do not let
    the string score carry a merge decision on its own.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def normalize(text: str, type_: str = "") -> NormalizedName:
    """Deterministic, model-free normalisation. Stage 1 of the pipeline.

    Exact match on ``(type, key)`` after this stage typically collapses a large
    fraction of mentions at zero cost -- always run it before anything expensive.
    """
    raw = unicodedata.normalize("NFKC", text)
    raw = re.sub(r"[​-‏﻿]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(r"^[\"'“‘(\[]+|[\"'”’)\]]+$", "", raw).strip()
    raw = re.sub(r"['’]s\b", "", raw)  # possessives

    tokens = [t for t in re.split(r"[\s,]+", raw) if t]
    while tokens and tokens[0].casefold() in DETERMINERS:
        tokens.pop(0)

    suffix = first = last = ""
    if type_ in ORG_TYPES:
        while len(tokens) > 1 and tokens[-1].casefold().strip(".,") in LEGAL_SUFFIXES:
            suffix = tokens.pop().strip(".,") + (" " + suffix if suffix else "")
    elif type_ in PERSON_TYPES:
        while tokens and tokens[0].casefold() in PERSON_TITLES:
            tokens.pop(0)
        while tokens and tokens[-1].casefold() in PERSON_SUFFIXES:
            tokens.pop()
        if tokens:
            first, last = tokens[0], tokens[-1]

    display = " ".join(tokens) if tokens else raw
    key = re.sub(r"[^\w\s]", "", display.casefold()).strip()
    key = re.sub(r"\s+", " ", key)
    stripped = "".join(tokens)
    is_acronym = bool(stripped) and stripped.isupper() and 2 <= len(stripped) <= 6

    return NormalizedName(
        key=key or display.casefold(),
        display=display or raw,
        tokens=tuple(t.casefold() for t in tokens),
        suffix=suffix,
        first=first,
        last=last,
        is_acronym=is_acronym,
        entropy=_entropy(key),
    )


ALIAS_PATTERNS = [
    # Northwind Logistics Inc. (NASDAQ: NWL)  /  Halcyon Semiconductor (NYSE: HLCN)
    re.compile(r"([A-Z][\w&.\-']*(?:\s+[A-Z][\w&.\-']*){0,5})\s*\(\s*(?:NASDAQ|NYSE|LSE|AMEX|TSX|ASX|OTC|Nasdaq|Ticker|TICKER)\s*:\s*([A-Z]{1,6})\s*\)"),
    # Northwind Logistics Inc. ("Northwind")  /  Acme Corp (the "Company")
    re.compile(r"([A-Z][\w&.\-']*(?:\s+[A-Z][\w&.\-']*){0,5})\s*\(\s*(?:the\s+)?[\"\u201c]([^\"\u201d)]{2,40})[\"\u201d]\s*\)"),
    # Northwind Logistics Inc. (NWL)
    re.compile(r"([A-Z][\w&.\-']*(?:\s+[A-Z][\w&.\-']*){1,5})\s*\(\s*([A-Z]{2,6})\s*\)"),
]


def detect_aliases(texts: Iterable[str]) -> set[tuple[str, str]]:
    """Mine explicit alias declarations out of the corpus itself.

    Business and legal writing introduces its own abbreviations, and the
    convention is machine-readable: ``Northwind Logistics Inc. (NASDAQ: NWL)``,
    ``Acme Corp (the "Company")``. Reading those gives an exact,
    evidence-backed alias table for free, and it solves the case string
    similarity cannot -- ``NWL`` and ``Northwind Logistics`` share almost no
    characters, so no fuzzy threshold will ever link them, and lowering the
    threshold far enough to try would merge half the corpus.

    An alias learned in one document is applied corpus-wide, which is the point:
    the press release that spells out the ticker teaches every later document
    what a bare ``NWL`` means.

    Returns normalised ``(long_form_key, short_form_key)`` pairs.
    """
    out: set[tuple[str, str]] = set()
    for text in texts:
        for pattern in ALIAS_PATTERNS:
            for match in pattern.finditer(text):
                long_form = normalize(match.group(1), "company")
                short_form = normalize(match.group(2), "company")
                if not long_form.key or not short_form.key:
                    continue
                if long_form.key == short_form.key:
                    continue
                # "the Company" is a role, not a name; skip generic placeholders
                if short_form.key in {"company", "group", "corporation", "issuer", "bank"}:
                    continue
                out.add((long_form.key, short_form.key))
    return out


def is_acronym_of(acronym: str, full_tokens: Sequence[str]) -> bool:
    """Does ``acronym`` abbreviate these tokens, allowing word-internal letters?

    ``NWL`` for *Northwind Logistics* takes the ``W`` from inside the first word,
    so first-letters-only matching misses it. Greedy subsequence matching over
    the concatenated name catches it, with two guards against the obvious false
    positives: the first letters must agree, and at least one later acronym
    letter must begin a token.
    """
    acronym = re.sub(r"[^a-z]", "", acronym.casefold())
    tokens = [re.sub(r"[^a-z]", "", t.casefold()) for t in full_tokens]
    tokens = [t for t in tokens if t]
    if len(acronym) < 2 or not tokens:
        return False
    if acronym[0] != tokens[0][0]:
        return False
    joined = "".join(tokens)
    pos = 0
    for ch in acronym:
        found = joined.find(ch, pos)
        if found < 0:
            return False
        pos = found + 1
    if len(tokens) == 1:
        return len(acronym) >= 3
    starts = {t[0] for t in tokens[1:]}
    return any(ch in starts for ch in acronym[1:])


@dataclass
class ScoredPair:
    """One candidate merge, with every component score kept for later re-analysis."""

    a: str
    b: str
    type: str
    block: str          # which blocker proposed it
    fuzz: float = 0.0
    embed: float = 0.0
    context: float = 0.0
    rule: float = 0.0   # deterministic evidence: acronym expansion, exact key, ...
    score: float = 0.0
    reason: str = ""

    def as_record(self) -> dict[str, Any]:
        return {
            "a": self.a, "b": self.b, "type": self.type, "block": self.block,
            "fuzz": round(self.fuzz, 3), "embed": round(self.embed, 3),
            "context": round(self.context, 3), "rule": round(self.rule, 3),
            "score": round(self.score, 3), "reason": self.reason,
        }


@dataclass
class CanonicalEntity:
    """One resolved real-world entity: a cluster of mentions with a stable id."""

    canon_id: str
    type: str
    canonical: str
    aliases: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    first_seen: str | None = None
    last_seen: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "canon_id": self.canon_id, "type": self.type, "canonical": self.canonical,
            "n_mentions": len(self.mentions), "n_docs": len(set(self.docs)),
            "aliases": ", ".join(sorted(self.aliases)),
        }


@dataclass
class Resolution:
    """The output of a resolution run."""

    entities: dict[str, CanonicalEntity]
    mention_to_canon: dict[str, str]
    pairs: list[ScoredPair]
    threshold: float
    stats: dict[str, Any] = field(default_factory=dict)

    def canon_for(self, mention_id: str) -> str:
        return self.mention_to_canon[mention_id]

    def frame(self):
        import pandas as pd

        return pd.DataFrame([e.as_record() for e in self.entities.values()]).sort_values(
            ["type", "n_mentions"], ascending=[True, False]
        )

    def pairs_frame(self):
        import pandas as pd

        return pd.DataFrame([p.as_record() for p in self.pairs])

    def clusters(self) -> list[set[str]]:
        return [set(e.mentions) for e in self.entities.values()]

    def review_band(self, low: float = 0.85, high: float | None = None) -> list[ScoredPair]:
        """Pairs too close to call. This is the slot an LLM adjudicator fills.

        Ambiguity is data. Emitting these as ``same_as`` candidates for review
        beats guessing, and beats silently dropping them.
        """
        high = self.threshold if high is None else high
        return sorted(
            (p for p in self.pairs if low <= p.score < high), key=lambda p: -p.score
        )

    def oversized(self, limit: int = 15) -> list[CanonicalEntity]:
        """Clusters big enough to suspect a transitive over-merge."""
        return [e for e in self.entities.values() if len(e.mentions) > limit]


def _slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.casefold()).strip()
    return re.sub(r"[\s_]+", "-", s) or "unnamed"


class EntityResolver:
    """Blocking + scoring + clustering over a mention set.

    Weights and thresholds are constructor arguments on purpose: they are the
    knobs a reader should move. The default auto-merge line of 0.90 is what the
    threshold sweep in the notebook lands on for this corpus (B-cubed F1 0.95);
    compliance-grade matching wants something nearer 0.98, and casual content
    tolerates 0.85. There is no universal value -- sweep it on your own data,
    which costs nothing because every pair score is retained.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.90,
        w_fuzz: float = 0.45,
        w_embed: float = 0.40,
        w_context: float = 0.15,
        embed_model: str | None = EMBED_MODEL,
        top_k: int = 12,
        min_entropy: float = 2.5,
        low_entropy_penalty: float = 0.25,
        use_embeddings: bool = True,
        aliases: Iterable[tuple[str, str]] = (),
    ) -> None:
        self.threshold = threshold
        self.w_fuzz, self.w_embed, self.w_context = w_fuzz, w_embed, w_context
        self.embed_model_name = embed_model
        self.top_k = top_k
        self.min_entropy = min_entropy
        self.low_entropy_penalty = low_entropy_penalty
        self.use_embeddings = use_embeddings
        self.aliases = {frozenset(pair) for pair in aliases}
        self._encoder = None

    def learn_aliases(self, texts: Iterable[str]) -> "EntityResolver":
        """Mine alias declarations from a corpus and add them to the table."""
        self.aliases |= {frozenset(pair) for pair in detect_aliases(texts)}
        return self

    # -- embeddings ------------------------------------------------------

    @property
    def encoder(self):
        """Lazily loaded. Falls back to string-only scoring if unavailable."""
        if self._encoder is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.embed_model_name)
            except Exception as exc:  # pragma: no cover - offline path
                import warnings

                warnings.warn(
                    f"embedding model unavailable ({exc}); falling back to string-only "
                    f"scoring. Recall on non-string-similar aliases will drop.",
                    RuntimeWarning,
                )
                self.use_embeddings = False
        return self._encoder

    def _embed(self, mentions: Sequence[Mention], norms: Mapping[str, NormalizedName]):
        if not self.use_embeddings or self.encoder is None:
            return None
        import numpy as np

        # Type + name + local context. Context is what separates "Apple" the
        # company from "Apple" the product; name alone cannot.
        keys = [
            f"{m.type}: {norms[m.mention_id].display}. {m.context[:180]}"
            for m in mentions
        ]
        return np.asarray(
            self.encoder.encode(keys, normalize_embeddings=True, show_progress_bar=False)
        )

    # -- stage 2: blocking ----------------------------------------------

    def block(
        self,
        mentions: Sequence[Mention],
        norms: Mapping[str, NormalizedName],
        embeddings: Any = None,
    ) -> dict[tuple[str, str], str]:
        """Propose candidate pairs. Returns ``{(a, b): blocker_name}``.

        Blocking is always type-constrained: "Apple" the company and "Apple" the
        product embed almost identically, and restricting candidate generation to
        same-type mentions is the cheapest precision win available.
        """
        by_type: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(mentions):
            by_type[m.type].append(i)

        pairs: dict[tuple[str, str], str] = {}

        def add(i: int, j: int, tag: str) -> None:
            a, b = mentions[i].mention_id, mentions[j].mention_id
            key = (a, b) if a < b else (b, a)
            pairs.setdefault(key, tag)

        # (a) deterministic key blocking: shared head token, or shared initials
        for _type, idxs in by_type.items():
            buckets: dict[str, list[int]] = defaultdict(list)
            for i in idxs:
                n = norms[mentions[i].mention_id]
                if n.tokens:
                    buckets[f"head:{n.tokens[0]}"].append(i)
                    buckets[f"tail:{n.tokens[-1]}"].append(i)
                if n.last:
                    buckets[f"last:{n.last.casefold()}"].append(i)
                buckets[f"init:{n.initials.casefold()}"].append(i)
                buckets[f"exact:{n.key}"].append(i)
                # A declared alias is a blocking key in its own right, so
                # `learn_aliases` still fires when embeddings are unavailable.
                for pair in self.aliases:
                    if n.key in pair:
                        buckets[f"alias:{'|'.join(sorted(pair))}"].append(i)
            for tag, members in buckets.items():
                # A huge bucket proposes O(n^2) useless pairs and blocks nothing
                # -- except an `exact:` bucket, where every pair IS a match and
                # skipping it would leave the most-repeated names unmerged.
                if len(members) > 60 and not tag.startswith("exact:"):
                    continue
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        add(members[x], members[y], tag.split(":")[0])

        # (b) embedding ANN blocking -- the workhorse for aliases that share no
        #     tokens at all ("IBM" / "International Business Machines").
        if embeddings is not None:
            import numpy as np

            for _type, idxs in by_type.items():
                if len(idxs) < 2:
                    continue
                sub = embeddings[idxs]
                sims = sub @ sub.T
                k = min(self.top_k + 1, len(idxs))
                top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
                for row, cols in enumerate(top):
                    for col in cols:
                        if row != col:
                            add(idxs[row], idxs[int(col)], "ann")
        return pairs

    # -- stage 3: scoring ------------------------------------------------

    def score_pair(
        self,
        ma: Mention,
        mb: Mention,
        na: NormalizedName,
        nb: NormalizedName,
        embed: float,
        block: str,
    ) -> ScoredPair:
        from rapidfuzz import fuzz

        p = ScoredPair(a=ma.mention_id, b=mb.mention_id, type=ma.type, block=block, embed=embed)

        # deterministic evidence first -- it is free and beats any learned score
        if na.key == nb.key:
            p.rule, p.reason, p.score = 1.0, "exact normalised match", 1.0
            p.fuzz = 1.0
            return p
        if frozenset((na.key, nb.key)) in self.aliases:
            p.rule, p.reason, p.score = 1.0, "alias declared in corpus text", 1.0
            p.fuzz = fuzz.WRatio(na.key, nb.key) / 100
            return p
        if na.is_acronym != nb.is_acronym:
            acro, full = (na, nb) if na.is_acronym else (nb, na)
            if is_acronym_of(acro.key, full.tokens):
                p.rule, p.reason, p.score = 1.0, "acronym expansion", 0.97
                p.fuzz = fuzz.WRatio(na.key, nb.key) / 100
                return p

        p.fuzz = max(
            fuzz.WRatio(na.key, nb.key),
            fuzz.token_sort_ratio(na.key, nb.key),
            fuzz.partial_ratio(na.key, nb.key) * 0.95,
        ) / 100

        # person names: a bare first name is weak evidence, a shared surname
        # plus a compatible given name is strong evidence
        if ma.type in PERSON_TYPES and na.last and nb.last:
            if na.last.casefold() == nb.last.casefold():
                short, long_ = sorted((na, nb), key=lambda n: len(n.tokens))
                if len(short.tokens) == 1 or short.first.casefold() == long_.first.casefold():
                    p.rule = 0.9
                    p.reason = "surname match, compatible given name"
            elif na.first.casefold() == nb.first.casefold() and min(
                len(na.tokens), len(nb.tokens)
            ) == 1:
                # symmetric in (a, b): the score must not depend on which mention
                # id happened to sort first
                p.rule = 0.5
                p.reason = "given-name-only match (ambiguous)"


        # Local-context agreement. Computed before the rules below because the
        # org-prefix rule needs it as corroboration.
        p.context = (
            fuzz.token_set_ratio(ma.context, mb.context) / 100
            if ma.context and mb.context else 0.0
        )

        # "Northwind" vs "Northwind Logistics": one org name is a whole-token
        # prefix of the other. Strong within a corpus, but not certain.
        if ma.type in ORG_TYPES and na.tokens and nb.tokens and na.key != nb.key:
            short, long_ = sorted((na, nb), key=lambda n: len(n.tokens))
            if len(short.tokens) >= 1 and long_.tokens[: len(short.tokens)] == short.tokens:
                # A prefix alone is not proof -- "Apple" and "Apple Bank" are
                # different companies -- so it clears the auto-merge line only
                # when the surrounding context agrees. A bare prefix match lands
                # in the review band instead of merging silently.
                #
                # 0.60 is where a sweep on this corpus puts the boundary: it
                # splits the Apple/Apple Bank pair (context agreement 0.50) while
                # leaving B-cubed F1 unchanged at 0.946 with embeddings on. The
                # unconditional rule merged that pair.
                corroborated = max(p.embed, p.context) >= 0.60
                p.rule = max(p.rule, 0.92 if corroborated else 0.88)
                p.reason = (p.reason + "; " if p.reason else "") + (
                    "org name is a token prefix, context agrees" if corroborated
                    else "org name is a token prefix (unconfirmed)"
                )

        base = self.w_fuzz * p.fuzz + self.w_embed * p.embed + self.w_context * p.context
        base = base / (self.w_fuzz + self.w_embed + self.w_context)
        p.score = max(base, p.rule)

        # the low-entropy trap: short, acronym-like names look similar to each
        # other for no good reason. Make the string score stop carrying the
        # decision on its own.
        if min(na.entropy, nb.entropy) < self.min_entropy and p.rule < 0.9:
            p.score -= self.low_entropy_penalty
            p.reason = (p.reason + "; " if p.reason else "") + "low-entropy name, string score discounted"
        p.score = max(0.0, min(1.0, p.score))
        return p

    # -- stage 4/5: cluster and canonicalise -----------------------------

    @staticmethod
    def cluster(
        mention_ids: Sequence[str], pairs: Sequence[ScoredPair], threshold: float
    ) -> list[set[str]]:
        """Connected components over accepted pairs.

        This is what almost everyone ships, and it is transitively greedy: one
        bad edge fuses two clusters. Guard it with a max-cluster-size alarm
        (:meth:`Resolution.oversized`) and a threshold sweep -- both are cheap
        because every pair score was kept.
        """
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(mention_ids)
        g.add_edges_from(
            (p.a, p.b, {"score": p.score}) for p in pairs if p.score >= threshold
        )
        return [set(c) for c in nx.connected_components(g)]

    def _canonicalize(
        self,
        cluster: set[str],
        by_id: Mapping[str, Mention],
        norms: Mapping[str, NormalizedName],
    ) -> CanonicalEntity:
        members = [by_id[mid] for mid in cluster]
        type_ = Counter(m.type for m in members).most_common(1)[0][0]

        # Prefer the longest surface, breaking ties by summed confidence; for
        # organisations, prefer a form that carries its legal suffix.
        def rank(m: Mention) -> tuple[int, int, float]:
            n = norms[m.mention_id]
            return (1 if (type_ in ORG_TYPES and n.suffix) else 0, len(n.display), m.confidence)

        best = max(members, key=rank)
        canonical = norms[best.mention_id].display
        if type_ in ORG_TYPES and norms[best.mention_id].suffix:
            canonical = f"{canonical} {norms[best.mention_id].suffix}".strip()

        aliases = sorted({m.text.strip() for m in members} - {canonical})
        attrs: dict[str, Any] = {}
        for m in members:  # last non-empty attribute wins
            attrs.update({k: v for k, v in m.attrs.items() if v})

        return CanonicalEntity(
            canon_id=f"{type_}:{_slug(canonical)}",
            type=type_,
            canonical=canonical,
            aliases=aliases,
            mentions=sorted(cluster),
            docs=sorted({m.doc_id for m in members}),
            attrs=attrs,
        )

    # -- driver ----------------------------------------------------------

    def resolve(
        self, mentions: Sequence[Mention], *, threshold: float | None = None
    ) -> Resolution:
        """Run all five stages. Returns every scored pair alongside the clusters."""
        threshold = self.threshold if threshold is None else threshold
        mentions = list(mentions)
        by_id = {m.mention_id: m for m in mentions}
        norms = {m.mention_id: normalize(m.text, m.type) for m in mentions}
        index = {m.mention_id: i for i, m in enumerate(mentions)}

        embeddings = self._embed(mentions, norms)
        blocked = self.block(mentions, norms, embeddings)

        pairs: list[ScoredPair] = []
        for (a, b), tag in blocked.items():
            ma, mb = by_id[a], by_id[b]
            if ma.type != mb.type:
                continue
            emb = 0.0
            if embeddings is not None:
                emb = float(embeddings[index[a]] @ embeddings[index[b]])
            pairs.append(self.score_pair(ma, mb, norms[a], norms[b], emb, tag))

        clusters = self.cluster([m.mention_id for m in mentions], pairs, threshold)
        entities: dict[str, CanonicalEntity] = {}
        mention_to_canon: dict[str, str] = {}
        for cluster in clusters:
            ent = self._canonicalize(cluster, by_id, norms)
            while ent.canon_id in entities:  # distinct clusters, colliding slug
                ent.canon_id += "-2"
            entities[ent.canon_id] = ent
            for mid in cluster:
                mention_to_canon[mid] = ent.canon_id

        n_pairs = len(mentions) * (len(mentions) - 1) // 2
        stats = {
            "mentions": len(mentions),
            "canonical_entities": len(entities),
            "reduction": round(1 - len(entities) / max(len(mentions), 1), 3),
            "candidate_pairs": len(pairs),
            "all_pairs": n_pairs,
            "reduction_ratio": round(1 - len(pairs) / max(n_pairs, 1), 4),
            "accepted_pairs": sum(1 for p in pairs if p.score >= threshold),
            "embeddings": embeddings is not None,
        }
        return Resolution(entities, mention_to_canon, pairs, threshold, stats)

    def blocking_report(
        self, mentions: Sequence[Mention], gold: Mapping[str, str]
    ) -> dict[str, float]:
        """Pair-completeness and reduction ratio against a gold clustering.

        ``gold`` maps ``mention_id -> gold_cluster_label``. Pair completeness is
        the ceiling on recall for everything downstream; report it separately
        from end-to-end F1 or you will tune the wrong stage.

        Blocking runs over the **full** mention set, not just the labelled
        subset -- the unlabelled mentions compete for ANN neighbour slots in the
        real pipeline, so excluding them would flatter the result.
        """
        mentions = list(mentions)
        norms = {m.mention_id: normalize(m.text, m.type) for m in mentions}
        blocked = self.block(mentions, norms, self._embed(mentions, norms))
        proposed = {tuple(sorted(k)) for k in blocked}
        known = {m.mention_id for m in mentions}
        gold = {k: v for k, v in gold.items() if k in known}

        truth: set[tuple[str, str]] = set()
        buckets: dict[str, list[str]] = defaultdict(list)
        for mid, label in gold.items():
            buckets[label].append(mid)
        for members in buckets.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    truth.add(tuple(sorted((members[i], members[j]))))

        hit = len(truth & proposed)
        n_all = len(mentions)
        all_pairs = n_all * (n_all - 1) // 2
        return {
            "gold_pairs": len(truth),
            "candidate_pairs": len(proposed),
            "all_pairs": all_pairs,
            # ceiling on downstream recall: gold pairs that became candidates
            "pair_completeness": round(hit / len(truth), 4) if truth else 1.0,
            # both terms are over the full mention set, not the labelled subset
            "reduction_ratio": round(1 - len(proposed) / max(all_pairs, 1), 4),
        }


class CanonicalRegistry:
    """An incremental registry: resolve new mentions *against* what is known.

    Batch resolution suits document intelligence, where the corpus arrives at
    once. Agent memory is the other shape -- episodes arrive one at a time and
    each must be linked to the existing graph before it is written.

    Two passes are needed, not one. New mentions are first matched against the
    registry's canonical entities; then the *unmatched* remainder is resolved
    among itself. Skipping the second pass means an episode that mentions "Acme"
    and "Acme Corp" for the first time creates two nodes, because neither had a
    registry entry to anchor to.
    """

    def __init__(self, resolver: EntityResolver | None = None) -> None:
        self.resolver = resolver or EntityResolver()
        self.entities: dict[str, CanonicalEntity] = {}
        self.mention_to_canon: dict[str, str] = {}
        self.pairs: list[ScoredPair] = []
        self._order: dict[str, int] = {}  # canon_id -> insertion order, for stable merges

    def _anchor(self, ent: CanonicalEntity) -> Mention:
        return Mention(
            mention_id=f"__registry__:{ent.canon_id}",
            doc_id="__registry__",
            type=ent.type,
            text=ent.canonical,
            start=-1,
            end=-1,
            confidence=1.0,
            context=" ".join(ent.aliases[:6]),
        )

    def add_episode(
        self, mentions: Sequence[Mention], *, episode_id: str | None = None
    ) -> dict[str, str]:
        """Link one episode's mentions into the registry. Returns the id map."""
        anchors = [self._anchor(e) for e in self.entities.values()]
        combined = anchors + list(mentions)
        res = self.resolver.resolve(combined)
        # keep only pairs between real mentions; the synthetic registry anchors
        # are an implementation detail and would pollute a threshold sweep
        self.pairs.extend(
            p for p in res.pairs
            if not p.a.startswith("__registry__:") and not p.b.startswith("__registry__:")
        )

        assigned: dict[str, str] = {}
        for cluster in res.clusters():
            existing = sorted(
                mid.split(":", 1)[1] for mid in cluster if mid.startswith("__registry__:")
            )
            fresh = [mid for mid in cluster if not mid.startswith("__registry__:")]
            if not fresh and not existing:
                continue

            if existing:
                # Merge onto the entity seen first, not the alphabetically-first
                # id -- otherwise first_seen regresses as the registry grows.
                existing.sort(key=lambda cid: (
                    self.entities[cid].first_seen or "", self._order.get(cid, 0)
                ))
                target = self.entities[existing[0]]
                for other in existing[1:]:
                    merged = self.entities.pop(other, None)
                    if merged is None:
                        continue
                    target.aliases = sorted(set(target.aliases) | set(merged.aliases) | {merged.canonical})
                    target.mentions = sorted(set(target.mentions) | set(merged.mentions))
                    target.docs = sorted(set(target.docs) | set(merged.docs))
                    target.attrs.update(merged.attrs)
                    target.first_seen = min(filter(None, (target.first_seen, merged.first_seen)),
                                            default=target.first_seen)
                    target.last_seen = max(filter(None, (target.last_seen, merged.last_seen)),
                                           default=target.last_seen)
                    for mid in merged.mentions:
                        self.mention_to_canon[mid] = target.canon_id
            else:
                by_id = {m.mention_id: m for m in mentions}
                norms = {mid: normalize(by_id[mid].text, by_id[mid].type) for mid in fresh}
                target = self.resolver._canonicalize(set(fresh), by_id, norms)
                while target.canon_id in self.entities:
                    target.canon_id += "-2"
                self.entities[target.canon_id] = target
                self._order[target.canon_id] = len(self._order)

            by_id = {m.mention_id: m for m in mentions}
            for mid in fresh:
                m = by_id[mid]
                if m.text.strip() != target.canonical:
                    target.aliases = sorted(set(target.aliases) | {m.text.strip()})
                target.mentions = sorted(set(target.mentions) | {mid})
                target.docs = sorted(set(target.docs) | {m.doc_id})
                if m.attrs:
                    target.attrs.update({k: v for k, v in m.attrs.items() if v})
                self.mention_to_canon[mid] = target.canon_id
                assigned[mid] = target.canon_id
            if episode_id and fresh:
                # only stamp entities this episode actually mentioned
                target.first_seen = target.first_seen or episode_id
                target.last_seen = episode_id
        return assigned

    def frame(self):
        import pandas as pd

        return pd.DataFrame([e.as_record() for e in self.entities.values()]).sort_values(
            ["type", "n_mentions"], ascending=[True, False]
        )

    def as_resolution(self) -> Resolution:
        return Resolution(
            dict(self.entities), dict(self.mention_to_canon), list(self.pairs),
            self.resolver.threshold,
            {"mentions": len(self.mention_to_canon), "canonical_entities": len(self.entities)},
        )


def bcubed(
    predicted: Mapping[str, str], gold: Mapping[str, str]
) -> dict[str, float]:
    """B-cubed precision / recall / F1 over a mention clustering.

    B-cubed rather than pairwise F1: pairwise scores are dominated by whichever
    cluster happens to be largest, so a single over-merge of the two most common
    entities can look like a good result.

    Both arguments map ``mention_id -> cluster_label``; only the intersection of
    their keys is scored.
    """
    keys = sorted(set(predicted) & set(gold))
    if not keys:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}

    pred_members: dict[str, set[str]] = defaultdict(set)
    gold_members: dict[str, set[str]] = defaultdict(set)
    for k in keys:
        pred_members[predicted[k]].add(k)
        gold_members[gold[k]].add(k)

    p_sum = r_sum = 0.0
    for k in keys:
        p_cluster = pred_members[predicted[k]]
        g_cluster = gold_members[gold[k]]
        shared = len(p_cluster & g_cluster)
        p_sum += shared / len(p_cluster)
        r_sum += shared / len(g_cluster)

    precision = p_sum / len(keys)
    recall = r_sum / len(keys)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n": len(keys),
    }
