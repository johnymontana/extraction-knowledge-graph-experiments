"""TypeSafe System One judgments as a pipeline stage.

Every extractor in this repo so far falls into one of two shapes. GLiNER2.5 is a
*span* model: it finds text and types it, and everything it knows has to be
anchored in a span. The ``claude`` CLI is a *generative* model: it will answer
anything, in prose, and the work is in constraining it back down to JSON. Both
have a hole in the same place -- the small semantic judgments the pipeline needs
between stages, which are neither spans nor essays:

    is this extracted edge actually *asserted*, or hedged, denied, or conditional?
    are these two mentions the same company, or two companies with similar names?

TypeSafe's System One model (Jev) is built for exactly that shape. You send
**state** (the text, or a JSON object of named fields) and **typed questions**,
and you get back typed answers with calibrated probabilities -- a ``Choice`` over
a fixed option set, a ``Noul`` (probability of yes), or a ``Score`` (a position
on ordered, described levels). There is no text to parse and no schema to
enforce, because the answer set is the request.

That makes it a different tool from either of the other two, and the notebook
uses it in the two places the existing pipeline explicitly leaves open:

**Assertion gating.** ``kgx.data.documents.MODALITY_TRAPS`` is a list of
sentences that have the surface shape of a fact and are not one. GLiNER's
span-attribute pass approximates this; a ``Choice`` over four assertion statuses
answers it directly.

**Resolution adjudication.** :meth:`kgx.resolve.Resolution.review_band` is
documented in this repo as "the slot an LLM adjudicator fills". TypeSafe's own
entity-alignment cookbook fills that slot with one ``Score`` over three levels
plus diagnostic ``Noul`` s, which is what :func:`adjudicate_pairs` implements.

Three design choices worth stating:

**Every response is cached to disk**, content-addressed on
``(model, state, questions)`` -- the same contract as
:class:`kgx.llm.ClaudeCLI`. A re-run is free and byte-identical, which is what
makes a measured comparison reproducible rather than re-billed and slightly
different each time. A cache hit reconstructs a real
:class:`~typesafe_sdk.SystemOneResponse`, so replayed and live code paths are
the same code path. Delete the cache directory to force real calls.

**Independent questions about the same state go in one request.** System One
evaluates every question in a request in parallel, and extra questions cost only
their own tokens. So edge gating sends one request per *document* carrying two
questions per candidate edge, not one request per edge. :func:`judge_edges`
chunks at ``max_questions`` to keep any single request bounded.

**Unlike the rest of the repo, this needs an API key** (``TYPESAFE_API_KEY``).
There is no local fallback and no stub: a stub would make the measurements
meaningless. Without a key, cached responses still replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .extract import DocGraph
from .resolve import Resolution, ScoredPair

__all__ = [
    "TypeSafeUnavailable",
    "CachedTypeSafe",
    "ASSERTION_LEVELS",
    "EdgeJudgment",
    "edge_questions",
    "judge_edges",
    "sentence_questions",
    "judge_sentences",
    "LINK_LEVELS",
    "LINK_OUTCOMES",
    "PairVerdict",
    "pair_questions",
    "pair_state",
    "adjudicate_pairs",
    "apply_verdicts",
    "TypeVerdict",
    "type_questions",
    "retype_mentions",
    "apply_types",
    "soft_typed",
    "fold_clones",
    "NO_RELATION",
    "RelationPick",
    "legal_relations",
    "candidate_pairs",
    "relation_questions",
    "select_relations",
    "picks_to_graphs",
    "ALTERNATIVE_KINDS",
    "alternatives_questions",
    "JevAlternatives",
]

DEFAULT_MODEL = "jev-latest"
DEFAULT_CACHE_DIR = "output/typesafe_cache"


class TypeSafeUnavailable(RuntimeError):
    """No API key, no cached response, or the API refused the call."""


# ---------------------------------------------------------------------------
# The cached client
# ---------------------------------------------------------------------------


class CachedTypeSafe:
    """A disk-cached System One client.

    Parameters
    ----------
    model:
        ``"jev-latest"`` by default. Pinning an exact version makes a cache
        directory reproducible across a model release; the key includes it.
    cache_dir:
        Where responses are stored, content-addressed. Delete it to force real
        calls. Defaults to ``output/typesafe_cache`` (gitignored, like the LLM
        cache).
    api_key:
        Falls back to ``TYPESAFE_API_KEY``. The client is constructed lazily, so
        a fully-cached notebook runs with no key set at all.

    Counters mirror :class:`kgx.llm.ClaudeCLI`: ``total_ms`` is wall time this
    run actually spent, ``observed_ms`` is what the latency *was* including
    replayed calls, so a cached notebook reports real numbers instead of zeros.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Any = None

        self.n_calls = 0            # live requests issued this run
        self.n_cached = 0           # requests served from disk
        self.n_questions = 0        # questions asked, live + replayed
        self.input_tokens = 0       # live only -- what this run billed
        self.output_tokens = 0
        self.observed_input_tokens = 0   # live + replayed
        self.observed_output_tokens = 0
        self.total_ms = 0           # live only
        self.observed_ms = 0        # live + replayed
        self.n_observed = 0

    def __repr__(self) -> str:
        # Relative to the working directory, so a notebook's printed repr does
        # not bake the author's home directory into a committed output cell.
        try:
            shown = os.path.relpath(self.cache_dir)
        except ValueError:  # pragma: no cover - different drive on Windows
            shown = str(self.cache_dir)
        return (
            f"CachedTypeSafe(model={self.model!r}, cache_dir={shown!r}, "
            f"key={'set' if self.available else 'unset'})"
        )

    # -- availability ----------------------------------------------------

    @property
    def available(self) -> bool:
        """Is a live call possible? A cached one does not need this."""
        return bool(self._api_key)

    def check(self) -> list[str]:
        """Raise unless the API is usable; return the model names it offers."""
        if not self.available:
            raise TypeSafeUnavailable(
                "TYPESAFE_API_KEY is not set. Get one at https://console.typesafe.ai/, "
                "or point cache_dir at a directory with cached responses to replay them."
            )
        return [m.name for m in self._ensure_client().models.list().models]

    def _ensure_client(self) -> Any:
        if self._client is None:
            import typesafe_sdk as ts

            if not self.available:
                raise TypeSafeUnavailable("TYPESAFE_API_KEY is not set")
            self._client = ts.TypeSafeClient(
                api_key=self._api_key,
                model=self.model,
                timeout=self.timeout,
                retry=ts.RetryPolicy(max_retries=self.max_retries),
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- caching ---------------------------------------------------------

    @staticmethod
    def _wire(questions: Mapping[str, Any]) -> Any:
        """The questions exactly as they go on the wire.

        Keying on the SDK's own serialisation rather than on ``repr`` means a
        question rewritten to the same wire form hits the same cache entry, and
        one rewritten to a different form misses -- which is the behaviour you
        want when you are tuning instructions.
        """
        import msgspec

        return json.loads(msgspec.json.encode(dict(questions)))

    def _cache_path(self, state: Any, questions: Mapping[str, Any]) -> Path:
        key = hashlib.sha256(
            json.dumps(
                [self.model, state, self._wire(questions)], sort_keys=True, default=str
            ).encode()
        ).hexdigest()[:32]
        return self.cache_dir / f"{key}.json"

    def cached(self, state: Any, questions: Mapping[str, Any]) -> bool:
        """Would :meth:`ask` be free?"""
        return self._cache_path(state, questions).exists()

    @staticmethod
    def _decoder() -> Any:
        """A typed decoder for a cache entry, built once.

        Decoding the file's *bytes* rather than converting a ``json.loads``
        dict is not a style preference: ``ScoreAnswer.legend`` and
        ``ScoreAnswer.probabilities`` are keyed by ``int``, JSON object keys are
        always strings, and only the JSON decoder knows to read them back as
        integers. Going through ``json.loads`` first throws that away and every
        replay of a ``Score`` fails.
        """
        if CachedTypeSafe.__dict__.get("_cached_decoder") is None:
            import msgspec
            import typesafe_sdk as ts

            # defstruct rather than a class body: this module uses
            # ``from __future__ import annotations``, so a class body's
            # annotations are strings that msgspec resolves against module
            # globals -- where the lazily imported ``ts`` does not exist.
            entry = msgspec.defstruct(
                "_CacheEntry",
                [
                    ("model", str),
                    ("usage", ts.Usage),
                    ("answers", dict[str, ts.Answer]),
                    ("duration_ms", int, 0),
                ],
            )
            CachedTypeSafe._cached_decoder = msgspec.json.Decoder(entry)
        return CachedTypeSafe._cached_decoder

    _cached_decoder: Any = None

    # -- the one call ----------------------------------------------------

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        refresh: bool = False,
    ) -> Any:
        """One System One request, cached on disk.

        Returns a :class:`~typesafe_sdk.SystemOneResponse`, whether live or
        replayed. ``request_id`` is the only attribute that distinguishes them:
        it is a response header, so a replay raises on it rather than inventing
        one.
        """
        import msgspec
        import typesafe_sdk as ts

        if not questions:
            raise ValueError("questions must be non-empty")

        path = self._cache_path(state, questions)
        if path.exists() and not refresh:
            entry = self._decoder().decode(path.read_bytes())
            self.n_cached += 1
            self.n_questions += len(questions)
            self.observed_ms += entry.duration_ms
            self.n_observed += 1
            self.observed_input_tokens += entry.usage.input_tokens or 0
            self.observed_output_tokens += entry.usage.output_tokens or 0
            return ts.SystemOneResponse(
                model=entry.model, usage=entry.usage, answers=entry.answers
            )

        if not self.available:
            raise TypeSafeUnavailable(
                f"TYPESAFE_API_KEY is not set and no cached response exists for this "
                f"request ({path.name}). Nothing to replay."
            )

        client = self._ensure_client()
        started = time.time()
        try:
            response = client.system_one(state, dict(questions), model=self.model)
        except ts.TypeSafeError as exc:  # auth, rate limit, validation, connection
            raise TypeSafeUnavailable(f"{type(exc).__name__}: {exc}") from exc
        duration_ms = int((time.time() - started) * 1000)

        self.n_calls += 1
        self.n_questions += len(questions)
        self.total_ms += duration_ms
        self.observed_ms += duration_ms
        self.n_observed += 1
        self.input_tokens += response.usage.input_tokens or 0
        self.output_tokens += response.usage.output_tokens or 0
        self.observed_input_tokens += response.usage.input_tokens or 0
        self.observed_output_tokens += response.usage.output_tokens or 0

        path.write_text(
            msgspec.json.encode(
                {
                    "model": response.model,
                    "usage": response.usage,
                    "answers": response.answers,
                    "duration_ms": duration_ms,
                    # Provenance only -- deliberately NOT part of the key, so
                    # re-running with a pinned model does not invalidate a cache
                    # built against the alias.
                    "asked_model": self.model,
                    "n_questions": len(questions),
                }
            ).decode()
        )
        return response

    # -- reporting -------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """What this run cost and what the calls were like when they were made."""
        return {
            "requests_live": self.n_calls,
            "requests_cached": self.n_cached,
            "questions": self.n_questions,
            "questions_per_request": round(
                self.n_questions / max(1, self.n_calls + self.n_cached), 1
            ),
            "input_tokens_billed": self.input_tokens,
            "output_tokens_billed": self.output_tokens,
            "input_tokens_observed": self.observed_input_tokens,
            "output_tokens_observed": self.observed_output_tokens,
            "ms_spent": self.total_ms,
            "ms_per_request_observed": round(self.observed_ms / max(1, self.n_observed), 1),
        }


# ---------------------------------------------------------------------------
# Stage 1 -- assertion gating on extracted edges
# ---------------------------------------------------------------------------

# Four statuses, not two. "Negated" and "hypothetical" are both reasons not to
# assert an edge, but they are different errors to make and the corpus plants
# both, so collapsing them would hide which one a gate is actually catching.
ASSERTION_LEVELS: dict[str, str] = {
    "asserted": (
        "The text states this relationship as a present or past fact. It holds, "
        "or it held."
    ),
    "hypothetical": (
        "The text raises this relationship only as a possibility, a condition, a "
        "proposal awaiting approval, or something that would happen if something "
        "else did. It is not claimed to have happened."
    ),
    "negated": (
        "The text denies this relationship, or states that it will not happen, is "
        "not planned, or is not expected."
    ),
    "forward_looking": (
        "The text projects, forecasts or guides that this relationship will hold "
        "in the future. It has not happened yet, but it is not merely conditional."
    ),
}

#: Statuses a fact-asserting graph should keep.
ASSERTED = frozenset({"asserted"})


@dataclass
class EdgeJudgment:
    """One candidate edge, judged.

    ``supported`` and ``status`` are deliberately separate questions. An edge can
    be clearly *about* the document and still not asserted by it (the modality
    traps), and an edge can be nominally asserted and simply not present (an
    extraction error). Asking one question cannot distinguish those.
    """

    doc_id: str
    head: str
    relation: str
    tail: str
    head_type: str = ""
    tail_type: str = ""
    extractor_confidence: float = 0.0
    supported: float = 0.0          # Noul: probability the document states it
    status: str = ""                # Choice: one of ASSERTION_LEVELS
    status_confidence: float = 0.0
    status_probabilities: dict[str, float] = field(default_factory=dict)
    snippet: str = ""

    @property
    def keep(self) -> bool:
        """The default gate: stated by the document, and stated as a fact."""
        return self.supported >= 0.5 and self.status in ASSERTED

    def triple(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)

    def as_record(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "gliner": round(self.extractor_confidence, 3),
            "supported": round(self.supported, 3),
            "status": self.status,
            "status_conf": round(self.status_confidence, 3),
            "keep": self.keep,
        }


def edge_questions(
    head: str, relation: str, tail: str, *, relation_description: str = ""
) -> dict[str, Any]:
    """The two questions asked about one candidate edge.

    The edge is named inside the *instructions* rather than put in the state,
    because the state is the whole document and is shared by every question in
    the request. Question ids are never sent to the model, so the instructions
    have to carry the entire meaning on their own.
    """
    import typesafe_sdk as ts

    means = f" Here, “{relation}” means: {relation_description}" if relation_description else ""
    claim = f"“{head}” — {relation} → “{tail}”"

    return {
        "supported": ts.Noul(
            instructions=(
                f"Does the document say anything that bears on this claim: {claim}?{means} "
                f"Answer yes if the document discusses this relationship at all, in any "
                f"modality — asserted, denied, proposed or merely possible. Answer no if "
                f"the document does not connect these two things."
            ),
            criteria={
                "true": "The document addresses this relationship between these two specific things.",
                "false": (
                    "The document does not connect these two things, or one of them does "
                    "not appear in it."
                ),
            },
        ),
        "status": ts.Choice(
            instructions=(
                f"How does the document present this claim: {claim}?{means} "
                f"Judge only how it is presented, not whether it is plausible."
            ),
            criteria=ASSERTION_LEVELS,
        ),
    }


def judge_edges(
    client: CachedTypeSafe,
    doc_graphs: Sequence[DocGraph],
    ontology: Any = None,
    *,
    max_questions: int = 24,
    include_derived: bool = False,
    refresh: bool = False,
) -> list[EdgeJudgment]:
    """Judge every extracted edge for support and assertion status.

    One request per document per chunk of ``max_questions // 2`` edges, with the
    document as shared state. The alternative -- one request per edge, with the
    snippet as state -- costs the document's tokens once per edge instead of
    once per chunk, and measurably loses the cross-sentence cases where the
    hedge and the claim are not adjacent.

    ``include_derived`` controls whether ``inverse=`` mirror edges are judged.
    They are the same fact stated twice, so by default they are skipped and the
    caller can copy the verdict across.
    """
    descriptions: dict[str, str] = {}
    if ontology is not None:
        descriptions = {r.name: r.description for r in ontology.relations}

    out: list[EdgeJudgment] = []
    per_request = max(1, max_questions // 2)

    for graph in doc_graphs:
        edges = [e for e in graph.edges if include_derived or not e.derived]
        for start in range(0, len(edges), per_request):
            chunk = edges[start : start + per_request]
            questions: dict[str, Any] = {}
            index: dict[str, Any] = {}
            for i, edge in enumerate(chunk):
                head, tail = graph.mention(edge.head), graph.mention(edge.tail)
                tag = f"e{start + i}"
                index[tag] = (edge, head, tail)
                for name, question in edge_questions(
                    head.text,
                    edge.type,
                    tail.text,
                    relation_description=descriptions.get(edge.type, ""),
                ).items():
                    questions[f"{tag}_{name}"] = question

            if not questions:
                continue
            response = client.ask({"document": graph.text}, questions, refresh=refresh)

            for tag, (edge, head, tail) in index.items():
                status = response.choices[f"{tag}_status"]
                out.append(
                    EdgeJudgment(
                        doc_id=graph.doc_id,
                        head=head.text,
                        relation=edge.type,
                        tail=tail.text,
                        head_type=head.type,
                        tail_type=tail.type,
                        extractor_confidence=edge.confidence,
                        supported=response.nouls[f"{tag}_supported"].noul,
                        status=status.choice,
                        status_confidence=status.confidence,
                        status_probabilities=dict(status.probabilities),
                        snippet=head.context or tail.context,
                    )
                )
    return out


def sentence_questions() -> dict[str, Any]:
    """The two questions asked about one sentence with no edge in view.

    The ``factual`` Noul is worded the way it is because of a failure. The
    first version asked *"can the main claim be recorded as a fact that
    currently holds?"* -- and on a denial ("Management has no plans to divest
    the Cascade brand") that has two readings. The divestiture is not a fact;
    the absence of a plan is. The model took the second reading, confidently,
    on every denial in the corpus. Notebook 10 §2.2 shows the before and after.

    So the question now names the thing to judge (the relationship the sentence
    is *about*, as a graph would record it) and names the trap. A Noul near 0.5
    means *equally likely yes or no*; it does not mean "the question was
    ambiguous", so ambiguity has to be removed from the question, not read off
    the answer.
    """
    import typesafe_sdk as ts

    return {
        "status": ts.Choice(
            instructions=(
                "Does `sentence` assert its main claim as a fact, or does it "
                "hedge, deny, project or merely propose it? Judge the modality "
                "of the sentence, not whether the claim is plausible."
            ),
            criteria=ASSERTION_LEVELS,
        ),
        "factual": ts.Noul(
            instructions=(
                "If a knowledge graph recorded the relationship this sentence is "
                "about as a plain fact, would the graph be correct? Be careful with "
                "denials: a sentence saying something will NOT happen or is NOT "
                "planned does not make that thing a fact."
            ),
            criteria={
                "true": "Yes: the sentence asserts the relationship as true now or in the past.",
                "false": (
                    "No: the sentence denies it, conditions it, proposes it, forecasts "
                    "it, or only reports that someone said it might happen."
                ),
            },
        ),
    }


def judge_sentences(
    client: CachedTypeSafe,
    sentences: Iterable[str],
    *,
    questions: Mapping[str, Any] | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Assertion status of whole sentences, with no edge in view.

    The focused version of the same question, used on
    :data:`kgx.data.documents.MODALITY_TRAPS`. One sentence per request, because
    each sentence is a different state.

    ``questions`` overrides :func:`sentence_questions`; it must still supply a
    ``status`` Choice and a ``factual`` Noul. The notebook uses this to replay
    the first, badly worded version beside the current one.
    """
    questions = dict(questions) if questions is not None else sentence_questions()

    rows: list[dict[str, Any]] = []
    for sentence in sentences:
        response = client.ask({"sentence": sentence}, questions, refresh=refresh)
        rows.append(
            {
                "sentence": sentence,
                "status": response.choices["status"].choice,
                "status_conf": response.choices["status"].confidence,
                "factual": response.nouls["factual"].noul,
                "probabilities": dict(response.choices["status"].probabilities),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Stage 2 -- resolution adjudication
# ---------------------------------------------------------------------------

# The shape TypeSafe's own entity-alignment cookbook recommends: one ordered
# Score whose levels ARE the outcomes, instead of a probability plus a pair of
# fitted thresholds. The middle level is not a hedge, it is the curator queue --
# which this repo already has a name for (`Resolution.review_band`).
LINK_LEVELS: list[str] = [
    "They refer to two different real-world entities.",
    "They refer to closely related entities that may or may not be the same one.",
    "They refer to one and the same real-world entity.",
]

LINK_OUTCOMES: dict[int, str] = {0: "reject", 1: "review", 2: "merge"}


@dataclass
class PairVerdict:
    """One candidate merge, adjudicated."""

    a: str                  # mention_id
    b: str                  # mention_id
    a_text: str
    b_text: str
    type: str
    resolver_score: float   # what kgx.resolve thought
    level: float = 0.0      # Score: expected level on LINK_LEVELS
    confidence: float = 0.0
    probabilities: dict[int, float] = field(default_factory=dict)
    same_name: float = 0.0
    same_context: float = 0.0
    abbreviation: float = 0.0

    @property
    def outcome(self) -> str:
        """The nearest level names the outcome. No fitted threshold."""
        return LINK_OUTCOMES[min(int(self.level + 0.5), len(LINK_LEVELS) - 1)]

    def as_record(self) -> dict[str, Any]:
        return {
            "a": self.a_text,
            "b": self.b_text,
            "type": self.type,
            "resolver": round(self.resolver_score, 3),
            "level": round(self.level, 3),
            "conf": round(self.confidence, 3),
            "same_name": round(self.same_name, 3),
            "same_context": round(self.same_context, 3),
            "abbrev": round(self.abbreviation, 3),
            "outcome": self.outcome,
        }


def pair_questions(type_: str, type_description: str = "") -> dict[str, Any]:
    """The four questions asked about one candidate merge.

    One ``Score`` that decides, and three ``Noul`` s that explain. The diagnostics
    do not feed the decision -- they are there so a disagreement with the
    resolver can be attributed to something rather than shrugged at, which is
    the whole reason to keep raw judgments instead of just an outcome.

    ``type_description`` is the ontology's annotation guideline for the type,
    when there is one. It goes into the ``Score`` instructions, because what
    counts as "the same product" is a domain decision -- the shopping ontology
    says outright that ``Aurora 14`` and ``Aurora 14 Pro`` are two products --
    and a judge that has not been told the rule can only guess at it. Left
    empty, the question is byte-identical to the one without it, so cached
    answers stay valid.
    """
    import typesafe_sdk as ts

    kind = type_ or "entity"
    guideline = f" In this domain, a {kind} is: {type_description.strip()}" if type_description else ""
    return {
        "link_state": ts.Score(
            instructions=(
                f"`mention_a` and `mention_b` are two mentions of a {kind}, each quoted "
                f"with the surrounding sentence it appeared in.{guideline} How do they relate?"
            ),
            criteria=LINK_LEVELS,
        ),
        "same_name": ts.Noul(
            instructions=(
                "Ignoring abbreviations, legal suffixes, honorifics and possessives, do "
                "`mention_a.text` and `mention_b.text` name the same thing?"
            ),
        ),
        "same_context": ts.Noul(
            instructions=(
                "Do `mention_a.context` and `mention_b.context` describe the same "
                "entity doing the same kind of thing, rather than two different "
                "entities that happen to share part of a name?"
            ),
        ),
        "abbreviation": ts.Noul(
            instructions=(
                "Is one of `mention_a.text` and `mention_b.text` an abbreviation, "
                "acronym, ticker symbol or short form of the other?"
            ),
        ),
    }


def pair_state(ma: Any, mb: Any) -> dict[str, Any]:
    """The state sent for one candidate merge: both mentions, with context.

    The ``context`` window is what makes ``Apple`` / ``Apple Bank`` separable,
    so it goes in beside the surface string rather than instead of it. Named
    fields, not a concatenated blob, so the questions can point at
    ``mention_a.text`` and ``mention_b.context`` by path.
    """
    return {
        "mention_a": {"text": ma.text, "type": ma.type, "context": ma.context},
        "mention_b": {"text": mb.text, "type": mb.type, "context": mb.context},
    }


def _type_guideline(type_label: str, ontology: Any) -> str:
    """The ontology descriptions behind a pair's type label.

    A cross-type pair carries a label like ``"company or security"``; both
    descriptions are sent, because the judge needs to know what each side was
    taken to be.
    """
    if ontology is None:
        return ""
    parts = []
    for name in type_label.split(" or "):
        try:
            desc = ontology.entity(name.strip()).description
        except KeyError:
            continue
        if desc:
            parts.append(desc.strip())
    return " / ".join(parts)


def adjudicate_pairs(
    client: CachedTypeSafe,
    pairs: Sequence[ScoredPair],
    mentions: Mapping[str, Any],
    *,
    ontology: Any = None,
    refresh: bool = False,
) -> list[PairVerdict]:
    """Adjudicate candidate merges the resolver could not call.

    ``mentions`` maps ``mention_id`` to :class:`~kgx.extract.Mention`. See
    :func:`pair_state` for what is sent. With ``ontology``, each pair's type
    description is passed to :func:`pair_questions` as the guideline.

    One request per pair: each pair is a different state, so there is nothing to
    share. The three diagnostic questions ride along for the price of their own
    tokens.
    """
    verdicts: list[PairVerdict] = []
    for pair in pairs:
        ma, mb = mentions[pair.a], mentions[pair.b]
        questions = pair_questions(pair.type, _type_guideline(pair.type, ontology))
        response = client.ask(pair_state(ma, mb), questions, refresh=refresh)
        score = response.scores["link_state"]
        verdicts.append(
            PairVerdict(
                a=pair.a,
                b=pair.b,
                a_text=ma.text,
                b_text=mb.text,
                type=pair.type,
                resolver_score=pair.score,
                level=score.score,
                confidence=score.confidence,
                probabilities=dict(score.probabilities),
                same_name=response.nouls["same_name"].noul,
                same_context=response.nouls["same_context"].noul,
                abbreviation=response.nouls["abbreviation"].noul,
            )
        )
    return verdicts


def apply_verdicts(
    resolution: Resolution,
    verdicts: Sequence[PairVerdict],
    *,
    accept: Iterable[str] = ("merge",),
) -> dict[str, str]:
    """Fold accepted merges into the resolver's clustering.

    Returns a fresh ``mention_id -> canonical_id`` map, which is what
    :func:`kgx.resolve.bcubed` scores and what :func:`kgx.graph.build_graph`
    takes as ``pin``. The original :class:`~kgx.resolve.Resolution` is not
    mutated, so the before/after comparison stays runnable in either order.

    Merging is transitive by union-find, which is the point and the risk: three
    pairwise merges can produce one cluster no single verdict proposed. Check
    :meth:`~kgx.resolve.Resolution.oversized` afterwards.
    """
    accept = set(accept)
    find, union = _union_find(resolution.mention_to_canon.values())
    for verdict in verdicts:
        if verdict.outcome in accept:
            union(resolution.mention_to_canon[verdict.a], resolution.mention_to_canon[verdict.b])
    return {mid: find(cid) for mid, cid in resolution.mention_to_canon.items()}


def _union_find(ids: Iterable[str]):
    """Union-find over canonical ids, with a deterministic merge direction.

    Only canonical ids enter the map -- never mention ids -- so a mention id
    that happens to look like a canon id cannot collide with one. Merging
    always points the lexically larger root at the smaller, so the result does
    not depend on the order the unions arrive in.
    """
    parent: dict[str, str] = {cid: cid for cid in ids}

    def find(x: str) -> str:
        root = x
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    return find, union


# ---------------------------------------------------------------------------
# Stage 2b -- typing mentions before blocking
# ---------------------------------------------------------------------------
#
# Blocking in kgx.resolve is type-scoped: a `security` and a `company` are never
# compared, whatever their strings say. Notebook 10 finds that this is where the
# corpus's remaining recall went -- `HLCN` typed as a ticker, `Halcyon` as the
# issuer -- and that both readings are defensible under the ontology, which
# defines `security` to include ticker symbols.
#
# So the question asked here is about the REFERENT, not the expression: what
# real-world thing does this mention stand for. And the answer kept is the
# distribution, not the label. A ticker that comes back 0.6 security / 0.4
# company is telling you the type is ambiguous under this ontology; the useful
# response is to let blocking see the mention under both types and let scoring
# decide, which is what `soft_typed` does.


@dataclass
class TypeVerdict:
    """One mention, re-typed against the ontology by referent."""

    mention_id: str
    doc_id: str
    text: str
    extractor_type: str
    type: str                    # the model's argmax
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.type != self.extractor_type

    def alternatives(self, min_prob: float = 0.25) -> list[str]:
        """Types other than the argmax that carry at least ``min_prob``."""
        return [
            t for t, p in sorted(self.probabilities.items(), key=lambda kv: -kv[1])
            if t != self.type and p >= min_prob
        ]

    def as_record(self) -> dict[str, Any]:
        second = self.alternatives(0.0)[:1]
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "gliner": self.extractor_type,
            "typesafe": self.type,
            "conf": round(self.confidence, 3),
            "runner_up": second[0] if second else "",
            "p_runner_up": round(self.probabilities.get(second[0], 0.0), 3) if second else 0.0,
            "changed": self.changed,
        }


def type_questions(text: str, ontology: Any) -> dict[str, Any]:
    """One ``Choice`` over the ontology's entity types, for one surface form.

    The criteria are the ontology's own descriptions -- the same annotation
    guidelines GLiNER was given -- so the model and the extractor are held to
    the same definitions. The instructions ask about the referent so that a
    ticker, nickname or abbreviation is judged by what it stands for.
    """
    import typesafe_sdk as ts

    return {
        "type": ts.Choice(
            instructions=(
                f"In `document`, what kind of real-world thing does the mention "
                f"“{text}” stand for? Judge the thing it refers to in this "
                f"document, not the form of the expression: a name, nickname, "
                f"abbreviation or ticker symbol used to talk about something stands "
                f"for that thing."
            ),
            criteria={e.name: e.description for e in ontology.entities},
        ),
    }


def retype_mentions(
    client: CachedTypeSafe,
    doc_graphs: Sequence[DocGraph],
    ontology: Any,
    *,
    max_questions: int = 24,
    refresh: bool = False,
) -> list[TypeVerdict]:
    """Re-type every mention by referent, one request per document per chunk.

    Questions are deduplicated on surface form within a document -- the same
    string in the same document stands for the same thing -- and the answer is
    copied to every mention with that surface. Returns one verdict per mention.
    """
    verdicts: list[TypeVerdict] = []
    for graph in doc_graphs:
        surfaces: dict[str, list[Any]] = {}
        for m in graph.mentions:
            surfaces.setdefault(m.text.strip(), []).append(m)
        order = list(surfaces)
        for start in range(0, len(order), max_questions):
            chunk = order[start : start + max_questions]
            questions = {
                f"s{start + i}": type_questions(text, ontology)["type"]
                for i, text in enumerate(chunk)
            }
            response = client.ask({"document": graph.text}, questions, refresh=refresh)
            for i, text in enumerate(chunk):
                answer = response.choices[f"s{start + i}"]
                for m in surfaces[text]:
                    verdicts.append(
                        TypeVerdict(
                            mention_id=m.mention_id,
                            doc_id=graph.doc_id,
                            text=text,
                            extractor_type=m.type,
                            type=answer.choice,
                            confidence=answer.confidence,
                            probabilities=dict(answer.probabilities),
                        )
                    )
    return verdicts


def apply_types(mentions: Sequence[Any], verdicts: Sequence[TypeVerdict]) -> list[Any]:
    """Mentions with the model's argmax type in place of the extractor's."""
    from dataclasses import replace

    by_id = {v.mention_id: v for v in verdicts}
    return [
        replace(m, type=by_id[m.mention_id].type) if m.mention_id in by_id else m
        for m in mentions
    ]


def soft_typed(
    mentions: Sequence[Any],
    verdicts: Sequence[TypeVerdict],
    *,
    min_prob: float = 0.25,
) -> tuple[list[Any], dict[str, str]]:
    """Offer ambiguous mentions to blocking under every type with real mass.

    Returns the argmax-typed mentions plus one *clone* per alternative type at
    or above ``min_prob``, and a map from clone id back to the original. Clone
    ids are ``<mention_id>~<type>``. Run the resolver over the augmented list,
    then :func:`fold_clones` to get a clustering over the originals.

    A clone enters the alternative type's block and is scored there like any
    other mention; it does not merge with anything by itself. So this widens
    candidate generation without touching the scoring rule -- which is exactly
    the dial the type wall was closing.
    """
    from dataclasses import replace

    by_id = {v.mention_id: v for v in verdicts}
    out: list[Any] = []
    clone_to_original: dict[str, str] = {}
    for m in mentions:
        v = by_id.get(m.mention_id)
        if v is None:
            out.append(m)
            continue
        out.append(replace(m, type=v.type))
        for alt in v.alternatives(min_prob):
            clone_id = f"{m.mention_id}~{alt}"
            out.append(replace(m, mention_id=clone_id, type=alt))
            clone_to_original[clone_id] = m.mention_id
    return out, clone_to_original


# ---------------------------------------------------------------------------
# Stage 0b -- relation selection over enumerated entity pairs
# ---------------------------------------------------------------------------
#
# System One has no spans: it cannot find an entity, and it cannot find an
# edge. What it can do is *select*. Given two mentions the extractor already
# found and the list of relations the ontology permits between their types, one
# Choice picks the relation the document states -- or `none`. Code enumerates
# the candidates; the model judges each one; nothing is generated.
#
# This is the "select instead of generate" pattern from the TypeSafe docs
# applied to edges, and it is how notebook 11 asks whether a judgment model can
# raise relation recall over joint decoding, given the same entities.

NO_RELATION = "none"


@dataclass
class RelationPick:
    """One candidate (head, tail) pair, with the relation the model selected."""

    doc_id: str
    head: str                # mention_id
    tail: str                # mention_id
    head_text: str
    tail_text: str
    head_type: str
    tail_type: str
    relation: str            # a relation name, or NO_RELATION
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    scope: str = ""

    @property
    def picked(self) -> bool:
        return self.relation != NO_RELATION

    @property
    def p_relation(self) -> float:
        """Probability of the chosen relation (0 when `none` was chosen)."""
        return 0.0 if not self.picked else self.probabilities.get(self.relation, 0.0)

    def as_record(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "head": self.head_text,
            "relation": self.relation,
            "tail": self.tail_text,
            "head_type": self.head_type,
            "tail_type": self.tail_type,
            "p": round(self.p_relation, 3),
            "conf": round(self.confidence, 3),
            "p_none": round(self.probabilities.get(NO_RELATION, 0.0), 3),
        }


def legal_relations(ontology: Any) -> dict[tuple[str, str], list[Any]]:
    """``(head_type, tail_type) -> [RelationType, ...]`` for every legal pattern."""
    out: dict[tuple[str, str], list[Any]] = {}
    for rel in ontology.relations:
        for h in rel.head:
            for t in rel.tail:
                out.setdefault((h, t), []).append(rel)
    return out


def _spans(text: str, scope: str) -> list[tuple[int, int]]:
    """Character spans of the units candidate pairs are drawn from."""
    if scope == "document":
        return [(0, len(text))]
    if scope == "paragraph":
        pattern = r"\n\s*\n"
    elif scope == "sentence":
        # Crude on purpose -- "Inc." will split a sentence -- and documented as
        # the pessimistic end of the range rather than fixed. Paragraphs are the
        # unit the notebook actually uses.
        pattern = r"(?<=[.!?])\s+(?=[A-Z\"“])"
    else:
        raise ValueError(f"scope must be document, paragraph or sentence, not {scope!r}")
    spans, pos = [], 0
    for m in re.finditer(pattern, text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    return spans


def candidate_pairs(
    graph: DocGraph, ontology: Any, *, scope: str = "paragraph"
) -> list[tuple[Any, Any]]:
    """Ordered mention pairs the ontology permits, co-occurring within ``scope``.

    Deduplicated on ``(head surface, tail surface)`` within a document, so a
    company named six times is asked about once per partner, not six times.
    Both directions of a pair are separate candidates when both are legal --
    ``supplies`` runs one way and ``subsidiary_of`` the other.
    """
    legal = legal_relations(ontology)
    pairs: list[tuple[Any, Any]] = []
    seen: set[tuple[str, str]] = set()
    for s0, s1 in _spans(graph.text, scope):
        inside = [m for m in graph.mentions if m.start >= s0 and m.end <= s1]
        for a in inside:
            for b in inside:
                if a is b:
                    continue
                key = (a.text.strip(), b.text.strip())
                if key in seen or key[0] == key[1] or (a.type, b.type) not in legal:
                    continue
                seen.add(key)
                pairs.append((a, b))
    return pairs


def relation_questions(head_text: str, tail_text: str, relations: Sequence[Any]) -> dict[str, Any]:
    """One ``Choice`` over the legal relations for a pair, plus ``none``.

    The criteria are the ontology's relation descriptions -- the same text
    GLiNER decodes against -- and ``none`` is always present, because the
    docs are explicit that a question must be able to answer "nothing fits".
    Direction is stated, since the reverse pair is a separate question.
    """
    import typesafe_sdk as ts

    criteria = {r.name: r.description for r in relations}
    criteria[NO_RELATION] = (
        "The document does not relate these two things in any of the ways listed, "
        "or relates them only in the opposite direction."
    )
    return {
        "relation": ts.Choice(
            instructions=(
                f"In `document`, which of these relationships does the text state or "
                f"discuss from “{head_text}” to “{tail_text}”, in that "
                f"direction? Pick the one relationship the text supports, or none."
            ),
            criteria=criteria,
        ),
    }


def select_relations(
    client: CachedTypeSafe,
    doc_graphs: Sequence[DocGraph],
    ontology: Any,
    *,
    scope: str = "paragraph",
    max_questions: int = 24,
    refresh: bool = False,
) -> list[RelationPick]:
    """Ask one relation question per candidate pair, batched per document.

    The document is the shared state; every question names its own pair. A
    document with more candidates than ``max_questions`` is split into
    several requests, as :func:`judge_edges` does.
    """
    legal = legal_relations(ontology)
    picks: list[RelationPick] = []
    for graph in doc_graphs:
        pairs = candidate_pairs(graph, ontology, scope=scope)
        for start in range(0, len(pairs), max_questions):
            chunk = pairs[start : start + max_questions]
            questions = {
                f"p{start + i}": relation_questions(a.text, b.text, legal[(a.type, b.type)])["relation"]
                for i, (a, b) in enumerate(chunk)
            }
            response = client.ask({"document": graph.text}, questions, refresh=refresh)
            for i, (a, b) in enumerate(chunk):
                answer = response.choices[f"p{start + i}"]
                picks.append(
                    RelationPick(
                        doc_id=graph.doc_id,
                        head=a.mention_id,
                        tail=b.mention_id,
                        head_text=a.text,
                        tail_text=b.text,
                        head_type=a.type,
                        tail_type=b.type,
                        relation=answer.choice,
                        confidence=answer.confidence,
                        probabilities=dict(answer.probabilities),
                        scope=scope,
                    )
                )
    return picks


def picks_to_graphs(
    doc_graphs: Sequence[DocGraph],
    picks: Sequence[RelationPick],
    *,
    min_prob: float = 0.5,
) -> list[DocGraph]:
    """Document graphs whose edges are the selected relations.

    Mentions are the extractor's, untouched; only the edges change. An edge's
    confidence is the probability of the relation the model chose, so the
    downstream ``min_confidence`` filters mean the same thing they do for
    GLiNER edges. Every edge is legal by construction -- the choices were
    drawn from :func:`legal_relations`.
    """
    from .extract import Edge

    by_doc: dict[str, list[RelationPick]] = {}
    for p in picks:
        by_doc.setdefault(p.doc_id, []).append(p)
    out: list[DocGraph] = []
    for g in doc_graphs:
        edges = [
            Edge(doc_id=g.doc_id, type=p.relation, head=p.head, tail=p.tail,
                 confidence=p.p_relation)
            for p in by_doc.get(g.doc_id, [])
            if p.picked and p.p_relation >= min_prob
        ]
        out.append(DocGraph(doc_id=g.doc_id, text=g.text, mentions=list(g.mentions),
                            edges=edges, feasible=g.feasible, meta=dict(g.meta)))
    return out


# ---------------------------------------------------------------------------
# Stage 5 -- deciding what contradicts what, for the temporal layer
# ---------------------------------------------------------------------------
#
# kgx.temporal supersedes a `prefers`/`avoids`/`uses_tool` fact only when an
# `alternative_fn(a, b)` says the two tails are alternatives -- and the README
# records that embeddings cannot make that call ("npm/pnpm" and "npm/Berlin"
# overlap under every template tried). It is a world-knowledge question, which
# is what a System One model is for. `JevAlternatives` is that function, with
# a log, so `TemporalGraph(alternative_fn=JevAlternatives(client))` just works.

ALTERNATIVE_KINDS: dict[str, str] = {
    "alternatives": (
        "Two options for the same role, such that adopting one usually means giving "
        "the other up: two package managers, two programming languages for the same "
        "job, two CI systems."
    ),
    "complementary": (
        "Different kinds of thing that are commonly held together and do not compete: "
        "a language and a database, a tool and a city."
    ),
    "same": "Two names for one and the same thing.",
    "unrelated": "Nothing to do with each other.",
}


def alternatives_questions(a: str, b: str, *, relation: str = "") -> dict[str, Any]:
    """Are ``a`` and ``b`` alternatives? A ``Noul`` to decide, a ``Choice`` to explain.

    The state is just the two names, because that is all
    :meth:`kgx.temporal.TemporalGraph._conflicts` has to offer -- the decision
    has to come from what the model knows about the things, which is the point.
    ``relation`` (e.g. ``prefers``) is named in the instructions when known.
    """
    import typesafe_sdk as ts

    about = f" Both are things someone might “{relation}”." if relation else ""
    return {
        "alternatives": ts.Noul(
            instructions=(
                f"Are `a` and `b` alternatives -- two options filling the same role, so "
                f"that choosing one usually means dropping the other?{about} Answer no "
                f"if they are different kinds of thing, or things commonly used together."
            ),
            criteria={
                "true": "They compete for the same role; a person typically settles on one.",
                "false": "They do not compete: different kinds of thing, or complementary, or the same thing.",
            },
        ),
        "kind": ts.Choice(
            instructions=f"How do `a` and `b` relate to each other?{about}",
            criteria=ALTERNATIVE_KINDS,
        ),
    }


class JevAlternatives:
    """An ``alternative_fn`` for :class:`kgx.temporal.TemporalGraph`, backed by System One.

    Callable as ``fn(a, b) -> bool``; every decision is appended to
    ``decisions`` with the probability and the explanatory ``kind``, so the
    supersessions the temporal layer makes can be audited afterwards. The pair
    is sorted before it is sent, so ``(npm, pnpm)`` and ``(pnpm, npm)`` share
    one cached answer.
    """

    def __init__(
        self,
        client: CachedTypeSafe,
        *,
        threshold: float = 0.5,
        relation: str = "",
    ) -> None:
        self.client = client
        self.threshold = threshold
        self.relation = relation
        self.decisions: list[dict[str, Any]] = []
        self._memo: dict[tuple[str, str], dict[str, Any]] = {}

    def judge(self, a: str, b: str) -> dict[str, Any]:
        key = tuple(sorted((a.strip(), b.strip())))
        if key in self._memo:
            return self._memo[key]
        response = self.client.ask(
            {"a": key[0], "b": key[1]}, alternatives_questions(key[0], key[1], relation=self.relation)
        )
        kind = response.choices["kind"]
        record = {
            "a": key[0],
            "b": key[1],
            "p_alternatives": response.nouls["alternatives"].noul,
            "kind": kind.choice,
            "kind_conf": kind.confidence,
        }
        self._memo[key] = record
        return record

    def __call__(self, a: str, b: str) -> bool:
        record = self.judge(a, b)
        decision = {**record, "decided": record["p_alternatives"] >= self.threshold}
        self.decisions.append(decision)
        return decision["decided"]

    def frame(self):
        import pandas as pd

        return pd.DataFrame(self.decisions)


def fold_clones(
    mention_to_canon: Mapping[str, str], clone_to_original: Mapping[str, str]
) -> dict[str, str]:
    """Collapse a clustering over originals + clones to one over originals.

    Wherever a clone landed in a cluster, its original joins that cluster;
    the union is transitive, so a clone that bridged two clusters merges them.
    """
    find, union = _union_find(mention_to_canon.values())
    for clone, original in clone_to_original.items():
        if clone in mention_to_canon and original in mention_to_canon:
            union(mention_to_canon[clone], mention_to_canon[original])
    return {
        mid: find(cid) for mid, cid in mention_to_canon.items()
        if mid not in clone_to_original
    }
