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

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CachedTypeSafe(model={self.model!r}, cache_dir={str(self.cache_dir)!r}, "
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


def pair_questions(type_: str) -> dict[str, Any]:
    """The four questions asked about one candidate merge.

    One ``Score`` that decides, and three ``Noul`` s that explain. The diagnostics
    do not feed the decision -- they are there so a disagreement with the
    resolver can be attributed to something rather than shrugged at, which is
    the whole reason to keep raw judgments instead of just an outcome.
    """
    import typesafe_sdk as ts

    kind = type_ or "entity"
    return {
        "link_state": ts.Score(
            instructions=(
                f"`mention_a` and `mention_b` are two mentions of a {kind}, each quoted "
                f"with the surrounding sentence it appeared in. How do they relate?"
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


def adjudicate_pairs(
    client: CachedTypeSafe,
    pairs: Sequence[ScoredPair],
    mentions: Mapping[str, Any],
    *,
    refresh: bool = False,
) -> list[PairVerdict]:
    """Adjudicate candidate merges the resolver could not call.

    ``mentions`` maps ``mention_id`` to :class:`~kgx.extract.Mention`. See
    :func:`pair_state` for what is sent.

    One request per pair: each pair is a different state, so there is nothing to
    share. The three diagnostic questions ride along for the price of their own
    tokens.
    """
    verdicts: list[PairVerdict] = []
    for pair in pairs:
        ma, mb = mentions[pair.a], mentions[pair.b]
        response = client.ask(pair_state(ma, mb), pair_questions(pair.type), refresh=refresh)
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
    # Union-find over canonical ids only. Mention ids never enter this map, so
    # a mention id that happens to look like a canon id cannot collide with one.
    parent: dict[str, str] = {cid: cid for cid in resolution.mention_to_canon.values()}

    def find(x: str) -> str:
        root = x
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for verdict in verdicts:
        if verdict.outcome not in accept:
            continue
        ra = find(resolution.mention_to_canon[verdict.a])
        rb = find(resolution.mention_to_canon[verdict.b])
        if ra != rb:
            # Deterministic direction, so the result does not depend on the
            # order verdicts came back in.
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    return {mid: find(cid) for mid, cid in resolution.mention_to_canon.items()}
