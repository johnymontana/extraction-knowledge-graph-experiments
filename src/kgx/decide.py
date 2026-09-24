"""GLiNER2.5-Decide as a pipeline stage: typed decisions over the extractor's output.

GLiNER2.5 (:mod:`kgx.extract`) is a span model -- it finds text and types it.
`GLiNER2.5-Decide <https://huggingface.co/fastino/GLiNER2.5-Decide>`_ is its
classification sibling: same ``gliner2`` library, a DeBERTa-v3-large encoder,
and one job -- pick a label from a set you pass at call time, and return the
probability of every label. No spans, no generated tokens, no API key.

That is the shape of the judgments the pipeline needs *between* stages, which
notebook 10 answered with a hosted model. This module asks them locally:

**Edge checks.** Is the sentence that carries an extracted edge a *fact*, or a
possibility, denial or forecast? And, given the relations the ontology permits
between the two endpoint types, does Decide pick the same one GLiNER decoded?
:func:`check_edges` asks both; :attr:`EdgeCheck.keep` requires both.

**Typing ahead of blocking.** :func:`type_mentions` asks which ontology type a
surface string names, and :func:`second_opinion` offers each mention to
blocking under Decide's type as well as GLiNER's, so a ticker typed ``security``
can still meet the company it stands for.

Four things were measured on this repo's corpus before any of this was written,
and they shape every schema below (notebook 15 shows each one):

**Short concept labels beat long criteria.** ``fact / possibility / denial /
forecast`` separates the planted modality traps from plain assertions; the same
four statuses written as TypeSafe-style sentences do not, and a yes/no question
("is this stated as fact?") is nearly flat. The label set *is* the question.

**Decide classifies the text, not a span inside it.** Pass a sentence and ask
what ``Marcus Webb`` is, and the answer is about the sentence -- which is about
a company. So typing sends the bare surface string, and edge checks send only
the sentences the edge spans. An instruction naming the span barely steers it.

**The task name is part of the prompt.** GLiNER2 encodes it with the labels, so
``"modality"`` and ``"m"`` are different questions and give different answers.
The names here were chosen for meaning and then left alone; the notebook
reports the spread across alternatives rather than picking the best one.

**It verifies; it does not find.** Asked to choose a relation for every
co-occurring pair, Decide almost never says ``no relation``, so precision
collapses. Asked whether an edge GLiNER already proposed is the relation the
sentence states, it is a useful second opinion. Same for resolution: typing a
mention works, deciding whether two mentions are one entity does not
(:func:`same_referent` exists so the notebook can show that).

The helpers reused from :mod:`kgx.typesafe` (``fold_clones``,
``candidate_pairs``, ``RelationPick``) are pure Python: importing them touches
neither the TypeSafe SDK nor an API key.
"""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .extract import DocGraph, Edge, Mention
from .typesafe import (
    NO_RELATION,
    RelationPick,
    candidate_pairs,
    fold_clones,
    legal_relations,
)

__all__ = [
    "DECIDE_MODEL",
    "DecideJudge",
    "MODALITY_LABELS",
    "FACT",
    "NO_RELATION_LABEL",
    "clean",
    "type_label",
    "relation_label",
    "sentence_spans",
    "edge_window",
    "modality_schema",
    "relation_schema",
    "type_schema",
    "judge_sentences",
    "EdgeCheck",
    "check_edges",
    "aggregate_checks",
    "TypeOpinion",
    "type_mentions",
    "second_opinion",
    "fold_clones",
    "legal_relations",
    "candidate_pairs",
    "select_relations",
    "same_referent",
]

DECIDE_MODEL = "fastino/GLiNER2.5-Decide"

#: Modality of the sentence carrying an edge. ``not stated`` gives the model a
#: way to say the window does not carry the claim at all; it measured better
#: with it than without (notebook 15, §2).
MODALITY_LABELS: tuple[str, ...] = ("fact", "possibility", "denial", "forecast", "not stated")
FACT = "fact"

#: The "nothing fits" label offered with every relation choice. It maps back to
#: :data:`kgx.typesafe.NO_RELATION` so :class:`RelationPick` works unchanged.
NO_RELATION_LABEL = "no relation"

# Marker tokens the gliner2 processor injects into the prompt. A label or
# instruction containing one is rejected at schema build time (it would corrupt
# logit-to-label alignment), so surface strings are scrubbed on the way in.
_RESERVED = re.compile(r"\[(?:P|L|C|E|R|DESCRIPTION|EXAMPLE|OUTPUT)\]|[()]")


def clean(text: str) -> str:
    """Make a surface string safe to put in a label or instruction.

    ``gliner2.classification`` refuses parentheses and its marker tokens in any
    string it will inject into the prompt. Mentions such as ``"(NYSE: HLCN)"``
    are rare but real, so strip rather than fail.
    """
    return re.sub(r"\s+", " ", _RESERVED.sub("", text)).strip() or "?"


def type_label(name: str) -> str:
    """``business_segment`` -> ``business segment``: the label Decide is shown."""
    return name.replace("_", " ")


def relation_label(name: str) -> str:
    """``officer_of`` -> ``officer of``. Generic on purpose -- a hand-written
    verb table ("is an officer of") measured no better, and would have to be
    rewritten for every ontology."""
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class DecideJudge:
    """GLiNER2.5-Decide behind one call: texts and schemas in, probabilities out.

    Every stage in this module talks to the model only through
    :meth:`probabilities`, which is what makes the stages testable without
    loading 486M parameters: pass anything with that method.

    Scoring goes through ``gliner2.classification.Classifier`` rather than the
    ``classify_text`` convenience method, because the classifier keeps the full
    distribution over labels -- ``classify_text`` returns only the argmax and its
    probability, and most of what the notebook does is read the rest.

    Outputs are deterministic and do not depend on batch size or on which
    schemas share a batch (measured: bit-identical at batch 1, 8 and 32).
    """

    def __init__(
        self,
        model_id: str = DECIDE_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 8,
        **load_kwargs: Any,
    ) -> None:
        from gliner2.classification import Classifier

        with warnings.catch_warnings():
            # Same two harmless load-time warnings GlinerExtractor silences.
            warnings.simplefilter("ignore", UserWarning)
            warnings.filterwarnings("ignore", message=".*attn_implementation.*")
            t0 = time.time()
            self.classifier = Classifier.from_pretrained(model_id, device=device, **load_kwargs)
        self.load_time_s = time.time() - t0
        self.model_id = model_id
        self.batch_size = batch_size
        self.n_texts = 0
        self.n_calls = 0
        self.seconds = 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = sum(p.numel() for p in self.classifier.model.parameters()) / 1e6
        return (f"<DecideJudge {self.model_id} {n:.0f}M params on {self.classifier.device}, "
                f"loaded in {self.load_time_s:.1f}s>")

    def probabilities(
        self, texts: Sequence[str], schemas: Any
    ) -> list[dict[str, dict[str, float]]]:
        """``{task: {label: probability}}`` for each text.

        ``schemas`` is one ``ClassificationSchema`` shared by every text, or a
        sequence with one per text -- the normal case here, since each edge's
        relation question names its own endpoints.
        """
        texts = list(texts)
        if not texts:
            return []
        if isinstance(schemas, (list, tuple)):
            compiled = [self.classifier.compile_schema(s) for s in schemas]
        else:
            compiled = self.classifier.compile_schema(schemas)
        t0 = time.time()
        scores = self.classifier.scorer.batch_score(texts, compiled, batch_size=self.batch_size)
        self.seconds += time.time() - t0
        self.n_calls += 1
        self.n_texts += len(texts)
        return [
            {task: {label: s.probability(task, label) for label in labels}
             for task, labels in s.tasks.items()}
            for s in scores
        ]

    def classify(self, text: str, schema: Any) -> dict[str, dict[str, float]]:
        return self.probabilities([text], schema)[0]

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "device": str(self.classifier.device),
            "load_s": round(self.load_time_s, 1),
            "texts_scored": self.n_texts,
            "batched_calls": self.n_calls,
            "forward_s": round(self.seconds, 1),
            "ms_per_text": round(1000 * self.seconds / max(self.n_texts, 1), 1),
        }


# ---------------------------------------------------------------------------
# Schemas -- pure, no model needed
# ---------------------------------------------------------------------------


def modality_schema(task: str = "modality"):
    """One single-label task over :data:`MODALITY_LABELS`, with no instruction.

    No instruction is deliberate. Naming the claim in an instruction ("how does
    the text present: Northwind acquires Cascade?") measured *worse* than asking
    about the window alone -- the window is already cut to the edge's sentences,
    and the instruction mostly adds noise.
    """
    from gliner2.classification import ClassificationSchema

    return ClassificationSchema().single(task, list(MODALITY_LABELS))


def relation_schema(
    head_text: str, tail_text: str, relations: Sequence[Any], *, task: str = "relation"
) -> tuple[Any, dict[str, str]]:
    """The ontology's legal relations for one ordered pair, plus ``no relation``.

    Returns the schema and a ``label -> relation name`` map (``no relation``
    maps to :data:`kgx.typesafe.NO_RELATION`). Labels are bare relation names:
    the ontology's descriptions, passed as label descriptions, bought no
    consistent gain (a little on gating, mixed on selection) for three times the
    forward-pass cost, so the simpler form is used.
    """
    from gliner2.classification import ClassificationSchema

    names = {relation_label(r.name): r.name for r in relations}
    names[NO_RELATION_LABEL] = NO_RELATION
    schema = ClassificationSchema().single(
        task,
        list(names),
        instruction=f"How is {clean(head_text)} related to {clean(tail_text)}?",
    )
    return schema, names


def type_schema(
    ontology: Any, *, labels: Mapping[str, str] | None = None, task: str = "type"
) -> tuple[Any, dict[str, str]]:
    """One single-label task over the ontology's entity types.

    Returns the schema and a ``label -> type name`` map. ``labels`` maps a type
    name to the word Decide is shown (``{"geography": "place"}``); types it
    omits fall back to :func:`type_label`. No descriptions: they describe
    context, and the surface string being typed has none.

    The choice of words moves agreement with GLiNER a lot and the resolution
    outcome not at all (notebook 15 §3.2): on business news, raw names agree on
    147/214 mentions, :func:`type_label` on 145, and hand-picked everyday words
    on 163 -- and all three take :func:`second_opinion` to B-cubed 1.000.
    Better words mean fewer spurious clones, not a different answer.
    """
    from gliner2.classification import ClassificationSchema

    labels = labels or {}
    names = {clean(labels.get(e.name) or type_label(e.name)): e.name for e in ontology.entities}
    if len(names) != len(ontology.entities):
        raise ValueError("two entity types collapse to the same label")
    return ClassificationSchema().single(task, list(names)), names


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences. Crude on purpose -- ``Inc.`` before a
    capital splits -- and the same splitter :mod:`kgx.typesafe` uses."""
    spans, pos = [], 0
    for m in _SENTENCE_BREAK.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    return spans


def edge_window(graph: DocGraph, edge: Edge, *, pad: int = 0) -> str:
    """The sentences an edge spans: from its first endpoint to its last.

    Joint decoding connects mentions across paragraphs routinely (notebook 11:
    130 of 170 edges), so the window can be several sentences long. ``pad``
    adds that many sentences either side.
    """
    h, t = graph.mention(edge.head), graph.mention(edge.tail)
    lo, hi = min(h.start, t.start), max(h.end, t.end)
    spans = sentence_spans(graph.text)
    inside = [i for i, (a, b) in enumerate(spans) if b > lo and a < hi]
    if not inside:  # offsets outside the text; fall back to the raw span
        return graph.text[lo:hi]
    i0 = max(0, inside[0] - pad)
    i1 = min(len(spans) - 1, inside[-1] + pad)
    return graph.text[spans[i0][0] : spans[i1][1]]


def _argmax(probs: Mapping[str, float]) -> str:
    return max(probs, key=probs.get)


# ---------------------------------------------------------------------------
# Stage 1 -- edge checks
# ---------------------------------------------------------------------------


def judge_sentences(judge: Any, sentences: Iterable[str], *, task: str = "modality") -> list[dict[str, Any]]:
    """Modality of whole sentences, no edge in view -- for the planted traps."""
    sentences = list(sentences)
    out = judge.probabilities(sentences, modality_schema(task))
    rows = []
    for sentence, probs in zip(sentences, out):
        p = probs[task]
        best = _argmax(p)
        rows.append({"sentence": sentence, "modality": best, "confidence": p[best],
                     "p_fact": p.get(FACT, 0.0), "probabilities": dict(p)})
    return rows


@dataclass
class EdgeCheck:
    """One extracted edge, checked two ways by Decide.

    ``modality`` is about the sentences the edge came from; ``relation_choice``
    is Decide's own pick among the relations the ontology allows between the
    two endpoint types. They fail on different edges -- a hedged acquisition is
    the right relation stated the wrong way, a ``partners_with`` that should be
    ``supplies`` is a plain fact with the wrong label -- which is why the gate
    needs both.
    """

    doc_id: str
    head: str                       # surface text
    relation: str                   # what GLiNER decoded
    tail: str
    head_id: str = ""               # mention ids, for aggregation
    tail_id: str = ""
    head_type: str = ""
    tail_type: str = ""
    extractor_confidence: float = 0.0
    window: str = ""
    modality: str = ""
    modality_probabilities: dict[str, float] = field(default_factory=dict)
    relation_choice: str = ""       # a relation name, or NO_RELATION
    relation_probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def p_fact(self) -> float:
        return self.modality_probabilities.get(FACT, 0.0)

    @property
    def p_relation(self) -> float:
        """Decide's probability for the relation GLiNER decoded."""
        return self.relation_probabilities.get(self.relation, 0.0)

    @property
    def asserted(self) -> bool:
        return self.modality == FACT

    @property
    def agrees(self) -> bool:
        return self.relation_choice == self.relation

    @property
    def keep(self) -> bool:
        """The gate: stated as fact, and the relation Decide would have picked."""
        return self.asserted and self.agrees

    def key(self) -> tuple[str, str, str, str]:
        """Mention ids, not surfaces: GLiNER often emits the same surface triple
        several times in one document, anchored on different mentions, and each
        copy has its own window and its own answer."""
        return (self.doc_id, self.head_id, self.relation, self.tail_id)

    def as_record(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "gliner": round(self.extractor_confidence, 3),
            "modality": self.modality,
            "p_fact": round(self.p_fact, 3),
            "decide_pick": self.relation_choice,
            "p_relation": round(self.p_relation, 3),
            "agrees": self.agrees,
            "keep": self.keep,
        }


def check_edges(
    judge: Any,
    doc_graphs: Sequence[DocGraph],
    ontology: Any,
    *,
    include_derived: bool = False,
    pad: int = 0,
    modality_task: str = "modality",
    relation_task: str = "relation",
) -> list[EdgeCheck]:
    """Check every extracted edge for modality and relation agreement.

    Two passes over the same windows -- one modality schema shared by every
    edge, one relation schema per edge naming its endpoints -- sent to the model
    as a single batch. The two tasks are *not* combined into one schema: sharing
    a prompt, they measurably change each other's answers (the combined
    modality head loses recall on the gold triples).

    ``include_derived`` controls whether ``inverse=`` mirror edges are checked;
    they are the same fact stated twice.
    """
    legal = legal_relations(ontology)
    items: list[tuple[DocGraph, Edge, Mention, Mention, str, dict[str, str]]] = []
    texts: list[str] = []
    schemas: list[Any] = []
    shared_modality = modality_schema(modality_task)
    for graph in doc_graphs:
        for edge in graph.edges:
            if edge.derived and not include_derived:
                continue
            h, t = graph.mention(edge.head), graph.mention(edge.tail)
            window = edge_window(graph, edge, pad=pad)
            rel_schema, names = relation_schema(
                h.text, t.text, legal.get((h.type, t.type), []), task=relation_task)
            items.append((graph, edge, h, t, window, names))
            texts += [window, window]
            schemas += [shared_modality, rel_schema]

    out = judge.probabilities(texts, schemas)
    checks: list[EdgeCheck] = []
    for i, (graph, edge, h, t, window, names) in enumerate(items):
        mod = out[2 * i][modality_task]
        rel = {names[label]: p for label, p in out[2 * i + 1][relation_task].items()}
        checks.append(EdgeCheck(
            doc_id=graph.doc_id, head=h.text, relation=edge.type, tail=t.text,
            head_id=h.mention_id, tail_id=t.mention_id,
            head_type=h.type, tail_type=t.type,
            extractor_confidence=edge.confidence, window=window,
            modality=_argmax(mod), modality_probabilities=dict(mod),
            relation_choice=_argmax(rel), relation_probabilities=rel,
        ))
    return checks


def aggregate_checks(
    checks: Sequence[EdgeCheck], mention_to_canon: Mapping[str, str]
) -> dict[tuple[str, str, str], EdgeCheck]:
    """Carry each document-level check up to the canonical edge it resolved to.

    Several sentences can support one canonical edge. The rule is the one
    :attr:`kgx.graph.GraphEdge.confidence` uses -- an edge stated clearly once
    and hedged twice is still stated clearly once -- so the representative is
    the best by (kept, stated as fact, p_fact), in that order. Ranking on the
    probability alone would let a confidently *hedged* sentence speak for an
    edge another document asserts.
    """
    def rank(c: EdgeCheck) -> tuple[bool, bool, float]:
        return (c.keep, c.asserted, c.p_fact)

    best: dict[tuple[str, str, str], EdgeCheck] = {}
    for c in checks:
        h, t = mention_to_canon.get(c.head_id), mention_to_canon.get(c.tail_id)
        if h is None or t is None:
            continue
        key = (h, c.relation, t)
        if key not in best or rank(c) > rank(best[key]):
            best[key] = c
    return best


# ---------------------------------------------------------------------------
# Stage 2 -- typing ahead of blocking
# ---------------------------------------------------------------------------


@dataclass
class TypeOpinion:
    """Decide's type for one mention, from its surface string alone.

    Duck-compatible with :class:`kgx.typesafe.TypeVerdict` (``mention_id``,
    ``type``, ``alternatives``), so :func:`kgx.typesafe.soft_typed` and
    :func:`kgx.typesafe.apply_types` accept it too.
    """

    mention_id: str
    doc_id: str
    text: str
    extractor_type: str
    type: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.type != self.extractor_type

    def alternatives(self, min_prob: float = 0.25) -> list[str]:
        return [t for t, p in sorted(self.probabilities.items(), key=lambda kv: -kv[1])
                if t != self.type and p >= min_prob]

    def as_record(self) -> dict[str, Any]:
        second = self.alternatives(0.0)[:1]
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "gliner": self.extractor_type,
            "decide": self.type,
            "conf": round(self.confidence, 3),
            "runner_up": second[0] if second else "",
            "p_runner_up": round(self.probabilities.get(second[0], 0.0), 3) if second else 0.0,
            "changed": self.changed,
        }


def type_mentions(
    judge: Any,
    mentions: Sequence[Mention],
    ontology: Any,
    *,
    labels: Mapping[str, str] | None = None,
    task: str = "type",
) -> list[TypeOpinion]:
    """Type every mention by its surface string. One forward pass per distinct surface.

    Context is left out on purpose. With the sentence in view, Decide types the
    *sentence* -- ``Marcus Webb`` in a sentence about Northwind comes back
    ``company`` -- and agreement with GLiNER halves (145/214 to 72 with the
    default labels). The price is homographs: a context-free answer cannot tell
    ``Cascade`` the company from a cascade of anything. That is why the opinion
    is used as a *second* type for blocking (:func:`second_opinion`), not a
    replacement. ``labels`` is passed to :func:`type_schema`.
    """
    schema, names = type_schema(ontology, labels=labels, task=task)
    surfaces = sorted({m.text.strip() for m in mentions})
    answers = dict(zip(surfaces, judge.probabilities(surfaces, schema)))
    out: list[TypeOpinion] = []
    for m in mentions:
        probs = {names[label]: p for label, p in answers[m.text.strip()][task].items()}
        best = _argmax(probs)
        out.append(TypeOpinion(
            mention_id=m.mention_id, doc_id=m.doc_id, text=m.text,
            extractor_type=m.type, type=best, confidence=probs[best], probabilities=probs,
        ))
    return out


def second_opinion(
    mentions: Sequence[Mention],
    opinions: Sequence[TypeOpinion],
    *,
    min_prob: float | None = None,
) -> tuple[list[Mention], dict[str, str]]:
    """Offer each mention to blocking under Decide's type as well as GLiNER's.

    Every original keeps the extractor's type. Where Decide's argmax differs, a
    clone ``<mention_id>~<type>`` is added under Decide's type; with
    ``min_prob``, clones are also added for any other type carrying that much
    mass. Resolve the augmented list, then :func:`fold_clones` back onto the
    originals.

    This differs from :func:`kgx.typesafe.soft_typed`, which *replaces* the type
    with the model's argmax. A context-free argmax is wrong often enough
    (``cash`` as a financial metric, ``agreement`` as a company) that replacing
    would corrupt the graph's labels; adding only ever widens candidate
    generation, and the resolver's own scoring still has to accept every merge.
    """
    by_id = {o.mention_id: o for o in opinions}
    out: list[Mention] = []
    clone_to_original: dict[str, str] = {}
    for m in mentions:
        out.append(m)
        o = by_id.get(m.mention_id)
        if o is None:
            continue
        extra = [o.type] if o.type != m.type else []
        if min_prob is not None:
            extra += [t for t, p in sorted(o.probabilities.items(), key=lambda kv: -kv[1])
                      if p >= min_prob and t not in (m.type, o.type)]
        for type_ in extra:
            clone_id = f"{m.mention_id}~{type_}"
            out.append(replace(m, mention_id=clone_id, type=type_))
            clone_to_original[clone_id] = m.mention_id
    return out, clone_to_original


# ---------------------------------------------------------------------------
# Negative results, kept runnable
# ---------------------------------------------------------------------------


def _pair_window(graph: DocGraph, a: Mention, b: Mention) -> str:
    lo, hi = min(a.start, b.start), max(a.end, b.end)
    spans = sentence_spans(graph.text)
    inside = [i for i, (x, y) in enumerate(spans) if y > lo and x < hi]
    return graph.text[spans[inside[0]][0] : spans[inside[-1]][1]] if inside else graph.text[lo:hi]


def select_relations(
    judge: Any,
    doc_graphs: Sequence[DocGraph],
    ontology: Any,
    *,
    scope: str = "paragraph",
    task: str = "relation",
) -> list[RelationPick]:
    """Notebook 11's experiment with Decide as the selector.

    Every legal co-occurring pair (:func:`kgx.typesafe.candidate_pairs`) gets one
    relation question over the sentences it spans. Kept because the result is
    worth reproducing: Decide picks *some* relation for most pairs, and the
    selected edges score far below joint decoding.
    """
    legal = legal_relations(ontology)
    items, texts, schemas = [], [], []
    for graph in doc_graphs:
        for a, b in candidate_pairs(graph, ontology, scope=scope):
            schema, names = relation_schema(a.text, b.text, legal[(a.type, b.type)], task=task)
            items.append((graph, a, b, names))
            texts.append(_pair_window(graph, a, b))
            schemas.append(schema)
    out = judge.probabilities(texts, schemas)
    picks = []
    for (graph, a, b, names), probs in zip(items, out):
        p = {names[label]: v for label, v in probs[task].items()}
        best = _argmax(p)
        picks.append(RelationPick(
            doc_id=graph.doc_id, head=a.mention_id, tail=b.mention_id,
            head_text=a.text, tail_text=b.text, head_type=a.type, tail_type=b.type,
            relation=best, confidence=p[best], probabilities=p, scope=scope,
        ))
    return picks


def same_referent(
    judge: Any, pairs: Sequence[tuple[Mention, Mention]], *, task: str = "answer"
) -> list[float]:
    """P(yes) that two mentions name the same entity, asked over both contexts.

    The model card's "question over a passage" shape: the two context windows
    are the text, the question names both surfaces. It was the best of the pair
    shapes tried, and it is still not a matcher -- see notebook 15, §3.4.
    """
    from gliner2.classification import ClassificationSchema

    texts, schemas = [], []
    for a, b in pairs:
        texts.append(f"{a.context}\n\n{b.context}")
        schemas.append(ClassificationSchema().single(
            task, ["yes", "no"],
            instruction=(f"Do {clean(a.text)} and {clean(b.text)} refer to the same "
                         f"{type_label(a.type)}?"),
        ))
    return [probs[task]["yes"] for probs in judge.probabilities(texts, schemas)]
