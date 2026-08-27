"""GLiNER2.5 extraction: text + ontology -> a document-local graph.

The unit of output is a :class:`DocGraph` -- mentions and edges scoped to one
document, with character offsets kept all the way through so every edge can be
traced back to the span that produced it. Document-local ids (``d01:e3``) are
deliberately *not* global: turning N document graphs into one graph is the job
of :mod:`kgx.resolve`.

Two extraction paths are wrapped here:

``JointIE``
    Decodes entities and relations together against a typed schema, subject to
    the ontology's structural constraints. Every edge it returns connects two
    entities that are actually in the result and satisfies the declared endpoint
    types. This is the path that builds the graph.

``AutoExtractor`` + ``AttributeGroup``
    A second pass that attaches span attributes (modality, direction, polarity)
    to already-extracted spans. JointSchema edges carry no properties, so
    qualifiers that would naturally be edge attributes are recovered here and
    stored on the node. Filings are full of sentences that are not assertions;
    without this, "we may face supply disruptions" becomes a fact.

Both share one set of weights -- ``JointIEEngine`` accepts an already-loaded
model, so a single 194M-parameter checkpoint serves both passes.
"""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .ontology import Ontology

__all__ = [
    "Mention",
    "Edge",
    "DocGraph",
    "GlinerExtractor",
    "DEFAULT_MODEL",
    "FAST_MODEL",
    "MULTILINGUAL_MODEL",
]

DEFAULT_MODEL = "fastino/gliner2.5-base-v1"      # 194M, English, ~407MB
FAST_MODEL = "fastino/gliner2.5-small-v1"        # 74M, English, ~296MB
MULTILINGUAL_MODEL = "fastino/gliner2.5-multi-v1"  # 287M, ~594MB


@dataclass(frozen=True)
class Mention:
    """One span of text, typed by the ontology, scoped to one document."""

    mention_id: str
    doc_id: str
    type: str
    text: str
    start: int
    end: int
    confidence: float = 1.0
    context: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "doc_id": self.doc_id,
            "type": self.type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 4),
            "context": self.context,
            **{f"attr_{k}": v for k, v in self.attrs.items()},
        }


@dataclass(frozen=True)
class Edge:
    """A typed, directed relation between two mentions in the same document."""

    doc_id: str
    type: str
    head: str  # mention_id
    tail: str  # mention_id
    confidence: float = 1.0
    derived: bool = False  # emitted by an `inverse=` mirror, not scored directly

    def key(self) -> tuple[str, str, str]:
        return (self.head, self.type, self.tail)


@dataclass
class DocGraph:
    """Everything extracted from one document."""

    doc_id: str
    text: str
    mentions: list[Mention] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    feasible: bool = True
    elapsed_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_id = {m.mention_id: m for m in self.mentions}

    def mention(self, mention_id: str) -> Mention:
        return self._by_id[mention_id]

    def reindex(self) -> "DocGraph":
        self._by_id = {m.mention_id: m for m in self.mentions}
        return self

    def by_type(self, type_: str) -> list[Mention]:
        return [m for m in self.mentions if m.type == type_]

    def triples(self) -> list[tuple[str, str, str]]:
        """Surface-form ``(head_text, relation, tail_text)`` triples."""
        return [
            (self.mention(e.head).text, e.type, self.mention(e.tail).text)
            for e in self.edges
        ]

    def describe(self, limit: int | None = None) -> str:
        rows = [
            f"  {self.mention(e.head).text!r} "
            f"-[{e.type} {e.confidence:.2f}{' derived' if e.derived else ''}]-> "
            f"{self.mention(e.tail).text!r}"
            for e in sorted(self.edges, key=lambda e: -e.confidence)[:limit]
        ]
        return (
            f"{self.doc_id}: {len(self.mentions)} mentions, {len(self.edges)} edges"
            f"{'' if self.feasible else '  [INFEASIBLE - constraints unsatisfiable]'}\n"
            + "\n".join(rows)
        )

    def validate(self, ontology: Ontology) -> list[Edge]:
        """Edges whose endpoint types the ontology does not permit.

        With ``JointIE`` this should always be empty -- typed endpoints are a hard
        decoding constraint. It is worth asserting anyway: it is exactly the check
        that fails when relations are extracted independently of entities.
        """
        return [
            e
            for e in self.edges
            if not ontology.permits(
                self.mention(e.head).type, e.type, self.mention(e.tail).type
            )
        ]


def _context_window(text: str, start: int, end: int, width: int = 90) -> str:
    """Surrounding text, used to disambiguate homographs during resolution."""
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


class GlinerExtractor:
    """Thin, opinionated wrapper over GLiNER2.5's joint and attribute passes."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        context_width: int = 90,
        long_input_warn_words: int = 400,
        **load_kwargs: Any,
    ) -> None:
        from gliner2 import AutoExtractor
        from gliner2.joint_ie import JointIEEngine

        with warnings.catch_warnings():
            # Two expected, harmless load-time warnings:
            #  - older checkpoints serialise `extra_special_tokens` as a list, and
            #    the library retries with a compatibility shim;
            #  - DeBERTa-v2/v3 has no SDPA kernel in transformers, so attention
            #    falls back to the eager implementation.
            warnings.simplefilter("ignore", UserWarning)
            warnings.filterwarnings("ignore", message=".*attn_implementation.*")
            warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")
            t0 = time.time()
            self.model = AutoExtractor.from_pretrained(model_id, **load_kwargs)
        self.load_time_s = time.time() - t0
        self.model_id = model_id
        self.context_width = context_width
        self.long_input_warn_words = long_input_warn_words
        # Shares weights with `self.model` rather than loading a second copy.
        self.joint = JointIEEngine(self.model, device=device)
        self._schema_cache: dict[str, Any] = {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = sum(p.numel() for p in self.model.parameters()) / 1e6
        return f"<GlinerExtractor {self.model_id} {n:.0f}M params loaded in {self.load_time_s:.1f}s>"

    # -- schema ----------------------------------------------------------

    def schema_for(self, ontology: Ontology):
        # Keyed on the ontology's content, not its name: two subsets both called
        # "agent_memory_subset" are different schemas, and a name-keyed cache
        # would silently hand back the wrong one.
        key = ontology.to_json(sort_keys=True)
        if key not in self._schema_cache:
            self._schema_cache[key] = ontology.compile(self.joint)
        return self._schema_cache[key]

    def config(self, **kwargs: Any):
        """Build a :class:`JointIEConfig`; see the notebook's tuning section."""
        from gliner2.joint_ie import JointIEConfig

        return JointIEConfig(**kwargs)

    # -- extraction ------------------------------------------------------

    def _to_doc_graph(
        self, result: Any, doc_id: str, text: str, elapsed: float, meta: dict[str, Any]
    ) -> DocGraph:
        mentions = [
            Mention(
                mention_id=f"{doc_id}:{e.id}",
                doc_id=doc_id,
                type=e.type,
                text=e.text,
                start=e.start,
                end=e.end,
                confidence=e.confidence if e.confidence is not None else 1.0,
                context=_context_window(text, e.start, e.end, self.context_width),
            )
            for e in result.entities
        ]
        edges = [
            Edge(
                doc_id=doc_id,
                type=r.type,
                head=f"{doc_id}:{r.head}",
                tail=f"{doc_id}:{r.tail}",
                confidence=r.confidence if r.confidence is not None else 1.0,
                derived=bool(r.derived),
            )
            for r in result.relations
        ]
        return DocGraph(
            doc_id=doc_id,
            text=text,
            mentions=mentions,
            edges=edges,
            feasible=bool(getattr(result, "feasible", True)),
            elapsed_s=elapsed,
            meta=meta,
        )

    def _warn_if_long(self, text: str, ontology: Ontology, config: Any) -> None:
        """Relation recall degrades sharply on long inputs, and can hit zero.

        Measured on this repo's corpora with a 15-relation ontology: entity
        recall holds while relation counts fall off sharply past roughly 400
        words. The notebook's length sweep bottoms out around a fifth of the
        peak rather than at zero, and an early ad-hoc run on a 518-word
        transcript did return zero edges with ``feasible=True`` -- so the
        failure mode ranges from "quietly lossy" to "indistinguishable from no
        facts here". Either way the model's relation-count head is deciding
        there are few or no instances, and it does not fail loudly.

        The fix is to window the input. ``max_len`` on the config, or
        :meth:`extract_long`, or -- best for conversations -- extract per
        episode (:meth:`kgx.coref.ConversationPreprocessor.episodes`). See the
        length sweep in the notebook for the actual curve.
        """
        n_words = len(text.split())
        max_len = getattr(config, "max_len", None) if config is not None else None
        if n_words > self.long_input_warn_words and (
            max_len is None or max_len > self.long_input_warn_words
        ):
            warnings.warn(
                f"input is {n_words} words with {len(ontology.relations)} relation types "
                f"and no windowing; relation recall degrades sharply past "
                f"~{self.long_input_warn_words} words and can lose most of a document's "
                f"relations. Pass config=JointIEConfig(max_len=512), use extract_long(), "
                f"or split into episodes.",
                RuntimeWarning,
                stacklevel=3,
            )

    def extract(
        self,
        text: str,
        ontology: Ontology,
        doc_id: str = "doc",
        *,
        config: Any = None,
        **meta: Any,
    ) -> DocGraph:
        """Extract one document into a document-local graph."""
        self._warn_if_long(text, ontology, config)
        schema = self.schema_for(ontology)
        t0 = time.time()
        result = self.joint.extract(text, schema, config=config)
        elapsed = time.time() - t0
        graph = self._to_doc_graph(result, doc_id, text, elapsed, dict(meta))
        if not graph.feasible:
            warnings.warn(
                f"{doc_id}: JointIE could not satisfy the ontology's constraints and "
                f"returned an empty assignment. This is NOT the same as 'no facts in "
                f"this text' -- relax unique_head/acyclic or check for symmetric=True.",
                RuntimeWarning,
                stacklevel=2,
            )
        return graph

    def extract_batch(
        self,
        docs: Sequence[Mapping[str, Any]],
        ontology: Ontology,
        *,
        text_key: str = "text",
        id_key: str = "doc_id",
        config: Any = None,
    ) -> list[DocGraph]:
        """Extract a corpus in one batched forward pass per batch.

        ``docs`` are mappings with at least ``text`` and ``doc_id``; every other
        key is carried through onto ``DocGraph.meta``.
        """
        schema = self.schema_for(ontology)
        texts = [d[text_key] for d in docs]
        for t in texts:
            self._warn_if_long(t, ontology, config)
        t0 = time.time()
        results = self.joint.batch_extract(texts, schema, config=config)
        elapsed = time.time() - t0
        per_doc = elapsed / max(len(texts), 1)
        graphs = [
            self._to_doc_graph(
                res,
                d.get(id_key, f"doc{i}"),
                d[text_key],
                per_doc,
                {k: v for k, v in d.items() if k not in {text_key, id_key}},
            )
            for i, (d, res) in enumerate(zip(docs, results))
        ]
        bad = [g.doc_id for g in graphs if not g.feasible]
        if bad:
            warnings.warn(
                f"{len(bad)} document(s) were infeasible under the ontology's constraints "
                f"and returned empty: {bad[:5]}{'...' if len(bad) > 5 else ''}. This is NOT "
                f"the same as 'no facts in this text'.",
                RuntimeWarning,
                stacklevel=2,
            )
        return graphs

    def extract_long(
        self,
        text: str,
        ontology: Ontology,
        doc_id: str = "doc",
        *,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        config: Any = None,
        **meta: Any,
    ) -> DocGraph:
        """Chunked extraction for documents past the ~4096-word window.

        A relation survives only if **both** endpoints land in the same chunk, so
        a fact stated across a chunk boundary is silently dropped. Raise
        ``chunk_overlap`` if long-range relations matter; the notebook measures
        the loss directly.
        """
        schema = self.schema_for(ontology)
        t0 = time.time()
        result = self.joint.extract_long(
            text, schema, config=config, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        elapsed = time.time() - t0
        meta = {**meta, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        return self._to_doc_graph(result, doc_id, text, elapsed, meta)

    # -- qualifier pass --------------------------------------------------

    def qualify(
        self,
        graph: DocGraph,
        groups: Mapping[str, Sequence[str]],
        *,
        applies_to: Mapping[str, Sequence[str]] | None = None,
        threshold: float = 0.5,
        tolerance: int = 2,
    ) -> DocGraph:
        """Attach span attributes (modality, direction, ...) to existing mentions.

        ``groups`` maps an attribute name to its label set, e.g.::

            {"modality": ["asserted", "forecast", "hypothetical", "negated"]}

        ``threshold`` is passed through to each :class:`AttributeGroup`, but note
        that for a single-label-per-group schema the model always returns its
        argmax label -- the threshold only bites on multi-label groups. Filter on
        the returned ``confidence`` instead.

        ``applies_to`` restricts each attribute to certain entity types. Results
        are joined back onto mentions by character offset (within ``tolerance``
        characters, since the two passes tokenise independently).

        Returns a new :class:`DocGraph`; mention ``attrs`` gain one entry per
        attribute, each ``{"label": ..., "confidence": ...}``.
        """
        from gliner2 import AttributeGroup

        applies_to = applies_to or {}
        target_types = sorted(
            {t for name in groups for t in applies_to.get(name, graph_types(graph))}
        )
        if not target_types:
            return graph

        schema = self.model.create_schema().entities(target_types)
        schema = schema.entity_attributes(
            {
                name: AttributeGroup(
                    list(labels),
                    applies_to=list(applies_to.get(name, target_types)),
                    threshold=threshold,
                    qualify_labels=True,
                )
                for name, labels in groups.items()
            }
        )
        raw = self.model.extract(
            graph.text, schema, include_spans=True, include_confidence=True
        )

        # index attribute results by (type, start)
        found: dict[tuple[str, int], dict[str, Any]] = {}
        for etype, items in (raw.get("entities") or {}).items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "start" not in item:
                    continue
                payload = {
                    k: v
                    for k, v in item.items()
                    if k in groups and isinstance(v, dict)
                }
                if payload:
                    found[(etype, int(item["start"]))] = payload

        out: list[Mention] = []
        for m in graph.mentions:
            hit: dict[str, Any] = {}
            for (etype, start), payload in found.items():
                if etype == m.type and abs(start - m.start) <= tolerance:
                    hit = payload
                    break
            out.append(replace(m, attrs={**m.attrs, **hit}) if hit else m)
        return DocGraph(
            doc_id=graph.doc_id,
            text=graph.text,
            mentions=out,
            edges=graph.edges,
            feasible=graph.feasible,
            elapsed_s=graph.elapsed_s,
            meta=graph.meta,
        )


def graph_types(graph: DocGraph) -> list[str]:
    return sorted({m.type for m in graph.mentions})


def mentions_frame(graphs: Iterable[DocGraph]):
    """All mentions across documents as a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame([m.as_record() for g in graphs for m in g.mentions])


def edges_frame(graphs: Iterable[DocGraph]):
    """All edges across documents as a pandas DataFrame, with surface forms."""
    import pandas as pd

    rows = []
    for g in graphs:
        for e in g.edges:
            h, t = g.mention(e.head), g.mention(e.tail)
            rows.append(
                {
                    "doc_id": g.doc_id,
                    "head_id": e.head,
                    "head": h.text,
                    "head_type": h.type,
                    "relation": e.type,
                    "tail_id": e.tail,
                    "tail": t.text,
                    "tail_type": t.type,
                    "confidence": round(e.confidence, 4),
                    "derived": e.derived,
                }
            )
    return pd.DataFrame(rows)
