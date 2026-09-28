"""GLiNER2.5 inside neo4j-graphrag: knowledge-graph construction with no LLM.

neo4j-graphrag builds a knowledge graph as a pipeline of components -- split,
embed, extract, prune, write, resolve -- and every one of them except the
extractor is model-agnostic. ``SimpleKGPipeline`` hardwires that one slot to an
LLM and exposes resolution as a boolean; the component ``Pipeline`` underneath
it does neither. This module supplies the pieces that make the no-LLM version a
drop-in:

:func:`graph_schema`
    A :class:`~kgx.ontology.Ontology` as the package's ``GraphSchema``, so one
    ontology drives GLiNER's decoding *and* the package's pruning. Node types
    declare the properties the extractor writes, because a node type that
    declares any properties defaults to ``additional_properties=False`` and
    :class:`GraphPruning` silently strips every property it was not told about.

:class:`GlinerEntityRelationExtractor`
    An ``EntityRelationExtractor`` that decodes each chunk with GLiNER2.5's
    joint entity+relation head against the ``GraphSchema`` the pipeline passes
    in. It returns the same ``Neo4jGraph`` the LLM extractor returns -- ids
    prefixed per chunk, lexical graph included -- so the package's pruner,
    writer and resolvers run on it unchanged.

:class:`TransitiveFuzzyMatchResolver`
    ``FuzzyMatchResolver`` with its merge-set consolidation replaced by a
    union-find. The built-in consolidation is not transitive: pairs
    ``{a,b}, {c,d}, {b,c}`` come out as the overlapping sets ``{a,b,c}`` and
    ``{c,d}``, and which of them survives depends on which node APOC keeps.

:class:`KgxResolver`
    :class:`kgx.EntityResolver` (normalise, block, score, cluster, canonicalise)
    as an ``EntityResolver`` component. It resolves the *mentions* behind the
    written nodes, each with its own chunk context, so resolving after every
    document gives the same clusters as resolving once. It merges each cluster
    in Neo4j keeping the canonical name, every alias, and each merged edge's
    support.

The package's resolvers merge with ``apoc.refactor.mergeNodes(..., {properties:
'discard'})``, which keeps the first node's properties and drops the rest, and
they consider every ``:__Entity__`` in the database unless given a
``filter_query``. :func:`attach_mentions` (or :class:`MentionLayer`, the same
step as a pipeline component) and :func:`read_clusters` are the measurement
harness: one ``:Mention`` node per extracted span, attached before resolution,
so that what a resolver merged can be read back and scored -- and undone.
"""

from __future__ import annotations

import re
import time
import warnings
from collections import defaultdict
from typing import Any, Iterable, Mapping, Optional, Sequence

import neo4j
from pydantic import validate_call

from neo4j_graphrag.components.base import Component, DataModel
from neo4j_graphrag.components.entity_relation_extractor import (
    EntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.components.resolver import EntityResolver, FuzzyMatchResolver
from neo4j_graphrag.components.schema import GraphSchema
from neo4j_graphrag.components.types import (
    DocumentInfo,
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
    ResolutionStats,
    TextChunk,
    TextChunks,
)

from .extract import DEFAULT_MODEL, Mention, _context_window
from .ontology import Ontology
from .resolve import EntityResolver as KgxEntityResolver

__all__ = [
    "PARTITION_LABEL",
    "PARTITION_FILTER",
    "NODE_PROPERTIES",
    "model_name",
    "graph_schema",
    "relation_options",
    "joint_schema",
    "GlinerEntityRelationExtractor",
    "consolidate_transitively",
    "TransitiveFuzzyMatchResolver",
    "KgxResolver",
    "clear_partition",
    "attach_mentions",
    "MentionLayer",
    "read_clusters",
]

#: neo4j-graphrag's ``Neo4jWriter`` puts this label on every node it writes, and
#: nothing else in this repo does, so it doubles as the notebook's partition.
PARTITION_LABEL = "__KGBuilder__"
#: A resolver ``filter_query`` that keeps resolution inside that partition.
PARTITION_FILTER = f"WHERE entity:{PARTITION_LABEL}"

_RESERVED = ("__Entity__", PARTITION_LABEL)

#: What :class:`GlinerEntityRelationExtractor` writes on every entity node.
#: Offsets are relative to the chunk the span came from.
NODE_PROPERTIES: tuple[dict[str, str], ...] = (
    {"name": "name", "type": "STRING", "description": "The span as written."},
    {"name": "confidence", "type": "FLOAT", "description": "GLiNER2.5 span confidence."},
    {"name": "start", "type": "INTEGER", "description": "Span start, chunk-relative."},
    {"name": "end", "type": "INTEGER", "description": "Span end, chunk-relative."},
)


# ---------------------------------------------------------------------------
# one ontology, two compilations
# ---------------------------------------------------------------------------

def _pascal(name: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[^0-9A-Za-z]+", name) if p)


def model_name(label: str) -> str:
    """The name GLiNER sees for a Neo4j label: ``BusinessSegment`` -> ``business_segment``.

    GLiNER encodes type names into its input alongside the text, so the wording
    is part of the prompt. Graph conventions (``PascalCase`` labels,
    ``UPPER_SNAKE`` relationship types) are not how anyone writes a concept;
    this turns both back into the lower-case words the ontology was written in.
    ``REPORTS_METRIC`` -> ``reports_metric``.
    """
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", label)
    return re.sub(r"[^0-9a-z]+", "_", words.lower()).strip("_")


def graph_schema(
    ontology: Ontology,
    *,
    node_properties: Sequence[Mapping[str, str]] = NODE_PROPERTIES,
) -> GraphSchema:
    """The ontology as a neo4j-graphrag ``GraphSchema``.

    ``PascalCase`` node labels, ``UPPER_SNAKE`` relationship types, one pattern
    per legal ``(head, relation, tail)`` triple. Every node type declares
    ``node_properties``: leave one out and :class:`GraphPruning` removes it from
    every node, logging the removal at DEBUG and nowhere else.
    """
    return GraphSchema.model_validate({
        "node_types": [
            {"label": _pascal(e.name), "description": e.description,
             "properties": [dict(p) for p in node_properties]}
            for e in ontology.entities
        ],
        "relationship_types": [
            {"label": r.name.upper(), "description": r.description}
            for r in ontology.relations
        ],
        "patterns": sorted(
            (_pascal(h), r.upper(), _pascal(t)) for h, r, t in ontology.patterns()
        ),
    })


def relation_options(ontology: Ontology) -> dict[str, dict[str, Any]]:
    """The GLiNER-only constraints ``GraphSchema`` has no field for, by relationship type.

    ``unique_head``, ``acyclic`` and friends are enforced *during* decoding.
    A ``GraphSchema`` can say which endpoints are legal, not that a company has
    one parent, so these travel beside it. ``inverse`` names are translated to
    relationship types too.
    """
    out: dict[str, dict[str, Any]] = {}
    for r in ontology.relations:
        opts: dict[str, Any] = {}
        for key in ("unique_head", "unique_tail", "acyclic", "allow_self"):
            if getattr(r, key):
                opts[key] = True
        if r.threshold is not None:
            opts["threshold"] = r.threshold
        if r.inverse:
            opts["inverse"] = r.inverse.upper()
        if opts:
            out[r.name.upper()] = opts
    return out


def joint_schema(
    schema: GraphSchema,
    *,
    options: Mapping[str, Mapping[str, Any]] | None = None,
    engine: Any = None,
    no_self_loops: bool = True,
) -> tuple[Any, dict[str, str], dict[str, str]]:
    """Compile a ``GraphSchema`` into the ``JointSchema`` GLiNER2.5 decodes against.

    Returns ``(joint_schema, node_label_for, rel_type_for)``, the two maps going
    from the names the model emits back to Neo4j labels and types.

    A relationship's head and tail types come from the schema's patterns. GLiNER
    permits every head type with every tail type, so a pattern set that is not
    a full cross product per relationship is looser here than in the schema;
    :class:`GraphPruning` is what enforces the difference. With no patterns at
    all, every relationship may connect any two node types.
    """
    if engine is not None and hasattr(engine, "create_schema"):
        js = engine.create_schema()
    else:
        from gliner2.joint_ie import JointSchema

        js = JointSchema()

    node_label_for: dict[str, str] = {}
    for nt in schema.node_types:
        name = model_name(nt.label)
        node_label_for[name] = nt.label
        js.entity(name, nt.description or None)

    heads: dict[str, set[str]] = defaultdict(set)
    tails: dict[str, set[str]] = defaultdict(set)
    for p in schema.patterns:
        heads[p.relationship].add(model_name(p.source))
        tails[p.relationship].add(model_name(p.target))

    every_type = sorted(node_label_for)
    options = options or {}
    rel_type_for: dict[str, str] = {}
    for rt in schema.relationship_types:
        name = model_name(rt.label)
        rel_type_for[name] = rt.label
        opts = dict(options.get(rt.label, {}))
        if "inverse" in opts:
            opts["inverse"] = model_name(opts["inverse"])
        js.relation(
            name,
            sorted(heads.get(rt.label, every_type)),
            sorted(tails.get(rt.label, every_type)),
            rt.description or None,
            **opts,
        )
        if no_self_loops and not opts.get("allow_self"):
            js.no_self_loops(name)
    return js, node_label_for, rel_type_for


# ---------------------------------------------------------------------------
# the extractor
# ---------------------------------------------------------------------------

class GlinerEntityRelationExtractor(EntityRelationExtractor):
    """GLiNER2.5 joint entity+relation extraction as a neo4j-graphrag component.

    Drop-in for ``LLMEntityRelationExtractor``: same inputs (``chunks``,
    ``document_info``, ``lexical_graph_config``, ``schema``), same output. The
    ``schema`` input is compiled to a GLiNER ``JointSchema`` on first use and
    cached by content, so the component follows whatever schema the pipeline
    hands it. ``examples`` is accepted and ignored -- an encoder takes no
    few-shot prompt.

    Args:
        model: a model id, a :class:`kgx.GlinerExtractor`, or a
            ``gliner2.joint_ie.JointIEEngine``. Pass a loaded one to share
            weights with the rest of a notebook.
        relation_options: GLiNER-only constraints by relationship type; see
            :func:`relation_options`.
        config: a ``gliner2.joint_ie.JointIEConfig``, passed to every decode.
        warn_words: warn when a chunk is longer than this. Relation recall falls
            off sharply past ~400 words (notebook 01); a text splitter is the
            fix, and this is how you find out you need one.
    """

    def __init__(
        self,
        model: Any = DEFAULT_MODEL,
        *,
        relation_options: Mapping[str, Mapping[str, Any]] | None = None,
        config: Any = None,
        create_lexical_graph: bool = True,
        on_error: OnError = OnError.RAISE,
        warn_words: int = 400,
    ) -> None:
        super().__init__(on_error=on_error, create_lexical_graph=create_lexical_graph)
        if isinstance(model, str):
            from .extract import GlinerExtractor

            model = GlinerExtractor(model)
        self.engine = getattr(model, "joint", model)
        self.relation_options = dict(relation_options or {})
        self.config = config
        self.warn_words = warn_words
        self._compiled: dict[str, tuple[Any, dict[str, str], dict[str, str]]] = {}
        self.last_timing: dict[str, float] = {}

    def compile(self, schema: GraphSchema) -> tuple[Any, dict[str, str], dict[str, str]]:
        key = schema.model_dump_json()
        if key not in self._compiled:
            self._compiled[key] = joint_schema(
                schema, options=self.relation_options, engine=self.engine
            )
        return self._compiled[key]

    def chunk_graph(
        self,
        result: Any,
        node_label_for: Mapping[str, str],
        rel_type_for: Mapping[str, str],
    ) -> Neo4jGraph:
        """One chunk's ``JointResult`` as a ``Neo4jGraph`` with chunk-local ids."""
        nodes = [
            Neo4jNode(
                id=e.id,
                label=node_label_for[e.type],
                properties={
                    "name": e.text,
                    "confidence": round(float(e.confidence if e.confidence is not None else 1.0), 4),
                    "start": int(e.start),
                    "end": int(e.end),
                },
            )
            for e in result.entities
        ]
        relationships = [
            Neo4jRelationship(
                start_node_id=r.head,
                end_node_id=r.tail,
                type=rel_type_for[r.type],
                properties={
                    "confidence": round(float(r.confidence if r.confidence is not None else 1.0), 4),
                    "derived": bool(r.derived),
                },
            )
            for r in result.relations
        ]
        return Neo4jGraph(nodes=nodes, relationships=relationships)

    def _handle_infeasible(self, chunk: TextChunk) -> None:
        message = (
            f"chunk {chunk.index}: JointIE could not satisfy the schema's constraints and "
            f"returned an empty assignment. That is not 'no facts in this chunk'."
        )
        if self.on_error == OnError.RAISE:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    @validate_call
    async def run(
        self,
        chunks: TextChunks,
        document_info: Optional[DocumentInfo] = None,
        lexical_graph_config: Optional[LexicalGraphConfig] = None,
        schema: Optional[GraphSchema] = None,
        examples: str = "",
        **kwargs: Any,
    ) -> Neo4jGraph:
        """Decode every chunk in one batched forward pass, then build the graph.

        Mirrors ``LLMEntityRelationExtractor.run``: optional lexical graph first,
        then per-chunk graphs with ids prefixed by the chunk id and a
        ``FROM_CHUNK`` edge from every entity to its chunk, then one combined
        graph.
        """
        if schema is None or not schema.node_types:
            raise ValueError(
                "GlinerEntityRelationExtractor needs a GraphSchema: GLiNER decodes against "
                "the types it is given and has no schema-free mode."
            )
        lexical_graph_builder = None
        lexical_graph = None
        if self.create_lexical_graph:
            config = lexical_graph_config or LexicalGraphConfig()
            lexical_graph_builder = LexicalGraphBuilder(config=config)
            lexical_graph = (
                await lexical_graph_builder.run(text_chunks=chunks, document_info=document_info)
            ).graph
        elif lexical_graph_config:
            lexical_graph_builder = LexicalGraphBuilder(config=lexical_graph_config)

        js, node_label_for, rel_type_for = self.compile(schema)
        texts = [c.text for c in chunks.chunks]
        for chunk in chunks.chunks:
            n_words = len(chunk.text.split())
            if n_words > self.warn_words:
                warnings.warn(
                    f"chunk {chunk.index} is {n_words} words; GLiNER2.5's relation recall "
                    f"falls off past ~{self.warn_words}. Use a smaller chunk_size.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        t0 = time.perf_counter()
        results = self.engine.batch_extract(texts, js, config=self.config) if texts else []
        self.last_timing = {"chunks": len(texts), "seconds": time.perf_counter() - t0}

        graph = lexical_graph.model_copy(deep=True) if lexical_graph else Neo4jGraph()
        for chunk, result in zip(chunks.chunks, results):
            if not getattr(result, "feasible", True):
                self._handle_infeasible(chunk)
            cg = self.chunk_graph(result, node_label_for, rel_type_for)
            self.update_ids(cg, chunk)
            if lexical_graph_builder:
                await lexical_graph_builder.process_chunk_extracted_entities(cg, chunk)
            graph.nodes.extend(cg.nodes)
            graph.relationships.extend(cg.relationships)
        return graph


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def consolidate_transitively(pairs: Iterable[set[str]]) -> list[set[str]]:
    """Connected components over merge pairs: a union-find, so the sets never overlap."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        items = list(pair)
        for other in items[1:]:
            ra, rb = find(items[0]), find(other)
            if ra != rb:
                parent[rb] = ra
    groups: dict[str, set[str]] = defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)
    return [g for g in groups.values() if len(g) > 1]


class TransitiveFuzzyMatchResolver(FuzzyMatchResolver):
    """``FuzzyMatchResolver`` whose merge sets are connected components.

    Same scoring (rapidfuzz ``WRatio`` on the name, per label) and the same APOC
    merge; only ``_consolidate_sets`` differs. The built-in walks the pairs once
    and adds each to the first set it touches, so a pair that bridges two sets
    already built joins one of them and the two sets end up overlapping.
    """

    _consolidate_sets = staticmethod(consolidate_transitively)


class KgxResolver(EntityResolver):
    """:class:`kgx.EntityResolver` as a neo4j-graphrag resolution component.

    It resolves **mentions, not nodes**. The package's resolvers compare entity
    nodes, and every merge they make keeps one node's ``name`` and discards the
    rest, so a node that has absorbed ``NWL``, ``Northwind`` and ``Northwind
    Logistics Inc.`` offers the next run exactly one of those strings. Here each
    written span keeps a ``:Mention`` of its own (:func:`attach_mentions`, run
    first and idempotent), with its text, offsets and chunk. Every run clusters
    all the mentions in scope -- normalise, block, score with string, embedding
    and context evidence, cluster, canonicalise -- and then merges whichever
    nodes each cluster's mentions sit on. The node holding most of a cluster's
    mentions survives, and gets the canonical name and every alias.

    So resolving after each document (as ``SimpleKGPipeline`` does) gives the
    same clusters as resolving once at the end. The cost is that each run
    re-scores the whole partition; ``kgx.CanonicalRegistry`` is the shape for
    when that stops being cheap.

    Relationships keep their evidence too. Merging collapses parallel
    relationships and keeps one of them, so before each merge
    :meth:`edge_evidence` totals what the merged edge rests on: ``support``
    (how many extracted relations) and ``confidence`` (the best of them).

    A node is never split. If a run would put two mentions an earlier run merged
    into different clusters, those clusters are merged together;
    ``last_conflicts`` counts how often.

    Blocking is by type, as the package's resolvers group by label: a ticker
    typed as a security never meets the company it stands for.

    Args:
        resolver: a configured :class:`kgx.EntityResolver`; defaults to one at
            ``threshold`` with embeddings on.
        learn_aliases: mine declarations like ``Northwind Logistics Inc.
            (NASDAQ: NWL)`` from the chunk text in scope before resolving.
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        filter_query: Optional[str] = None,
        *,
        resolver: Any = None,
        threshold: float = 0.90,
        learn_aliases: bool = True,
        neo4j_database: Optional[str] = None,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> None:
        super().__init__(driver, filter_query)
        self.resolver = resolver or KgxEntityResolver(threshold=threshold)
        self.learn = learn_aliases
        self.neo4j_database = neo4j_database
        self.lexical = lexical_graph_config
        self.last_resolution: Any = None
        self.last_conflicts = 0

    def _query(self, query: str, **params: Any) -> list[Any]:
        records, _, _ = self.driver.execute_query(
            query, parameters_=params, database_=self.neo4j_database
        )
        return records

    def read_mentions(self) -> tuple[list[Mention], dict[str, str], list[str]]:
        """Mentions in scope, the node each sits on, and the chunk texts they came from."""
        cfg = self.lexical
        rows = self._query(
            f"MATCH (entity:__Entity__) {self.filter_query or ''} "
            "MATCH (entity)-[:HAS_MENTION]->(m:Mention) "
            f"OPTIONAL MATCH (m)-[:IN_CHUNK]->(c:{cfg.chunk_node_label}) "
            f"OPTIONAL MATCH (c)-[:{cfg.chunk_to_document_relationship_type}]->"
            f"(d:{cfg.document_node_label}) "
            "RETURN elementId(entity) AS node, m.id AS id, m.label AS label, m.text AS text, "
            "       m.start AS start, m.end AS end, m.confidence AS confidence, "
            f"       c.{cfg.chunk_text_property} AS chunk, d.path AS doc"
        )
        mentions: list[Mention] = []
        node_of: dict[str, str] = {}
        chunks: dict[str, None] = {}
        for r in rows:
            if not r["label"] or not r["text"]:
                continue
            chunk = r["chunk"] or ""
            start, end = r["start"] or 0, r["end"] or 0
            if chunk:
                chunks[chunk] = None
            node_of[r["id"]] = r["node"]
            mentions.append(Mention(
                mention_id=r["id"],
                doc_id=r["doc"] or "",
                type=model_name(r["label"]),
                text=r["text"],
                start=start,
                end=end,
                confidence=r["confidence"] if r["confidence"] is not None else 1.0,
                context=_context_window(chunk, start, end) if chunk else "",
            ))
        return mentions, node_of, list(chunks)

    def merge_plan(
        self, res: Any, mentions: Sequence[Mention], node_of: Mapping[str, str]
    ) -> list[dict[str, Any]]:
        """One row per group of nodes to become one: ids (survivor first), name, aliases."""
        text = {m.mention_id: m.text for m in mentions}
        clusters = list(res.entities.values())
        # Clusters that share a node cannot be kept apart without splitting it,
        # so union them: a union-find over cluster indices, joined through nodes.
        parent = list(range(len(clusters)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        owner: dict[str, int] = {}
        for i, ent in enumerate(clusters):
            for mid in ent.mentions:
                node = node_of[mid]
                if node in owner:
                    parent[find(i)] = find(owner[node])
                else:
                    owner[node] = i
        members: dict[int, list[Any]] = defaultdict(list)
        for i, ent in enumerate(clusters):
            members[find(i)].append(ent)
        self.last_conflicts = sum(len(v) - 1 for v in members.values())

        rows = []
        for ents in members.values():
            mids = [mid for ent in ents for mid in ent.mentions]
            held: dict[str, int] = defaultdict(int)
            for mid in mids:
                held[node_of[mid]] += 1
            lead = max(ents, key=lambda e: (len(e.mentions), e.canonical))
            rows.append({
                "ids": sorted(held, key=lambda n: (-held[n], n)),
                "name": lead.canonical,
                "aliases": sorted({text[mid] for mid in mids} | {lead.canonical}),
            })
        return rows

    def edge_evidence(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """What each relationship will rest on once ``rows`` are merged: support and best confidence.

        ``mergeRels: true`` collapses parallel relationships into one and keeps
        the first one's properties, so without this an edge that five extracted
        relations support ends up with one arbitrary ``confidence`` and nothing
        saying there were five. ``support`` counts the extracted relations
        behind an edge, summed across runs; ``confidence`` is the best of them.
        """
        survivor = {node: row["ids"][0] for row in rows for node in row["ids"]}
        records = self._query(
            "MATCH (a:__Entity__)-[r]->(b:__Entity__) "
            "WHERE elementId(a) IN $ids AND elementId(b) IN $ids "
            "RETURN elementId(a) AS a, type(r) AS type, elementId(b) AS b, "
            "       coalesce(r.support, 1) AS support, r.confidence AS confidence",
            ids=list(survivor),
        )
        agg: dict[tuple[str, str, str], list[float]] = {}
        for r in records:
            key = (survivor[r["a"]], r["type"], survivor[r["b"]])
            support, best = agg.get(key, [0, 0.0])
            agg[key] = [support + r["support"], max(best, r["confidence"] or 0.0)]
        return [
            {"a": a, "type": t, "b": b, "support": int(s), "confidence": c}
            for (a, t, b), (s, c) in agg.items()
        ]

    async def run(self) -> ResolutionStats:
        attach_mentions(self.driver, self.neo4j_database, self.lexical)
        mentions, node_of, texts = self.read_mentions()
        if not mentions:
            return ResolutionStats(number_of_nodes_to_resolve=0)
        if self.learn:
            self.resolver.learn_aliases(texts)
        res = self.resolver.resolve(mentions)
        self.last_resolution = res
        rows = self.merge_plan(res, mentions, node_of)
        edges = self.edge_evidence(rows)
        # apoc.refactor.mergeNodes keeps the first node of the list, and
        # `UNWIND ... MATCH` does not promise list order, so sort by index.
        self._query(
            "UNWIND $rows AS row "
            "CALL (row) { "
            "  UNWIND range(0, size(row.ids) - 1) AS i "
            "  MATCH (n) WHERE elementId(n) = row.ids[i] "
            "  WITH n, i ORDER BY i "
            "  WITH collect(n) AS nodes "
            "  CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) "
            "  YIELD node "
            "  SET node.name = row.name, node.aliases = row.aliases "
            "  RETURN count(*) AS merged "
            "} "
            "RETURN sum(merged) AS merged",
            rows=rows,
        )
        self._query(
            "UNWIND $edges AS e "
            "MATCH (a) WHERE elementId(a) = e.a "
            "MATCH (b) WHERE elementId(b) = e.b "
            "MATCH (a)-[r]->(b) WHERE type(r) = e.type "
            "SET r.support = e.support, r.confidence = e.confidence",
            edges=edges,
        )
        return ResolutionStats(
            number_of_nodes_to_resolve=len(set(node_of.values())),
            number_of_created_nodes=len(rows),
        )


# ---------------------------------------------------------------------------
# measurement harness
# ---------------------------------------------------------------------------

def clear_partition(driver: neo4j.Driver, database: str | None = None) -> int:
    """Delete everything neo4j-graphrag wrote (``:__KGBuilder__``), plus the mention layer."""
    total = 0
    while True:
        records, _, _ = driver.execute_query(
            f"MATCH (n:{PARTITION_LABEL}) WITH n LIMIT 10000 DETACH DELETE n "
            f"RETURN count(*) AS deleted",
            database_=database,
        )
        deleted = records[0]["deleted"] if records else 0
        total += deleted
        if not deleted:
            return total


def attach_mentions(
    driver: neo4j.Driver,
    database: str | None = None,
    lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
) -> int:
    """Give every entity node not yet seen a ``:Mention`` of its own. Run before resolving.

    Why a node and not the lexical graph: ``mergeNodes(..., {mergeRels: true})``
    collapses relationships of one type to one target, so two mentions of
    Northwind in one chunk leave a single ``FROM_CHUNK`` after a merge. Each
    ``HAS_MENTION`` points at a different node and survives any merge, which
    is what lets :func:`read_clusters` see what a resolver did, and each
    mention keeps its own ``IN_CHUNK`` edge. The mention carries the partition
    label, so :func:`clear_partition` removes it too, and no ``__Entity__``
    label, so no resolver ever sees it.
    """
    cfg = lexical_graph_config
    records, _, _ = driver.execute_query(
        f"MATCH (e:__Entity__:{PARTITION_LABEL}) "
        "WHERE NOT (e)-[:HAS_MENTION]->() "
        f"OPTIONAL MATCH (e)-[:{cfg.node_to_chunk_relationship_type}]->(c:{cfg.chunk_node_label}) "
        "WITH e, head(collect(c)) AS c "
        f"CREATE (e)-[:HAS_MENTION]->(m:Mention:{PARTITION_LABEL} "
        "  {id: elementId(e), text: e.name, start: e.start, end: e.end, confidence: e.confidence, "
        "   label: [l IN labels(e) WHERE NOT l IN $reserved][0]}) "
        "FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END | "
        "  CREATE (m)-[:IN_CHUNK]->(c)) "
        "RETURN count(m) AS n",
        reserved=list(_RESERVED),
        database_=database,
    )
    return records[0]["n"]


class MentionCount(DataModel):
    mentions: int


class MentionLayer(Component):
    """:func:`attach_mentions` as a pipeline step, between the writer and the resolver."""

    def __init__(
        self,
        driver: neo4j.Driver,
        neo4j_database: Optional[str] = None,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> None:
        self.driver = driver
        self.neo4j_database = neo4j_database
        self.lexical = lexical_graph_config

    async def run(self) -> MentionCount:
        return MentionCount(mentions=attach_mentions(self.driver, self.neo4j_database, self.lexical))


def read_clusters(driver: neo4j.Driver, database: str | None = None) -> list[dict[str, Any]]:
    """One row per mention: which entity node it now belongs to, and that node's name.

    ``mention`` is the element id of the node the span was written as, unique
    within one write. ``key`` -- document, chunk index, offsets, label -- is the
    same span across rewrites of the same graph, for comparing two runs.
    """
    records, _, _ = driver.execute_query(
        f"MATCH (e:__Entity__:{PARTITION_LABEL})-[:HAS_MENTION]->(m:Mention) "
        "OPTIONAL MATCH (m)-[:IN_CHUNK]->(c)-[:FROM_DOCUMENT]->(d) "
        "RETURN m.id AS mention, m.text AS text, m.label AS label, "
        "       coalesce(d.path, '') + ':' + coalesce(toString(c.index), '') + ':' + "
        "       toString(m.start) + '-' + toString(m.end) + ':' + m.label AS key, "
        "       elementId(e) AS entity, e.name AS name",
        database_=database,
    )
    return [dict(r) for r in records]
