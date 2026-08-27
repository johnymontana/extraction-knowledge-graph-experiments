"""Loading a :class:`~kgx.graph.KnowledgeGraph` into Neo4j, idempotently.

Three things make this less trivial than a loop of ``MERGE`` statements.

**Dynamic labels.** The ontology decides the node labels at runtime, and Cypher
does not let you parameterise a label. Neo4j 5.26 has no dynamic-label syntax
(that arrives in 2025.x as ``MERGE (n:$(label))``), so the options are one
statement per label or APOC. This module uses ``apoc.merge.node`` /
``apoc.merge.relationship`` when APOC is present and falls back to grouping by
label and emitting one parameterised statement per group when it is not —
either way the write is batched with ``UNWIND``.

**Idempotency.** Everything is ``MERGE`` on ``canon_id``, so re-running after a
re-extraction updates in place. That matters more than it sounds: an extraction
pipeline gets re-run constantly while the ontology is being tuned, and a loader
that duplicates on every pass makes the graph useless as a working surface.

**Evidence.** Relationship properties carry the document ids, confidences and
text snippets that justified the edge, so a suspicious relationship in the
browser is one click from the sentence that produced it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .graph import KnowledgeGraph

ENTITY_LABEL = "__Entity__"
"""Shared secondary label on every canonical entity.

Cypher indexes are per label, so a graph with thirteen ontology labels would
otherwise need thirteen vector indexes to be searchable as a whole. One shared
label makes a single index cover every entity type, which is what the
neo4j-graphrag retrievers expect and what makes "find the entity most like this
question" a one-index query. The ontology label is still there for querying.
"""

__all__ = [
    "ENTITY_LABEL",
    "Neo4jConfig",
    "supports_dynamic_labels",
    "has_apoc",
    "label_for",
    "rel_type_for",
    "connect",
    "clear_database",
    "drop_search_indexes",
    "create_constraints",
    "load_graph",
    "load_documents",
    "schema_summary",
    "counts",
]


@dataclass(frozen=True)
class Neo4jConfig:
    """Connection settings, overridable from the environment."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"
    database: str = "neo4j"

    @classmethod
    def from_env(cls, **overrides: Any) -> "Neo4jConfig":
        import os

        return cls(
            uri=overrides.get("uri") or os.environ.get("NEO4J_URI", cls.uri),
            user=overrides.get("user") or os.environ.get("NEO4J_USER", cls.user),
            password=overrides.get("password") or os.environ.get("NEO4J_PASSWORD", cls.password),
            database=overrides.get("database") or os.environ.get("NEO4J_DATABASE", cls.database),
        )


def label_for(entity_type: str) -> str:
    """``business_segment`` -> ``BusinessSegment``."""
    parts = [p for p in re.split(r"[^0-9A-Za-z]+", entity_type) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or "Entity"


def rel_type_for(relation: str) -> str:
    """``partners_with`` -> ``PARTNERS_WITH``."""
    return re.sub(r"[^\w]", "_", relation).upper()


def connect(
    config: Neo4jConfig | None = None,
    *,
    notifications: str | None = "WARNING",
    **overrides: Any,
):
    """Open a verified driver. Caller owns closing it.

    ``notifications`` maps to the driver's minimum notification severity.
    ``"OFF"`` is worth setting in a notebook: querying a relationship type the
    data happens not to contain is a legitimate exploratory move, and the server
    answers every one of them with a multi-line ``UNRECOGNIZED`` warning on
    stderr that buries the actual output.
    """
    import neo4j

    config = config or Neo4jConfig.from_env(**overrides)
    kwargs: dict[str, Any] = {}
    if notifications is not None:
        # notifications_min_severity filters server-side; warn_notification_severity
        # only controls the client warning and still logs the notification.
        kwargs["notifications_min_severity"] = neo4j.NotificationMinimumSeverity(notifications)
    driver = neo4j.GraphDatabase.driver(
        config.uri, auth=(config.user, config.password), **kwargs
    )
    driver.verify_connectivity()
    return driver


def supports_dynamic_labels(driver, database: str = "neo4j") -> bool:
    """Does this server accept ``MERGE (n:$(label))``? (Neo4j 5.26+.)

    Probed rather than version-sniffed: the feature landed across 5.24-5.26 for
    different clauses, and asking the server is cheaper than encoding that table.
    """
    try:
        driver.execute_query(
            "MERGE (n:$($l) {probe: $p}) DELETE n",
            l="__DynamicLabelProbe__", p=1, database_=database,
        )
        return True
    except Exception:
        return False


def has_apoc(driver, database: str = "neo4j") -> bool:
    try:
        records, _, _ = driver.execute_query(
            "CALL apoc.help('merge') YIELD name RETURN count(*) AS n", database_=database
        )
        return bool(records and records[0]["n"])
    except Exception:
        return False


def clear_database(driver, database: str = "neo4j", *, batch: int = 10_000) -> int:
    """Delete every node and relationship. Batched, so a big graph does not OOM."""
    total = 0
    while True:
        records, _, _ = driver.execute_query(
            "MATCH (n) WITH n LIMIT $batch DETACH DELETE n RETURN count(*) AS deleted",
            batch=batch, database_=database,
        )
        deleted = records[0]["deleted"] if records else 0
        total += deleted
        if not deleted:
            return total


def drop_search_indexes(driver, database: str = "neo4j") -> list[str]:
    """Drop every VECTOR and FULLTEXT index, leaving constraint-backed ones alone.

    Neo4j rejects a second index on the same (label, property), and
    ``CREATE ... IF NOT EXISTS`` treats an *equivalent* index under a different
    name as already satisfied -- so it succeeds, creates nothing, and the name
    you asked for does not exist. The retriever then fails with "No index with
    name ... found" and the cause is three cells earlier.

    Clearing search indexes as part of a reset makes the starting state knowable.
    """
    records, _, _ = driver.execute_query(
        "SHOW INDEXES YIELD name, type, owningConstraint "
        "WHERE type IN ['VECTOR', 'FULLTEXT'] AND owningConstraint IS NULL "
        "RETURN name",
        database_=database,
    )
    dropped = [r["name"] for r in records]
    for name in dropped:
        driver.execute_query(f"DROP INDEX {name} IF EXISTS", database_=database)
    return dropped


def create_constraints(driver, graph: KnowledgeGraph, database: str = "neo4j") -> list[str]:
    """A uniqueness constraint per label, plus one for documents.

    Constraints first, always: they create the backing index that makes the
    subsequent ``MERGE`` calls fast instead of a full scan per row.
    """
    labels = sorted({label_for(e.type) for e in graph.entities.values()})
    labels += [ENTITY_LABEL, "Document"]
    made = []
    for label in labels:
        key = "doc_id" if label == "Document" else "canon_id"
        name = f"{re.sub(r'[^a-z0-9]', '', label.lower())}_{key}"
        driver.execute_query(
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE",
            database_=database,
        )
        made.append(name)
    return made


def _entity_rows(graph: KnowledgeGraph) -> list[dict[str, Any]]:
    rows = []
    for e in graph.entities.values():
        props: dict[str, Any] = {
            "name": e.canonical,
            "type": e.type,
            "aliases": list(e.aliases),
            "n_mentions": len(e.mentions),
            "n_docs": len(set(e.docs)),
            "docs": sorted(set(e.docs)),
        }
        for k, v in e.attrs.items():
            props[k] = v.get("label") if isinstance(v, dict) else v
        rows.append({
            "canon_id": e.canon_id,
            "label": label_for(e.type),
            "labels": [label_for(e.type), ENTITY_LABEL],
            "props": props,
        })
    return rows


def _edge_rows(graph: KnowledgeGraph, min_confidence: float, min_support: int) -> list[dict[str, Any]]:
    rows = []
    for edge in graph.edges:
        if edge.confidence < min_confidence or edge.support < min_support:
            continue
        if edge.head not in graph.entities or edge.tail not in graph.entities:
            continue
        evidence = sorted(edge.evidence, key=lambda ev: -ev.confidence)
        rows.append({
            "head": edge.head,
            "tail": edge.tail,
            "rel_type": rel_type_for(edge.type),
            "props": {
                "relation": edge.type,
                "confidence": round(edge.confidence, 4),
                "support": edge.support,
                "n_docs": edge.n_docs,
                "docs": sorted({ev.doc_id for ev in edge.evidence}),
                "evidence": [ev.snippet for ev in evidence[:3]],
                "derived": edge.derived,
            },
        })
    return rows


def _chunks(rows: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def load_graph(
    driver,
    graph: KnowledgeGraph,
    *,
    database: str = "neo4j",
    min_confidence: float = 0.0,
    min_support: int = 1,
    batch_size: int = 500,
    method: str = "auto",
) -> dict[str, Any]:
    """Write the graph. Returns what was written and which path was taken.

    ``method`` is ``"auto"`` (use ``$()`` dynamic labels when the server supports
    them), ``"dynamic"``, or ``"grouped"`` for the portable path.
    """
    create_constraints(driver, graph, database)

    nodes = _entity_rows(graph)
    edges = _edge_rows(graph, min_confidence, min_support)
    mode = method if method != "auto" else ("dynamic" if supports_dynamic_labels(driver, database) else "grouped")

    if mode == "dynamic":
        node_q = """
        UNWIND $rows AS row
        MERGE (n:$(row.label) {canon_id: row.canon_id})
        SET n:$($entity_label), n += row.props
        """
        for chunk in _chunks(nodes, batch_size):
            driver.execute_query(node_q, rows=list(chunk), entity_label=ENTITY_LABEL,
                                 database_=database)

        edge_q = """
        UNWIND $rows AS row
        MATCH (h:$($entity_label) {canon_id: row.head})
        MATCH (t:$($entity_label) {canon_id: row.tail})
        MERGE (h)-[r:$(row.rel_type)]->(t)
        SET r += row.props
        """
        for chunk in _chunks(edges, batch_size):
            driver.execute_query(edge_q, rows=list(chunk), entity_label=ENTITY_LABEL,
                                 database_=database)
    else:
        # Portable fallback: group by label / relationship type so the identifier
        # can be interpolated once per group rather than parameterised per row.
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in nodes:
            by_label.setdefault(row["label"], []).append(row)
        for label, rows in by_label.items():
            q = (f"UNWIND $rows AS row MERGE (n:{label} {{canon_id: row.canon_id}}) "
                 f"SET n:{ENTITY_LABEL}, n += row.props")
            for chunk in _chunks(rows, batch_size):
                driver.execute_query(q, rows=list(chunk), database_=database)

        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in edges:
            by_type.setdefault(row["rel_type"], []).append(row)
        for rel_type, rows in by_type.items():
            q = (f"UNWIND $rows AS row "
                 f"MATCH (h:{ENTITY_LABEL} {{canon_id: row.head}}) "
                 f"MATCH (t:{ENTITY_LABEL} {{canon_id: row.tail}}) "
                 f"MERGE (h)-[r:{rel_type}]->(t) SET r += row.props")
            for chunk in _chunks(rows, batch_size):
                driver.execute_query(q, rows=list(chunk), database_=database)

    return {"nodes": len(nodes), "relationships": len(edges), "method": mode}


def load_documents(
    driver,
    documents: Sequence[Mapping[str, Any]],
    graph: KnowledgeGraph | None = None,
    *,
    database: str = "neo4j",
    batch_size: int = 200,
) -> dict[str, int]:
    """Load the source documents and link entities to the documents they came from.

    Retrieval needs text to retrieve. The extracted graph holds names and
    relations, not prose, so the corpus is loaded alongside it as ``(:Document)``
    nodes and joined by the ``docs`` list already carried on every entity. That
    ``MENTIONED_IN`` edge is what lets a vector hit on a document jump straight
    into the structured graph.
    """
    rows = [
        {
            "doc_id": d["doc_id"],
            "props": {k: v for k, v in d.items() if k != "doc_id" and not isinstance(v, (dict, list))},
        }
        for d in documents
    ]
    for chunk in _chunks(rows, batch_size):
        driver.execute_query(
            "UNWIND $rows AS row MERGE (d:Document {doc_id: row.doc_id}) SET d += row.props",
            rows=list(chunk), database_=database,
        )

    links = 0
    if graph is not None:
        known = {d["doc_id"] for d in documents}
        pairs = [
            {"canon_id": e.canon_id, "doc_id": doc}
            for e in graph.entities.values()
            for doc in sorted(set(e.docs))
            if doc in known
        ]
        for chunk in _chunks(pairs, batch_size):
            driver.execute_query(
                "UNWIND $rows AS row "
                "MATCH (n {canon_id: row.canon_id}), (d:Document {doc_id: row.doc_id}) "
                "MERGE (n)-[:MENTIONED_IN]->(d)",
                rows=list(chunk), database_=database,
            )
        links = len(pairs)
    return {"documents": len(rows), "mention_links": links}


def counts(driver, database: str = "neo4j") -> dict[str, Any]:
    records, _, _ = driver.execute_query(
        "MATCH (n) RETURN count(n) AS nodes", database_=database)
    nodes = records[0]["nodes"]
    records, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN count(r) AS rels", database_=database)
    rels = records[0]["rels"]
    records, _, _ = driver.execute_query(
        "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC",
        database_=database)
    by_label = {r["l"]: r["c"] for r in records}
    records, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC",
        database_=database)
    by_type = {r["t"]: r["c"] for r in records}
    return {"nodes": nodes, "relationships": rels, "by_label": by_label, "by_type": by_type}


def schema_summary(driver, database: str = "neo4j"):
    """The graph's actual shape, as a DataFrame of (head)-[rel]->(tail) patterns.

    Read back from the data rather than from the ontology -- this is what is
    really in the database, which is the thing to check after a load and the
    thing to hand to a text2cypher prompt.
    """
    import pandas as pd

    records, _, _ = driver.execute_query(
        """
        MATCH (h)-[r]->(t)
        RETURN labels(h)[0] AS head, type(r) AS relationship, labels(t)[0] AS tail,
               count(*) AS count
        ORDER BY count DESC
        """,
        database_=database,
    )
    return pd.DataFrame([dict(r) for r in records])
