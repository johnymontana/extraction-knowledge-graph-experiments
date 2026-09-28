"""Tests for ``kgx.graphrag`` that need neither the model nor a database.

The extractor talks to GLiNER2.5 only through ``engine.batch_extract``, so a
scripted engine returning real ``gliner2.joint_ie`` result objects stands in
for it; the resolver talks to Neo4j only through ``driver.execute_query``, so a
fake driver that answers by query shape stands in for that. What is tested is
everything around them: that one ontology compiles to the same ``JointSchema``
by either route, that the component's output has the shape the package's
pruner, writer and lexical graph expect, and the merge bookkeeping -- survivor
choice, aliases, conflicts, edge support -- that the notebook can only show on
one corpus.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from gliner2.joint_ie.result import JointEntity, JointRelation, JointResult
from neo4j_graphrag.components.entity_relation_extractor import OnError
from neo4j_graphrag.components.graph_pruning import GraphPruning
from neo4j_graphrag.components.resolver import FuzzyMatchResolver
from neo4j_graphrag.components.schema import GraphSchema
from neo4j_graphrag.components.types import DocumentInfo, TextChunk, TextChunks

from kgx.graphrag import (
    NODE_PROPERTIES,
    GlinerEntityRelationExtractor,
    KgxResolver,
    TransitiveFuzzyMatchResolver,
    consolidate_transitively,
    graph_schema,
    joint_schema,
    model_name,
    relation_options,
)
from kgx.ontology import BUSINESS_NEWS


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label, name", [
    ("Company", "company"),
    ("BusinessSegment", "business_segment"),
    ("FinancialMetric", "financial_metric"),
    ("REPORTS_METRIC", "reports_metric"),
    ("HAS_STAKE_IN", "has_stake_in"),
])
def test_model_name_recovers_the_ontology_words(label, name):
    assert model_name(label) == name


def test_graph_schema_carries_the_whole_ontology():
    schema = graph_schema(BUSINESS_NEWS)
    assert len(schema.node_types) == len(BUSINESS_NEWS.entities) == 13
    assert len(schema.relationship_types) == len(BUSINESS_NEWS.relations) == 15
    assert len(schema.patterns) == len(BUSINESS_NEWS.patterns()) == 52
    for p in schema.patterns:
        assert BUSINESS_NEWS.permits(model_name(p.source), model_name(p.relationship),
                                     model_name(p.target))


def test_graph_schema_declares_every_property_the_extractor_writes():
    # A node type that declares any property defaults to additional_properties=False,
    # and the pruner then strips everything undeclared.
    declared = {p["name"] for p in NODE_PROPERTIES}
    for nt in graph_schema(BUSINESS_NEWS).node_types:
        assert {p.name for p in nt.properties} == declared
        assert nt.additional_properties is False


def test_relation_options_are_the_decoding_only_constraints():
    opts = relation_options(BUSINESS_NEWS)
    assert opts["SUBSIDIARY_OF"] == {"unique_head": True, "acyclic": True}
    assert opts["OFFICER_OF"] == {"unique_head": True}
    assert opts["ACQUIRES"] == {"acyclic": True}
    assert "PARTNERS_WITH" not in opts


def _canonical(js):
    d = js.to_dict()
    for spec in d["relations"].values():
        spec["head"], spec["tail"] = sorted(spec["head"]), sorted(spec["tail"])
    d["constraints"] = sorted(json.dumps(c, sort_keys=True) for c in d["constraints"])
    return d


def test_the_graphschema_route_compiles_to_the_same_joint_schema():
    via_graphrag, nodes, rels = joint_schema(graph_schema(BUSINESS_NEWS),
                                             options=relation_options(BUSINESS_NEWS))
    assert _canonical(via_graphrag) == _canonical(BUSINESS_NEWS.compile())
    assert nodes["business_segment"] == "BusinessSegment"
    assert rels["reports_metric"] == "REPORTS_METRIC"


def _schema(patterns):
    return GraphSchema.model_validate({
        "node_types": [{"label": l, "properties": [{"name": "name", "type": "STRING"}]}
                       for l in ("Person", "Company", "Place")],
        "relationship_types": [{"label": "WORKS_AT"}, {"label": "LOCATED_IN"}],
        "patterns": patterns,
    })


def test_heads_and_tails_come_from_the_patterns():
    js, _, _ = joint_schema(_schema([("Person", "WORKS_AT", "Company"),
                                     ("Company", "LOCATED_IN", "Place"),
                                     ("Person", "LOCATED_IN", "Place")]))
    rels = js.to_dict()["relations"]
    assert set(rels["works_at"]["head"]) == {"person"}
    assert set(rels["works_at"]["tail"]) == {"company"}
    assert set(rels["located_in"]["head"]) == {"company", "person"}


def test_a_relationship_without_patterns_may_join_any_two_types():
    js, _, _ = joint_schema(_schema([("Person", "WORKS_AT", "Company")]))
    rels = js.to_dict()["relations"]
    assert set(rels["located_in"]["head"]) == {"person", "company", "place"}
    assert set(rels["located_in"]["tail"]) == {"person", "company", "place"}


# ---------------------------------------------------------------------------
# extractor
# ---------------------------------------------------------------------------

class FakeEngine:
    """Answers ``batch_extract`` from a script: one ``JointResult`` per text."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def batch_extract(self, texts, schema, config=None):
        self.calls.append(list(texts))
        return [self.script(t) for t in texts]


def northwind(text, relation="operates_in"):
    return JointResult(
        text=text,
        entities=[
            JointEntity(id="e0", type="company", text="Northwind", start=0, end=9, confidence=0.91),
            JointEntity(id="e1", type="geography", text="Seattle", start=20, end=27, confidence=0.8),
        ],
        relations=[JointRelation(type=relation, head="e0", tail="e1", confidence=0.7)],
    )


def chunks(*texts):
    return TextChunks(chunks=[TextChunk(text=t, index=i, uid=f"c{i}") for i, t in enumerate(texts)])


def test_extractor_output_has_the_package_shape():
    engine = FakeEngine(northwind)
    ex = GlinerEntityRelationExtractor(engine)
    g = run(ex.run(chunks=chunks("one", "two"), schema=graph_schema(BUSINESS_NEWS),
                   document_info=DocumentInfo(path="d01", uid="doc")))

    assert engine.calls == [["one", "two"]]                       # one batched pass
    labels = sorted(n.label for n in g.nodes)
    assert labels == ["Chunk", "Chunk", "Company", "Company", "Document", "Geography", "Geography"]
    company = next(n for n in g.nodes if n.id == "c0:e0")          # chunk-prefixed ids
    assert company.label == "Company"
    assert company.properties == {"name": "Northwind", "confidence": 0.91, "start": 0, "end": 9}

    types = [r.type for r in g.relationships]
    assert types.count("OPERATES_IN") == 2
    assert types.count("FROM_CHUNK") == 4                           # every entity, to its chunk
    assert types.count("FROM_DOCUMENT") == 2
    assert types.count("NEXT_CHUNK") == 1
    edge = next(r for r in g.relationships if r.type == "OPERATES_IN")
    assert (edge.start_node_id, edge.end_node_id) == ("c0:e0", "c0:e1")
    assert edge.properties == {"confidence": 0.7, "derived": False}


def test_extractor_output_survives_pruning_intact():
    ex = GlinerEntityRelationExtractor(FakeEngine(northwind))
    schema = graph_schema(BUSINESS_NEWS)
    g = run(ex.run(chunks=chunks("one"), schema=schema))
    result = run(GraphPruning().run(graph=g, schema=schema))
    stats = result.pruning_stats
    assert (stats.pruned_nodes, stats.pruned_relationships, stats.pruned_properties) == ([], [], [])

    name_only = graph_schema(BUSINESS_NEWS, node_properties=[{"name": "name", "type": "STRING"}])
    stripped = run(GraphPruning().run(graph=g, schema=name_only))
    assert {p.item for p in stripped.pruning_stats.pruned_properties} == {"confidence", "start", "end"}


def test_the_pruner_still_enforces_patterns_the_decoder_did_not():
    # GLiNER cannot emit this (company -produces-> geography is not a legal pattern);
    # an engine that did would be caught downstream, as an LLM extractor is.
    ex = GlinerEntityRelationExtractor(FakeEngine(lambda t: northwind(t, relation="produces")))
    schema = graph_schema(BUSINESS_NEWS)
    g = run(ex.run(chunks=chunks("one"), schema=schema, lexical_graph_config=None))
    result = run(GraphPruning().run(graph=g, schema=schema))
    assert [p.pruned_reason.value for p in result.pruning_stats.pruned_relationships] == ["INVALID_PATTERN"]


def test_extractor_needs_a_schema():
    with pytest.raises(ValueError, match="GraphSchema"):
        run(GlinerEntityRelationExtractor(FakeEngine(northwind)).run(chunks=chunks("x")))


def test_extractor_compiles_each_schema_once():
    ex = GlinerEntityRelationExtractor(FakeEngine(northwind))
    schema = graph_schema(BUSINESS_NEWS)
    run(ex.run(chunks=chunks("a"), schema=schema))
    run(ex.run(chunks=chunks("b"), schema=graph_schema(BUSINESS_NEWS)))
    assert len(ex._compiled) == 1


def test_extractor_warns_on_a_long_chunk():
    ex = GlinerEntityRelationExtractor(FakeEngine(northwind), warn_words=5)
    with pytest.warns(RuntimeWarning, match="6 words"):
        run(ex.run(chunks=chunks("one two three four five six"), schema=graph_schema(BUSINESS_NEWS)))


def test_an_infeasible_chunk_is_not_silently_empty():
    infeasible = lambda text: JointResult(text=text, feasible=False)
    schema = graph_schema(BUSINESS_NEWS)
    with pytest.raises(RuntimeError, match="empty assignment"):
        run(GlinerEntityRelationExtractor(FakeEngine(infeasible)).run(chunks=chunks("x"), schema=schema))
    lenient = GlinerEntityRelationExtractor(FakeEngine(infeasible), on_error=OnError.IGNORE)
    with pytest.warns(RuntimeWarning, match="empty assignment"):
        run(lenient.run(chunks=chunks("x"), schema=schema))


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def test_consolidation_is_transitive():
    pairs = [{"a1", "a2"}, {"b1", "b2"}, {"a2", "b1"}]
    assert consolidate_transitively(pairs) == [{"a1", "a2", "b1", "b2"}]
    assert sorted(map(sorted, consolidate_transitively([{"a", "b"}, {"c", "d"}]))) == [["a", "b"], ["c", "d"]]


def test_the_transitive_resolver_only_swaps_the_consolidation():
    assert TransitiveFuzzyMatchResolver._consolidate_sets is consolidate_transitively
    assert TransitiveFuzzyMatchResolver.compute_similarity is FuzzyMatchResolver.compute_similarity
    assert TransitiveFuzzyMatchResolver.run is FuzzyMatchResolver.run


class FakeDriver:
    """Records every query and answers by the first matching substring."""

    def __init__(self, answers=None):
        self._pool = SimpleNamespace(pool_config=SimpleNamespace(user_agent=None))
        self.answers = answers or {}
        self.queries = []

    def execute_query(self, query, parameters_=None, database_=None, **params):
        self.queries.append((query, {**(parameters_ or {}), **params}))
        for needle, records in self.answers.items():
            if needle in query:
                return records, None, None
        return [{"n": 0}], None, None


def resolution(*clusters):
    """A stand-in for kgx's Resolution: ``(canonical, [mention ids])`` per entity."""
    return SimpleNamespace(entities={
        canonical: SimpleNamespace(canon_id=canonical, canonical=canonical, mentions=list(mids))
        for canonical, mids in clusters
    })


def mentions(**texts):
    return [SimpleNamespace(mention_id=mid, text=text) for mid, text in texts.items()]


def test_merge_plan_keeps_the_node_that_holds_most_of_the_cluster():
    r = KgxResolver(FakeDriver())
    node_of = {"m1": "n1", "m2": "n2", "m3": "n2", "m4": "n3"}
    rows = r.merge_plan(
        resolution(("Northwind Logistics Inc", ["m1", "m2", "m3"]), ("Halcyon", ["m4"])),
        mentions(m1="NWL", m2="Northwind", m3="Northwind Logistics Inc.", m4="Halcyon"),
        node_of,
    )
    by_name = {row["name"]: row for row in rows}
    assert by_name["Northwind Logistics Inc"]["ids"] == ["n2", "n1"]
    assert by_name["Northwind Logistics Inc"]["aliases"] == [
        "NWL", "Northwind", "Northwind Logistics Inc", "Northwind Logistics Inc."]
    assert by_name["Halcyon"]["ids"] == ["n3"]
    assert r.last_conflicts == 0


def test_merge_plan_never_splits_a_node():
    # an earlier run merged m1 and m2 into n1; this run puts them in different clusters
    r = KgxResolver(FakeDriver())
    rows = r.merge_plan(
        resolution(("A", ["m1", "m3"]), ("B", ["m2"])),
        mentions(m1="a", m2="b", m3="a2"),
        {"m1": "n1", "m2": "n1", "m3": "n2"},
    )
    assert len(rows) == 1
    assert set(rows[0]["ids"]) == {"n1", "n2"}
    assert rows[0]["name"] == "A"                       # the larger cluster names it
    assert r.last_conflicts == 1


def test_edge_evidence_totals_support_across_a_merge():
    driver = FakeDriver({"MATCH (a:__Entity__)-[r]->(b:__Entity__)": [
        {"a": "n1", "type": "ACQUIRES", "b": "n9", "support": 1, "confidence": 0.52},
        {"a": "n2", "type": "ACQUIRES", "b": "n9", "support": 3, "confidence": 0.94},
        {"a": "n2", "type": "PARTNERS_WITH", "b": "n9", "support": 1, "confidence": None},
    ]})
    r = KgxResolver(driver)
    edges = r.edge_evidence([{"ids": ["n2", "n1"]}, {"ids": ["n9"]}])
    by_type = {e["type"]: e for e in edges}
    assert by_type["ACQUIRES"] == {"a": "n2", "type": "ACQUIRES", "b": "n9", "support": 4,
                                   "confidence": 0.94}
    assert by_type["PARTNERS_WITH"]["confidence"] == 0.0


def test_read_mentions_uses_each_mentions_own_chunk():
    chunk = "SEATTLE - Northwind Logistics Inc. (NASDAQ: NWL) today announced a deal."
    driver = FakeDriver({"HAS_MENTION]->(m:Mention)": [
        {"node": "n1", "id": "m1", "label": "Company", "text": "Northwind Logistics Inc.",
         "start": 10, "end": 34, "confidence": 0.9, "chunk": chunk, "doc": "d01"},
        {"node": "n1", "id": "m2", "label": "Security", "text": "NWL",
         "start": 44, "end": 47, "confidence": None, "chunk": chunk, "doc": "d01"},
    ]})
    found, node_of, texts = KgxResolver(driver, filter_query="WHERE entity:__KGBuilder__").read_mentions()
    assert [m.type for m in found] == ["company", "security"]
    assert node_of == {"m1": "n1", "m2": "n1"}
    assert texts == [chunk]
    assert found[0].context.startswith("SEATTLE") and found[1].confidence == 1.0
    assert "WHERE entity:__KGBuilder__" in driver.queries[0][0]
