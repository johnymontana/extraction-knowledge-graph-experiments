"""Fast, model-free tests for the pure-python half of ``kgx``.

Nothing here loads GLiNER2.5 or a sentence-transformer: every resolver runs with
``use_embeddings=False``, and mentions, edges, document graphs and resolutions
are hand-built dataclasses. ``gliner2.joint_ie`` is imported (by
``Ontology.compile``) because constructing a ``JointSchema`` downloads nothing.

Covered: ontology construction/validation/compilation, name normalisation and
entity resolution, canonical graph assembly and Cypher export, the bi-temporal
fact store, and the deterministic coreference layers. Two tests are marked
and the deterministic coreference layers.
"""

from __future__ import annotations

import json

import pytest

from kgx.coref import USER_CANON_ID, ConversationPreprocessor
from kgx.extract import DocGraph, Edge, Mention
from kgx.graph import Evidence, GraphEdge, KnowledgeGraph, build_graph, to_cypher
from kgx.ontology import AGENT_MEMORY, BUSINESS_NEWS, EntityType, Ontology, RelationType
from kgx.resolve import (
    CanonicalEntity,
    CanonicalRegistry,
    EntityResolver,
    Resolution,
    ScoredPair,
    bcubed,
    detect_aliases,
    is_acronym_of,
    normalize,
)
from kgx.temporal import Fact, TemporalGraph, explicit_alternatives, graph_alternatives

# ---------------------------------------------------------------------------
# hand-built object builders
# ---------------------------------------------------------------------------


def mention(mention_id, type_, text, *, doc_id="d1", start=0, confidence=1.0, context=""):
    """A :class:`Mention` with an end offset derived from the surface."""
    return Mention(
        mention_id=mention_id,
        doc_id=doc_id,
        type=type_,
        text=text,
        start=start,
        end=start + len(text),
        confidence=confidence,
        context=context,
    )


def entity(canon_id, type_, canonical, *, aliases=(), mentions=(), docs=()):
    return CanonicalEntity(
        canon_id=canon_id,
        type=type_,
        canonical=canonical,
        aliases=list(aliases),
        mentions=list(mentions),
        docs=list(docs),
    )


def resolution(entities, mention_to_canon, *, threshold=0.9):
    return Resolution({e.canon_id: e for e in entities}, dict(mention_to_canon), [], threshold)


def string_only_resolver(**kwargs):
    """An :class:`EntityResolver` that never touches a sentence-transformer."""
    kwargs.setdefault("use_embeddings", False)
    return EntityResolver(**kwargs)


def toy_ontology():
    """Three node labels, two edge types, one of them multi-tailed."""
    return Ontology(
        name="toy",
        entities=(EntityType("a"), EntityType("b"), EntityType("c")),
        relations=(
            RelationType("r1", ("a",), ("b", "c")),
            RelationType("r2", ("a",), ("c",)),
        ),
    )


def mirrored_ontology():
    return Ontology(
        name="inv",
        entities=(EntityType("a"), EntityType("b")),
        relations=(
            RelationType("fwd", ("a",), ("b",), inverse="bwd"),
            RelationType("bwd", ("b",), ("a",), inverse="fwd"),
        ),
    )


# ===========================================================================
# kgx.ontology
# ===========================================================================


def test_entity_and_relation_dataclasses_validate_their_own_fields():
    with pytest.raises(ValueError, match="non-empty"):
        EntityType("   ")
    with pytest.raises(ValueError, match="at least one head and tail"):
        RelationType("r", (), ("a",))

    coerced = RelationType("r", "a", "b")  # a bare string is a one-element tuple

    assert (coerced.head, coerced.tail) == (("a",), ("b",))


def test_ontology_rejects_a_relation_over_an_undeclared_entity_type():
    with pytest.raises(ValueError, match=r"undeclared entity types: \['zzz'\]"):
        Ontology("bad", (EntityType("a"),), (RelationType("r", ("a",), ("zzz",)),))


def test_ontology_rejects_duplicate_relation_and_entity_names():
    with pytest.raises(ValueError, match="duplicate relation name 'r'"):
        Ontology(
            "bad",
            (EntityType("a"),),
            (RelationType("r", ("a",), ("a",)), RelationType("r", ("a",), ("a",))),
        )
    with pytest.raises(ValueError, match="duplicate entity type names"):
        Ontology("bad", (EntityType("a"), EntityType("a")), ())


def test_ontology_requires_an_inverse_pair_to_mirror_its_endpoints():
    not_mirrored = (
        RelationType("fwd", ("a",), ("b",), inverse="bwd"),
        RelationType("bwd", ("a",), ("b",), inverse="fwd"),  # should be b -> a
    )

    with pytest.raises(ValueError, match="not mirrored"):
        Ontology("bad", (EntityType("a"), EntityType("b")), not_mirrored)

    mirrored = mirrored_ontology()
    assert mirrored.relation_names == ("fwd", "bwd")
    assert mirrored.relation("fwd").inverse == "bwd"


def test_patterns_and_permits_agree_on_which_triples_are_legal():
    onto = toy_ontology()

    assert onto.patterns() == {("a", "r1", "b"), ("a", "r1", "c"), ("a", "r2", "c")}
    assert onto.permits("a", "r1", "b") is True
    assert onto.permits("b", "r1", "a") is False
    assert onto.permits("a", "nope", "b") is False
    with pytest.raises(KeyError):
        onto.relation("nope")


def test_subset_prunes_endpoint_types_and_drops_relations_left_empty():
    onto = toy_ontology()

    sub = onto.subset(entities=["a", "b"])

    assert sub.name == "toy_subset"
    assert sub.entity_names == ("a", "b")
    assert sub.relation_names == ("r1",)  # r2 only ever pointed at "c"
    assert sub.relation("r1").tail == ("b",)  # "c" pruned off r1's tail


def test_subset_by_relation_clears_a_dangling_inverse_and_checks_its_arguments():
    sub = mirrored_ontology().subset(relations=["fwd"], name="one_way")

    assert sub.name == "one_way"
    assert sub.relation_names == ("fwd",)
    assert sub.relation("fwd").inverse is None
    with pytest.raises(KeyError, match="zzz"):
        toy_ontology().subset(entities=["a", "zzz"])


def test_ontology_round_trips_through_dict_and_json():
    for onto in (AGENT_MEMORY, BUSINESS_NEWS):
        assert Ontology.from_dict(onto.to_dict()) == onto
        assert Ontology.from_dict(json.loads(onto.to_json())) == onto

    lists_not_tuples = Ontology.from_dict(
        {
            "name": "x",
            "entities": [{"name": "a"}, {"name": "b"}],
            "relations": [{"name": "r", "head": "a", "tail": "b"}],
        }
    )

    assert lists_not_tuples.relation("r").head == ("a",)
    assert lists_not_tuples.no_self_loops is True


def test_compile_emits_one_joint_schema_spec_per_declared_type():
    schema = AGENT_MEMORY.compile()

    assert len(schema.entity_specs) == len(AGENT_MEMORY.entities) == 13
    assert len(schema.relation_specs) == len(AGENT_MEMORY.relations) == 16
    assert {s.name for s in schema.entity_specs} == set(AGENT_MEMORY.entity_names)
    assert {s.name for s in schema.relation_specs} == set(AGENT_MEMORY.relation_names)


def test_compile_translates_structural_flags_into_schema_constraints():
    onto = Ontology(
        name="c",
        entities=(EntityType("a"), EntityType("b")),
        relations=(
            RelationType("uh", ("a",), ("b",), unique_head=True),
            RelationType("ut", ("a",), ("b",), unique_tail=True),
            RelationType("dag", ("a",), ("a",), acyclic=True, allow_self=True),
        ),
    )

    schema = onto.compile()
    specs = {s.name: s for s in schema.relation_specs}

    assert specs["uh"].max_per_head == 1
    assert specs["ut"].max_per_tail == 1
    assert specs["dag"].allow_self is True
    # no_self_loops is applied per relation, so a relation that explicitly
    # permits self-loops is exempt rather than being overridden by a global one
    by_kind = {}
    for c in schema.constraints:
        by_kind.setdefault(c.__class__.__name__, set()).add(getattr(c, "relation", None))
    assert by_kind["NoSelfLoops"] == {"uh", "ut"}
    assert by_kind["AcyclicRelation"] == {"dag"}
    # the `symmetric` flag is documented as broken upstream; compile must never set it
    assert not any(s.symmetric for s in specs.values())

    loops_allowed = Ontology(
        name="loops",
        entities=(EntityType("a"),),
        relations=(RelationType("r", ("a",), ("a",), allow_self=True),),
        no_self_loops=False,
    )
    assert loops_allowed.compile().constraints == ()

    mirrored = {s.name: s for s in mirrored_ontology().compile().relation_specs}
    assert (mirrored["fwd"].inverse, mirrored["bwd"].inverse) == ("bwd", "fwd")


def test_compile_uses_an_engines_create_schema_when_one_is_given():
    class FakeEngine:
        def __init__(self):
            self.calls = 0

        def create_schema(self):
            from gliner2.joint_ie import JointSchema

            self.calls += 1
            return JointSchema()

    engine = FakeEngine()

    schema = toy_ontology().compile(engine)

    assert engine.calls == 1
    assert len(schema.entity_specs) == 3


def test_to_mermaid_declares_every_node_and_deduplicates_edges():
    onto = Ontology(
        name="m",
        entities=(EntityType("a"), EntityType("b")),
        relations=(RelationType("r", ("a", "a"), ("b",)),),
    )

    lines = onto.to_mermaid().splitlines()

    assert lines[0] == "graph LR"
    assert "  a([a])" in lines and "  b([b])" in lines
    assert lines.count("  a -->|r| b") == 1


# ===========================================================================
# kgx.resolve -- normalize
# ===========================================================================


def test_normalize_moves_an_org_legal_suffix_into_its_own_field():
    norm = normalize("Northwind Logistics Inc.", "organization")

    assert norm.key == "northwind logistics"
    assert norm.display == "Northwind Logistics"
    assert norm.suffix == "Inc"
    assert norm.tokens == ("northwind", "logistics")
    # a punctuation variant of the same firm lands on the same matching key
    assert normalize("Northwind Logistics, Inc.", "company").key == norm.key


def test_normalize_splits_a_person_into_given_and_family_names():
    full = normalize("Dr. Anita Desai Jr.", "person")
    bare = normalize("Anita", "person")

    assert full.key == "anita desai"
    assert (full.first, full.last) == ("Anita", "Desai")
    assert (bare.first, bare.last) == ("Anita", "Anita")
    assert bare.tokens == ("anita",)


def test_normalize_drops_leading_determiners_and_possessive_endings():
    determined = normalize("The Acme Corporation", "company")

    assert determined.key == "acme"
    assert determined.display == "Acme"
    assert determined.suffix == "Corporation"
    assert normalize("Priya's", "user").key == "priya"
    assert normalize("Acme’s revenue", "").key == "acme revenue"


def test_normalize_composes_decomposed_unicode_accents():
    precomposed = normalize("Caf\u00e9 Rouge", "company")  # e-acute as a single code point
    decomposed = normalize("Cafe\u0301 Rouge", "company")  # "e" + a combining acute

    assert precomposed.key == decomposed.key == "café rouge"
    assert precomposed.tokens == decomposed.tokens


def test_normalize_survives_empty_and_single_character_input():
    empty = normalize("", "company")
    single = normalize("A", "")

    assert (empty.key, empty.tokens, empty.entropy) == ("", (), 0.0)
    assert empty.is_acronym is False
    assert (single.display, single.key, single.entropy) == ("A", "a", 0.0)
    assert single.is_acronym is False  # one letter is too short to be an acronym


def test_normalize_flags_a_short_all_caps_token_as_an_acronym():
    assert normalize("NWL", "company").is_acronym is True
    assert normalize("Northwind", "company").is_acronym is False
    assert normalize("Northwind Logistics", "company").initials == "nl"


# ===========================================================================
# kgx.resolve -- acronyms and corpus-mined aliases
# ===========================================================================


def test_is_acronym_of_accepts_word_internal_letters():
    assert is_acronym_of("NWL", ("northwind", "logistics")) is True
    assert is_acronym_of("NOR", ("northwind",)) is True


def test_is_acronym_of_rejects_the_obvious_false_positives():
    assert is_acronym_of("IBM", ("northwind", "logistics")) is False  # first letters differ
    assert is_acronym_of("NWX", ("northwind", "logistics")) is False  # "x" is absent
    assert is_acronym_of("NW", ("northwind",)) is False  # single token needs 3+ letters
    assert is_acronym_of("N", ("northwind",)) is False  # too short
    assert is_acronym_of("NWL", ()) is False  # nothing to abbreviate


def test_detect_aliases_mines_ticker_and_quoted_short_form_declarations():
    corpus = [
        "Northwind Logistics Inc. (NASDAQ: NWL) reported record volume.",
        'Northwind Logistics Inc. ("Northwind") said today.',
        "Halcyon Semiconductor Corp (HLCN) fell four percent.",
    ]

    assert detect_aliases(corpus) == {
        ("northwind logistics", "nwl"),
        ("northwind logistics", "northwind"),
        ("halcyon semiconductor", "hlcn"),
    }
    # "the Company" is a role, not a name
    assert detect_aliases(['Acme Corp (the "Company") announced.']) == set()


# ===========================================================================
# kgx.resolve -- EntityResolver
# ===========================================================================


def test_resolver_merges_mentions_with_an_identical_normalised_key():
    mentions = [
        mention("a", "company", "Northwind Logistics Inc."),
        mention("b", "company", "Northwind Logistics, Inc."),
    ]

    result = string_only_resolver().resolve(mentions)

    (pair,) = result.pairs
    assert (pair.reason, pair.score) == ("exact normalised match", 1.0)
    assert len(result.entities) == 1
    assert result.canon_for("a") == result.canon_for("b")


def test_low_entropy_names_are_discounted_and_npm_stays_apart_from_pnpm():
    resolver = string_only_resolver()
    npm, pnpm = mention("a", "tool", "npm"), mention("b", "tool", "pnpm")

    scored = resolver.score_pair(
        npm, pnpm, normalize("npm", "tool"), normalize("pnpm", "tool"), 0.0, "test"
    )
    result = resolver.resolve([npm, pnpm])

    assert scored.fuzz > 0.9  # the two strings look nearly identical
    assert "low-entropy" in scored.reason
    assert scored.score == pytest.approx(0.4275 - resolver.low_entropy_penalty)
    assert scored.score < resolver.threshold
    assert len(result.entities) == 2
    assert result.canon_for("a") != result.canon_for("b")


def test_two_similar_acronyms_are_blocked_together_but_not_merged():
    result = string_only_resolver().resolve(
        [mention("a", "company", "NWL"), mention("b", "company", "NWT")]
    )

    (pair,) = result.pairs
    assert pair.block == "init"  # shared initials proposed the pair
    assert "low-entropy" in pair.reason
    assert pair.score < 0.9
    assert len(result.entities) == 2


def test_resolver_merges_an_org_name_that_is_a_whole_token_prefix():
    # The prefix rule needs corroboration from context; shared context supplies it.
    shared = "Northwind Logistics reported quarterly revenue and operates across North America"
    mentions = [
        mention("a", "company", "Northwind", context=shared),
        mention("b", "company", "Northwind Logistics Inc.", context=shared),
    ]

    result = string_only_resolver().resolve(mentions)

    (pair,) = result.pairs
    assert pair.reason == "org name is a token prefix, context agrees"
    assert pair.rule == 0.92
    # canonicalisation prefers the longest surface, and keeps the legal suffix
    (ent,) = result.entities.values()
    assert ent.canonical == "Northwind Logistics Inc"
    assert ent.canon_id == "company:northwind-logistics-inc"
    assert "Northwind" in ent.aliases


def test_raising_the_threshold_above_the_prefix_rule_keeps_orgs_apart():
    shared = "Northwind Logistics reported quarterly revenue and operates across North America"
    mentions = [
        mention("a", "company", "Northwind", context=shared),
        mention("b", "company", "Northwind Logistics", context=shared),
    ]
    resolver = string_only_resolver()

    merged = resolver.resolve(mentions, threshold=0.90)
    split = resolver.resolve(mentions, threshold=0.95)

    assert len(merged.entities) == 1
    assert len(split.entities) == 2
    assert split.stats["accepted_pairs"] == 0
    assert split.pairs[0].score == pytest.approx(0.92)  # re-clustering, not re-scoring


def test_an_uncorroborated_org_prefix_lands_in_the_review_band_not_a_merge():
    """"Apple" and "Apple Bank" share a prefix and are different companies."""
    mentions = [
        mention("a", "company", "Apple",
                context="Apple reported record iPhone sales in Cupertino this quarter."),
        mention("b", "company", "Apple Bank",
                context="Apple Bank for Savings is a retail bank headquartered in New York."),
    ]

    result = string_only_resolver().resolve(mentions)

    assert len(result.entities) == 2
    (pair,) = result.pairs
    assert "unconfirmed" in pair.reason
    assert pair.score < result.threshold


def test_resolver_merges_a_bare_surname_onto_a_full_person_name():
    mentions = [mention("a", "person", "Dev Shah"), mention("b", "person", "Shah")]

    result = string_only_resolver().resolve(mentions)

    (pair,) = result.pairs
    assert pair.reason == "surname match, compatible given name"
    assert len(result.entities) == 1
    assert result.canon_for("b") == "person:dev-shah"


def test_resolver_refuses_to_merge_on_a_given_name_alone():
    mentions = [mention("a", "person", "Dev"), mention("b", "person", "Dev Kumar")]

    result = string_only_resolver().resolve(mentions)

    (pair,) = result.pairs
    assert pair.rule == 0.5
    assert "ambiguous" in pair.reason
    assert len(result.entities) == 2


def test_deterministic_evidence_short_circuits_the_weighted_score():
    acro, full = mention("a", "company", "NWL"), mention("b", "company", "Northwind Logistics")
    na, nb = normalize("NWL", "company"), normalize("Northwind Logistics", "company")

    expansion = string_only_resolver().score_pair(acro, full, na, nb, 0.0, "test")
    declared = string_only_resolver(aliases=[("northwind logistics", "nwl")]).score_pair(
        acro, full, na, nb, 0.0, "test"
    )

    assert expansion.reason == "acronym expansion"
    assert (expansion.rule, expansion.score) == (1.0, pytest.approx(0.97))
    assert declared.reason == "alias declared in corpus text"
    assert declared.score == 1.0


def test_cluster_is_connected_components_over_pairs_at_or_above_the_threshold():
    pairs = [
        ScoredPair("a", "b", "t", "exact", score=0.95),
        ScoredPair("b", "c", "t", "exact", score=0.91),
        ScoredPair("d", "e", "t", "fuzz", score=0.89),  # just below the line
    ]

    clusters = EntityResolver.cluster(["a", "b", "c", "d", "e"], pairs, 0.90)

    assert sorted(sorted(c) for c in clusters) == [["a", "b", "c"], ["d"], ["e"]]


# ===========================================================================
# kgx.resolve -- B-cubed
# ===========================================================================


def test_bcubed_is_perfect_on_an_exact_clustering():
    assert bcubed({"a": "1", "b": "1", "c": "2"}, {"a": "g", "b": "g", "c": "h"}) == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "n": 3,
    }
    assert bcubed({}, {"a": "g"}) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    assert bcubed({"a": "1", "z": "2"}, {"a": "g"})["n"] == 1  # keys are intersected


def test_bcubed_penalises_recall_when_one_gold_cluster_is_split():
    # gold {a, b, c}; predicted {a, b} and {c}.
    # precision: every predicted neighbour is genuinely a neighbour     -> 3/3 = 1.0
    # recall:    (2/3 + 2/3 + 1/3) / 3 = 5/9                            -> 0.5556
    # f1:        2 * 1 * (5/9) / (1 + 5/9) = 10/14                      -> 0.7143
    assert bcubed({"a": "1", "b": "1", "c": "2"}, {"a": "g", "b": "g", "c": "g"}) == {
        "precision": 1.0,
        "recall": 0.5556,
        "f1": 0.7143,
        "n": 3,
    }


def test_bcubed_penalises_precision_on_an_over_merge():
    # gold {a, b} and {c}; predicted one cluster {a, b, c} -- the mirror image.
    assert bcubed({"a": "1", "b": "1", "c": "1"}, {"a": "g", "b": "g", "c": "h"}) == {
        "precision": 0.5556,
        "recall": 1.0,
        "f1": 0.7143,
        "n": 3,
    }


# ===========================================================================
# kgx.resolve -- CanonicalRegistry
# ===========================================================================


def test_registry_links_a_second_episode_onto_the_entity_from_the_first():
    registry = CanonicalRegistry(string_only_resolver())
    registry.add_episode(
        [mention("e1:m1", "company", "Northwind Logistics Inc.", doc_id="e1")],
        episode_id="ep1",
    )

    assigned = registry.add_episode(
        [mention("e2:m1", "company", "Northwind Logistics", doc_id="e2")],
        episode_id="ep2",
    )

    assert list(registry.entities) == ["company:northwind-logistics-inc"]
    assert assigned == {"e2:m1": "company:northwind-logistics-inc"}
    ent = registry.entities["company:northwind-logistics-inc"]
    assert ent.mentions == ["e1:m1", "e2:m1"]
    assert ent.docs == ["e1", "e2"]
    assert "Northwind Logistics" in ent.aliases
    assert (ent.first_seen, ent.last_seen) == ("ep1", "ep2")


def test_registry_still_creates_a_node_for_an_unseen_mention():
    registry = CanonicalRegistry(string_only_resolver())
    registry.add_episode(
        [mention("e1:m1", "company", "Northwind Logistics Inc.", doc_id="e1")],
        episode_id="ep1",
    )

    registry.add_episode(
        [
            mention("e2:m1", "company", "Northwind Logistics", doc_id="e2"),
            mention("e2:m2", "company", "Cascade Freight", doc_id="e2"),
        ],
        episode_id="ep2",
    )

    assert set(registry.entities) == {
        "company:northwind-logistics-inc",
        "company:cascade-freight",
    }
    assert registry.mention_to_canon["e2:m2"] == "company:cascade-freight"
    assert registry.as_resolution().stats["canonical_entities"] == 2


# ===========================================================================
# kgx.graph
# ===========================================================================

ACQUISITION_TEXT = "Northwind acquires Cascade. Northwind Logistics acquires Cascade Freight."


def acquisition_fixture():
    """One document asserting the same acquisition twice, under two spellings."""
    mentions = [
        mention("d1:m1", "company", "Northwind", start=0),
        mention("d1:m2", "company", "Cascade", start=19),
        mention("d1:m3", "company", "Northwind Logistics", start=28),
        mention("d1:m4", "company", "Cascade Freight", start=57),
    ]
    edges = [
        Edge("d1", "acquires", "d1:m1", "d1:m2", 0.90),
        Edge("d1", "acquires", "d1:m3", "d1:m4", 0.70),
    ]
    doc = DocGraph("d1", ACQUISITION_TEXT, mentions, edges)
    res = resolution(
        [
            entity("company:northwind", "company", "Northwind Logistics",
                   aliases=["Northwind"], mentions=["d1:m1", "d1:m3"], docs=["d1"]),
            entity("company:cascade", "company", "Cascade Freight",
                   aliases=["Cascade"], mentions=["d1:m2", "d1:m4"], docs=["d1"]),
        ],
        {
            "d1:m1": "company:northwind",
            "d1:m3": "company:northwind",
            "d1:m2": "company:cascade",
            "d1:m4": "company:cascade",
        },
    )
    onto = Ontology(
        name="news",
        entities=(EntityType("company"), EntityType("person")),
        relations=(
            RelationType("acquires", ("company",), ("company",)),
            RelationType("officer_of", ("person",), ("company",)),
        ),
    )
    return doc, res, onto


def test_build_graph_collapses_duplicate_edges_and_keeps_every_occurrence():
    doc, res, onto = acquisition_fixture()

    kg = build_graph([doc], res, onto)

    (edge,) = kg.edges
    assert edge.key() == ("company:northwind", "acquires", "company:cascade")
    assert edge.support == 2
    assert edge.n_docs == 1
    assert [ev.head_span for ev in edge.evidence] == [(0, 9), (28, 47)]
    assert all("acquires" in ev.snippet for ev in edge.evidence)
    assert kg.stats["raw_edges"] == 2
    assert kg.stats["canonical_edges"] == 1
    # the same evidence is reachable by canonical name, strongest first
    assert [ev.confidence for ev in kg.evidence_for("Northwind", "acquires", "Cascade")] == [
        0.90,
        0.70,
    ]


def test_build_graph_aggregates_confidence_with_max_not_mean():
    doc, res, onto = acquisition_fixture()

    (edge,) = build_graph([doc], res, onto).edges

    assert edge.confidence == pytest.approx(0.90)  # not the 0.80 mean
    assert sorted(ev.confidence for ev in edge.evidence) == [0.70, 0.90]


def test_build_graph_drops_self_loops_and_edges_it_cannot_resolve():
    doc, res, onto = acquisition_fixture()
    doc.edges.append(Edge("d1", "acquires", "d1:m1", "d1:m3", 0.6))  # both -> northwind
    doc.mentions.append(mention("d1:m9", "company", "Ghost", start=0))
    doc.reindex()
    doc.edges.append(Edge("d1", "acquires", "d1:m1", "d1:m9", 0.9))  # Ghost never resolved

    dropped = build_graph([doc], res, onto)
    kept = build_graph([doc], res, onto, drop_self_loops=False)

    assert dropped.stats["dropped_self_loops"] == 1
    assert dropped.stats["dropped_unresolved"] == 1
    assert [e.key() for e in dropped.edges] == [
        ("company:northwind", "acquires", "company:cascade")
    ]
    assert kept.stats["dropped_self_loops"] == 0
    assert any(e.head == e.tail for e in kept.edges)


def test_build_graph_folds_pinned_mentions_into_a_single_entity():
    docs = [
        DocGraph(
            "d1",
            "Priya Raman works at Northwind.",
            [
                mention("d1:u", "user", "Priya Raman", doc_id="d1", start=0),
                mention("d1:o", "organization", "Northwind", doc_id="d1", start=21),
            ],
            [Edge("d1", "works_at", "d1:u", "d1:o", 0.9)],
        ),
        DocGraph(
            "d2",
            "Priya works at Northwind.",
            [
                mention("d2:u", "user", "Priya", doc_id="d2", start=0),
                mention("d2:o", "organization", "Northwind", doc_id="d2", start=15),
            ],
            [Edge("d2", "works_at", "d2:u", "d2:o", 0.8)],
        ),
    ]
    res = resolution(
        [
            entity("user:priya-raman", "user", "Priya Raman", mentions=["d1:u"], docs=["d1"]),
            entity("user:priya", "user", "Priya", mentions=["d2:u"], docs=["d2"]),
            entity("organization:northwind", "organization", "Northwind",
                   mentions=["d1:o", "d2:o"], docs=["d1", "d2"]),
        ],
        {
            "d1:u": "user:priya-raman",
            "d2:u": "user:priya",
            "d1:o": "organization:northwind",
            "d2:o": "organization:northwind",
        },
    )

    kg = build_graph(docs, res, pin={"d1:u": USER_CANON_ID, "d2:u": USER_CANON_ID})

    assert set(kg.entities) == {USER_CANON_ID, "organization:northwind"}
    pinned = kg.entities[USER_CANON_ID]
    assert pinned.type == "user"
    assert pinned.mentions == ["d1:u", "d2:u"]
    assert pinned.docs == ["d1", "d2"]
    assert pinned.canonical not in pinned.aliases
    (edge,) = kg.edges
    assert edge.key() == (USER_CANON_ID, "works_at", "organization:northwind")
    assert edge.support == 2
    assert edge.n_docs == 2


def test_find_and_the_edge_accessors_walk_the_canonical_graph():
    doc, res, onto = acquisition_fixture()
    kg = build_graph([doc], res, onto)
    acquisition = ("company:northwind", "acquires", "company:cascade")

    assert [e.canon_id for e in kg.find("northwind logistics")] == ["company:northwind"]
    assert [e.canon_id for e in kg.find("NORTHWIND")] == ["company:northwind"]  # alias
    assert kg.find("Northwind Logistics", type_="person") == []
    assert kg.find("nobody") == []
    assert [e.key() for e in kg.out_edges("company:northwind")] == [acquisition]
    assert kg.out_edges("company:northwind", relation="officer_of") == []
    assert [e.key() for e in kg.in_edges("company:cascade")] == [acquisition]
    assert kg.neighbors("company:northwind") == ["company:cascade"]
    assert kg.neighbors("company:cascade") == ["company:northwind"]
    assert kg.name("company:cascade") == "Cascade Freight"
    assert kg.name("nope") == "nope"


def test_triples_filter_on_confidence_support_and_relation():
    doc, res, onto = acquisition_fixture()
    kg = build_graph([doc], res, onto)

    assert kg.triples() == [("Northwind Logistics", "acquires", "Cascade Freight")]
    assert kg.triples(min_support=2) == kg.triples()
    assert kg.triples(min_support=3) == []
    assert kg.triples(min_confidence=0.95) == []
    assert kg.triples(relation="officer_of") == []


def test_violations_reports_edges_the_ontology_forbids():
    doc, res, onto = acquisition_fixture()
    doc.edges.append(Edge("d1", "officer_of", "d1:m1", "d1:m2", 0.9))
    kg = build_graph([doc], res, onto)

    (bad,) = kg.violations()

    assert bad.type == "officer_of"  # officer_of is person -> company, not company -> company
    assert bad.head == "company:northwind"
    assert build_graph([doc], res).violations() == []  # nothing to check against


def quoting_graph():
    """A graph whose names and evidence contain double quotes."""
    ents = {
        "company:acme": entity("company:acme", "company", 'Acme "Best" Corp',
                               aliases=['A "B"'], mentions=["d1:x"], docs=["d1"]),
        "company:cascade": entity("company:cascade", "company", "Cascade Freight",
                                  mentions=["d1:y"], docs=["d1"]),
    }
    edges = [
        GraphEdge("company:acme", "competes_with", "company:cascade",
                  [Evidence("d1", (0, 4), (5, 9), 0.8, 'he said "hi"')])
    ]
    return KnowledgeGraph(ents, edges)


def test_to_cypher_writes_only_idempotent_merge_statements():
    script = to_cypher(quoting_graph())

    assert "CREATE (" not in script
    assert script.count("MERGE (n:Company {canon_id:") == 2
    assert "MERGE (a)-[r:COMPETES_WITH]->(b)" in script
    assert "CREATE CONSTRAINT company_id IF NOT EXISTS" in script

    filtered = to_cypher(quoting_graph(), min_confidence=0.9, include_constraints=False)
    assert "CREATE CONSTRAINT" not in filtered
    assert "COMPETES_WITH" not in filtered  # the only edge scores 0.8
    assert "MERGE (n:Company" in filtered  # nodes are still written


def test_to_cypher_escapes_quotes_in_names_aliases_and_snippets():
    script = to_cypher(quoting_graph())

    assert r'n.name = "Acme \"Best\" Corp"' in script
    assert r'n.aliases = ["A \"B\""]' in script
    assert r'r.evidence = ["he said \"hi\""]' in script


# ===========================================================================
# kgx.temporal
# ===========================================================================


def fact(relation, tail, tail_name, *, tail_type="tool", **kwargs):
    return Fact(
        head="user:self",
        relation=relation,
        tail=tail,
        head_name="Priya Raman",
        tail_name=tail_name,
        tail_type=tail_type,
        **kwargs,
    )


def test_repeating_a_triple_corroborates_it_instead_of_appending():
    store = TemporalGraph()
    store.assert_fact(fact("located_in", "place:berlin", "Berlin", tail_type="place",
                           episode_id="e1", confidence=0.7, valid_from="2026-01-01"))

    superseded = store.assert_fact(
        fact("located_in", "place:berlin", "Berlin", tail_type="place",
             episode_id="e2", confidence=0.9, valid_from="2026-02-01")
    )

    assert superseded == []
    assert len(store.facts) == 1
    assert store.facts[0].support == 2
    assert store.facts[0].confidence == pytest.approx(0.9)
    assert store.facts[0].episode_id == "e1"  # the original record is the one kept


def test_a_single_valued_relation_supersedes_the_previous_value():
    store = TemporalGraph()
    old = fact("located_in", "place:berlin", "Berlin", tail_type="place",
               episode_id="e1", valid_from="2026-01-01")
    store.assert_fact(old)
    new = fact("located_in", "place:lisbon", "Lisbon", tail_type="place",
               episode_id="e2", valid_from="2026-03-01")

    superseded = store.assert_fact(new)

    assert superseded == [old]
    assert old.is_current is False
    # two timelines, two stamps: valid_until is world time (when Berlin stopped
    # being true), superseded_at is transaction time (when we learned it)
    assert old.valid_until == "2026-03-01"
    assert old.superseded_at == new.recorded_at
    assert old.superseded_by == new.fact_id
    assert [f.tail_name for f in store.current()] == ["Lisbon"]
    assert store.contradictions() == [(old, new)]


def test_alternatives_policy_supersedes_only_the_values_that_actually_compete():
    store = TemporalGraph(alternative_fn=explicit_alternatives([["npm", "pnpm", "yarn"]]))
    store.assert_fact(fact("prefers", "tool:npm", "npm", episode_id="e1"))

    competing = store.assert_fact(fact("prefers", "tool:pnpm", "pnpm", episode_id="e2"))
    unrelated = store.assert_fact(fact("prefers", "tool:dark", "dark mode", episode_id="e3"))

    assert [f.tail_name for f in competing] == ["npm"]
    assert unrelated == []
    assert sorted(f.tail_name for f in store.current()) == ["dark mode", "pnpm"]
    assert store.candidates == []


def test_alternatives_policy_without_a_function_queues_a_candidate_conflict():
    store = TemporalGraph()
    old = fact("prefers", "tool:npm", "npm", episode_id="e1")
    store.assert_fact(old)
    new = fact("prefers", "tool:pnpm", "pnpm", episode_id="e2")

    superseded = store.assert_fact(new)

    assert superseded == []
    assert sorted(f.tail_name for f in store.current()) == ["npm", "pnpm"]
    assert store.candidates == [(old, new)]


def test_accumulate_policy_never_supersedes_and_never_queues():
    store = TemporalGraph()
    store.assert_fact(fact("has_constraint", "c:1", "no meetings before 10",
                           tail_type="constraint", episode_id="e1"))

    superseded = store.assert_fact(
        fact("has_constraint", "c:2", "EU data residency", tail_type="constraint", episode_id="e2")
    )

    assert superseded == []
    assert len(store.current()) == 2
    assert store.candidates == []


def backdated_store():
    """Berlin was learned promptly; the move to Lisbon is reported three months late."""
    store = TemporalGraph()
    store.assert_fact(fact("located_in", "place:berlin", "Berlin", tail_type="place",
                           episode_id="e1", valid_from="2026-01-01",
                           recorded_at="2026-01-05T00:00:00+00:00"))
    store.assert_fact(fact("located_in", "place:lisbon", "Lisbon", tail_type="place",
                           episode_id="e2", valid_from="2026-03-01",
                           recorded_at="2026-06-01T00:00:00+00:00"))
    return store


def test_as_of_gives_different_answers_on_the_two_timelines():
    store = backdated_store()

    world = [f.tail_name for f in store.as_of("2026-04-01", timeline="valid")]
    believed = [f.tail_name for f in store.as_of("2026-04-01", timeline="transaction")]

    # In April she really was in Lisbon, but the system did not find out until
    # June -- so on the transaction timeline it still believed Berlin. That
    # divergence is the entire reason to keep both timelines.
    assert world == ["Lisbon"]
    assert believed == ["Berlin"]

    # and by July the two agree again
    assert [f.tail_name for f in store.as_of("2026-07-01", timeline="valid")] == ["Lisbon"]
    assert [f.tail_name for f in store.as_of("2026-07-01", timeline="transaction")] == ["Lisbon"]


def test_as_of_agrees_before_the_move_and_after_the_correction_is_recorded():
    store = backdated_store()

    assert [f.tail_name for f in store.as_of("2026-02-15")] == ["Berlin"]
    assert [f.tail_name for f in store.as_of("2026-02-15", timeline="transaction")] == ["Berlin"]
    assert [f.tail_name for f in store.as_of("2026-07-01", timeline="transaction")] == ["Lisbon"]
    assert store.as_of("2026-07-01", head="somebody else") == []
    assert len(store.as_of("2026-07-01", head="Priya Raman")) == 1
    with pytest.raises(ValueError, match="'valid' or 'transaction'"):
        store.as_of("2026-07-01", timeline="bogus")


def test_history_returns_every_assertion_oldest_first():
    store = backdated_store()

    entries = store.history("user:self", "located_in")

    assert [f.tail_name for f in entries] == ["Berlin", "Lisbon"]
    assert [f.valid_from for f in entries] == ["2026-01-01", "2026-03-01"]
    assert store.history("user:self", "prefers") == []


def test_memory_prompt_groups_current_facts_and_omits_superseded_ones():
    store = backdated_store()
    store.assert_fact(fact("has_skill", "skill:go", "Go", tail_type="skill", episode_id="e3"))

    prompt = store.memory_prompt("Priya Raman")

    assert "- located in: Lisbon" in prompt
    assert "- has skill: Go" in prompt
    assert "Berlin" not in prompt
    assert TemporalGraph().memory_prompt("Nobody") == "No stored facts about Nobody."


def replaces_graph():
    ents = {
        "tool:npm": entity("tool:npm", "tool", "npm"),
        "tool:pnpm": entity("tool:pnpm", "tool", "pnpm"),
        "tool:jenkins": entity("tool:jenkins", "tool", "Jenkins"),
    }
    edges = [
        GraphEdge("tool:pnpm", "replaces", "tool:npm", [Evidence("d1", (0, 4), (5, 9), 0.95)]),
        GraphEdge("tool:jenkins", "replaces", "tool:npm", [Evidence("d1", (0, 4), (5, 9), 0.20)]),
    ]
    return KnowledgeGraph(ents, edges)


def test_graph_alternatives_reads_replaces_edges_symmetrically():
    is_alternative = graph_alternatives(replaces_graph())

    assert is_alternative("pnpm", "npm") is True
    assert is_alternative("npm", "pnpm") is True  # the mirror pair is registered too
    assert is_alternative("NPM", "PnPm") is True
    assert is_alternative("npm", "Berlin") is False
    # the Jenkins edge scores 0.20 and is below the default confidence floor
    assert is_alternative("Jenkins", "npm") is False
    assert graph_alternatives(replaces_graph(), min_confidence=0.1)("Jenkins", "npm") is True


def test_explicit_alternatives_only_links_names_inside_one_group():
    is_alternative = explicit_alternatives([["npm", "pnpm"], ["Python", "Go"]])

    assert is_alternative("npm", "PNPM") is True
    assert is_alternative("npm", "Go") is False
    assert is_alternative("npm", "something unlisted") is False


# ===========================================================================
# kgx.coref
# ===========================================================================


def session(*texts, session_id="s1", **meta):
    """Turns default to the user; pass ``("assistant", text)`` for the other side."""
    turns = []
    for item in texts:
        speaker, text = item if isinstance(item, tuple) else ("user", item)
        turns.append({"speaker": speaker, "text": text})
    return {"session_id": session_id, "turns": turns, **meta}


def test_render_rewrites_first_person_and_possessives_to_the_user_name():
    pre = ConversationPreprocessor("Priya Raman")

    rendered = pre.render(session("I use Postgres for my side project."))

    assert rendered.text == (
        "Priya Raman: Priya Raman use Postgres for Priya Raman's side project."
    )
    assert [(r.original, r.replacement) for r in rendered.rewrites] == [
        ("I", "Priya Raman"),
        ("my", "Priya Raman's"),
    ]
    assert {r.layer for r in rendered.rewrites} == {"first_person"}


def test_render_leaves_plurals_and_assistant_turns_in_the_first_person():
    pre = ConversationPreprocessor("Priya Raman")

    plural = pre.render(session("We migrated our cluster."))
    assistant = pre.render(session(("assistant", "I can help with my tooling.")))

    assert plural.text == "Priya Raman: We migrated our cluster."  # "we"/"our" mean the team
    assert plural.rewrites == []
    assert assistant.text == "Assistant: I can help with my tooling."

    user_only = ConversationPreprocessor("Priya Raman", include_assistant=False)
    dropped = user_only.render(session("I ship on Friday.", ("assistant", "Noted.")))
    assert dropped.turn_texts == ["Priya Raman: Priya Raman ship on Friday."]


def test_render_expands_a_bare_first_name_when_the_roster_is_unambiguous():
    pre = ConversationPreprocessor("Priya Raman", roster=["Dev Shah"])

    rendered = pre.render(session("Dev shipped the API."))

    assert rendered.text == "Priya Raman: Dev Shah shipped the API."
    (rewrite,) = rendered.rewrites
    assert (rewrite.layer, rewrite.replacement) == ("first_name", "Dev Shah")


def test_render_refuses_to_expand_a_first_name_two_people_share():
    pre = ConversationPreprocessor("Priya Raman", roster=["Dev Shah", "Dev Kumar"])

    rendered = pre.render(session("Dev shipped the API."))

    assert rendered.text == "Priya Raman: Dev shipped the API."
    assert [r for r in rendered.rewrites if r.layer == "first_name"] == []


def test_render_expands_a_first_name_typed_without_its_accents():
    pre = ConversationPreprocessor("Priya Raman", roster=["José Álvarez"])

    rendered = pre.render(session("Jose owns the deploy."))

    assert rendered.text == "Priya Raman: José Álvarez owns the deploy."
    assert rendered.rewrites[0].original == "Jose"


def test_render_binds_a_sentence_initial_pronoun_to_a_unique_antecedent():
    pre = ConversationPreprocessor("Priya Raman", roster=["Dev Shah", "Alex Ng"])

    rendered = pre.render(
        session("Dev Shah runs the platform team.", "He owns the migration.")
    )

    assert rendered.turn_texts[1] == "Priya Raman: Dev Shah owns the migration."
    (rewrite,) = [r for r in rendered.rewrites if r.layer == "pronoun"]
    assert (rewrite.original, rewrite.replacement) == ("He", "Dev Shah")


def test_render_leaves_a_pronoun_alone_without_a_single_recent_antecedent():
    pre = ConversationPreprocessor("Priya Raman", roster=["Dev Shah", "Alex Ng"])
    narrow = ConversationPreprocessor("Priya Raman", roster=["Dev Shah"], pronoun_window=1)

    ambiguous = pre.render(
        session("Dev Shah runs platform.", "Alex Ng runs infra.", "He owns the migration.")
    )
    stale = narrow.render(
        session("Dev Shah runs platform.", "Unrelated turn.", "He owns the migration.")
    )

    assert ambiguous.turn_texts[2] == "Priya Raman: He owns the migration."
    assert [r for r in ambiguous.rewrites if r.layer == "pronoun"] == []
    assert stale.turn_texts[2] == "Priya Raman: He owns the migration."


def test_layers_can_be_ablated_one_at_a_time():
    turns = session("I use Postgres for my side project.")

    speaker_only = ConversationPreprocessor("Priya Raman", layers=("speaker",)).render(turns)
    nothing = ConversationPreprocessor("Priya Raman", layers=()).render(turns)

    assert speaker_only.text == "Priya Raman: I use Postgres for my side project."
    assert speaker_only.rewrites == []
    assert nothing.text == "I use Postgres for my side project."


def six_turn_session():
    return session(*[f"turn {i}" for i in range(6)], session_id="s6", timestamp="2026-03-01")


def test_episodes_slice_a_session_into_overlapping_windows_carrying_provenance():
    pre = ConversationPreprocessor("Priya Raman")

    episodes = pre.episodes([six_turn_session()], window=4, stride=2)

    assert [e["doc_id"] for e in episodes] == ["s6:w00", "s6:w02"]
    assert [e["turn_start"] for e in episodes] == [0, 2]
    assert [e["text"].splitlines()[0] for e in episodes] == [
        "Priya Raman: turn 0",
        "Priya Raman: turn 2",
    ]
    assert episodes[0]["text"].count("\n") == 3  # four turns per window
    assert episodes[0]["session_id"] == "s6"
    assert episodes[0]["timestamp"] == "2026-03-01"
    assert episodes[0]["drop_speakers"] == ["Assistant"]


def test_episodes_emit_one_short_window_for_a_session_below_the_window_size():
    pre = ConversationPreprocessor("Priya Raman")
    short = session("turn 0", "turn 1", "turn 2", session_id="s7")

    episodes = pre.episodes([short], window=4, drop_speaker_mentions=False)

    assert len(episodes) == 1
    assert episodes[0]["doc_id"] == "s7:w00"
    assert episodes[0]["text"].count("\n") == 2
    assert episodes[0]["timestamp"] is None
    assert episodes[0]["drop_speakers"] == []


def test_clean_mentions_removes_speaker_labels_and_the_edges_that_used_them():
    doc = DocGraph(
        "d1",
        "Assistant: Dev Shah owns the User Story.",
        [
            mention("m1", "person", "Assistant"),
            mention("m2", "person", "Dev Shah"),
            mention("m3", "user", "User"),
            mention("m4", "artifact", "User Story"),  # not a bare label; must survive
        ],
        [
            Edge("d1", "collaborates_with", "m1", "m2"),
            Edge("d1", "produced", "m2", "m4"),
        ],
    )
    pre = ConversationPreprocessor("Priya Raman")

    removed = pre.clean_mentions([doc])

    assert removed == 2
    assert [m.text for m in doc.mentions] == ["Dev Shah", "User Story"]
    assert [e.key() for e in doc.edges] == [("m2", "produced", "m4")]
    assert doc.mention("m2").text == "Dev Shah"  # the index was rebuilt
    with pytest.raises(KeyError):
        doc.mention("m1")


def test_clean_mentions_respects_a_custom_assistant_name():
    doc = DocGraph(
        "d1",
        "Ada: hello",
        [mention("m1", "person", "Ada"), mention("m2", "person", "Dev Shah")],
        [],
    )

    removed = ConversationPreprocessor("Priya Raman", assistant_name="Ada").clean_mentions([doc])

    assert removed == 1
    assert [m.text for m in doc.mentions] == ["Dev Shah"]


# ===========================================================================
# ===========================================================================


def test_episodes_cover_every_turn_of_a_session():
    pre = ConversationPreprocessor("Priya Raman")

    episodes = pre.episodes([six_turn_session()], window=4)

    covered = {line for e in episodes for line in e["text"].splitlines()}
    assert covered == {f"Priya Raman: turn {i}" for i in range(6)}


def test_a_corroborated_fact_stays_visible_on_the_valid_timeline():
    store = TemporalGraph()
    store.assert_fact(fact("located_in", "place:berlin", "Berlin", tail_type="place",
                           episode_id="e1", valid_from="2026-01-01",
                           recorded_at="2026-01-05T00:00:00+00:00"))
    store.assert_fact(fact("located_in", "place:berlin", "Berlin", tail_type="place",
                           episode_id="e2", valid_from="2026-02-01",
                           recorded_at="2026-02-01T00:00:00+00:00"))

    assert store.facts[0].is_current is True
    assert [f.tail_name for f in store.as_of("2026-02-15")] == ["Berlin"]


# ---------------------------------------------------------------------------
# kgx.neo4j_io -- pure functions only; nothing here touches a database
# ---------------------------------------------------------------------------


def test_label_for_camel_cases_multiword_types():
    from kgx.neo4j_io import label_for

    # underscore is a word character, so a \W split leaves "Business_segment"
    assert label_for("business_segment") == "BusinessSegment"
    assert label_for("financial_metric") == "FinancialMetric"
    assert label_for("company") == "Company"
    assert label_for("user") == "User"
    assert label_for("") == "Entity"


def test_rel_type_for_upper_snake_cases_relations():
    from kgx.neo4j_io import rel_type_for

    assert rel_type_for("partners_with") == "PARTNERS_WITH"
    assert rel_type_for("acquires") == "ACQUIRES"
    assert rel_type_for("has-stake-in") == "HAS_STAKE_IN"


def test_neo4j_config_prefers_explicit_overrides_then_env(monkeypatch):
    from kgx.neo4j_io import Neo4jConfig

    monkeypatch.setenv("NEO4J_URI", "bolt://from-env:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "env-secret")

    from_env = Neo4jConfig.from_env()
    assert from_env.uri == "bolt://from-env:7687"
    assert from_env.password == "env-secret"
    assert from_env.user == "neo4j"          # falls back to the class default

    explicit = Neo4jConfig.from_env(uri="bolt://explicit:7687")
    assert explicit.uri == "bolt://explicit:7687"
    assert explicit.password == "env-secret"


def test_entity_rows_carry_both_labels_and_flattened_attrs():
    from kgx.neo4j_io import ENTITY_LABEL, _entity_rows

    entity = CanonicalEntity(
        canon_id="company:acme", type="company", canonical="Acme Corp",
        aliases=["Acme"], mentions=["d1:e1", "d2:e1"], docs=["d1", "d2"],
        attrs={"modality": {"label": "asserted", "confidence": 0.9}},
    )
    (row,) = _entity_rows(KnowledgeGraph({"company:acme": entity}, []))

    assert row["labels"] == ["Company", ENTITY_LABEL]
    assert row["props"]["n_mentions"] == 2
    assert row["props"]["n_docs"] == 2
    assert row["props"]["modality"] == "asserted"      # nested dict flattened to its label


def test_edge_rows_filter_on_confidence_and_support():
    from kgx.neo4j_io import _edge_rows

    entities = {
        "company:a": CanonicalEntity("company:a", "company", "A"),
        "company:b": CanonicalEntity("company:b", "company", "B"),
    }
    strong = GraphEdge("company:a", "acquires", "company:b", [
        Evidence("d1", (0, 1), (2, 3), 0.9, "strong"),
        Evidence("d2", (0, 1), (2, 3), 0.8, "again"),
    ])
    weak = GraphEdge("company:a", "partners_with", "company:b", [
        Evidence("d1", (0, 1), (2, 3), 0.2, "weak"),
    ])
    dangling = GraphEdge("company:a", "supplies", "company:missing", [
        Evidence("d1", (0, 1), (2, 3), 0.99, "dangling"),
    ])
    graph = KnowledgeGraph(entities, [strong, weak, dangling])

    rows = _edge_rows(graph, min_confidence=0.5, min_support=2)
    assert [r["rel_type"] for r in rows] == ["ACQUIRES"]
    assert rows[0]["props"]["support"] == 2
    assert rows[0]["props"]["n_docs"] == 2
    assert rows[0]["props"]["docs"] == ["d1", "d2"]
    # dangling endpoints are dropped regardless of confidence
    assert not [r for r in _edge_rows(graph, 0.0, 1) if r["rel_type"] == "SUPPLIES"]


def test_knowledge_graph_round_trips_through_json_with_evidence():
    graph = KnowledgeGraph(
        {"company:a": CanonicalEntity("company:a", "company", "Acme", aliases=["ACME"],
                                      mentions=["d1:e1"], docs=["d1"])},
        [GraphEdge("company:a", "acquires", "company:a",
                   [Evidence("d1", (0, 4), (5, 9), 0.9, "Acme bought Acme")])],
    )

    restored = KnowledgeGraph.from_json(graph.to_json())

    assert restored.entities["company:a"].aliases == ["ACME"]
    assert restored.edges[0].confidence == 0.9
    assert restored.edges[0].evidence[0].snippet == "Acme bought Acme"
    assert restored.edges[0].evidence[0].head_span == (0, 4)


def test_from_json_distinguishes_a_path_from_a_json_string(tmp_path):
    """Path.exists() raises OSError on a long string, so the sniff must come first."""
    onto = Ontology(name="x", entities=(EntityType("a"),),
                    relations=(RelationType("r", ("a",), ("a",)),))

    assert Ontology.from_json(onto.to_json()).name == "x"      # long JSON string

    path = tmp_path / "onto.json"
    onto.to_json(path)
    assert Ontology.from_json(path).name == "x"                # real path


# ---------------------------------------------------------------------------
# kgx.evaluate -- triple scoring. No model, no network.
# ---------------------------------------------------------------------------


def test_score_triples_is_one_to_one():
    """One prediction cannot satisfy two gold triples."""
    from kgx.evaluate import score_triples

    gold = [("A", "r", "B"), ("A", "r", "B")]        # gold happens to repeat
    s = score_triples([("A", "r", "B")], gold)

    assert len(s.matched) == 1
    assert len(s.missed) == 1
    assert s.precision == 1.0
    assert s.recall == 0.5


def test_score_triples_requires_exact_relation_match():
    from kgx.evaluate import score_triples

    s = score_triples([("A", "works_for", "B")], [("A", "employed_by", "B")])
    assert s.matched == []
    assert s.f1 == 0.0


def test_score_triples_normalises_names_but_not_relations():
    from kgx.evaluate import score_triples

    s = score_triples(
        [("Acme Corp.", "acquires", "the Globex")],
        [("Acme", "acquires", "Globex")],
    )
    assert len(s.matched) == 1


def test_aliases_let_a_known_variant_count():
    from kgx.evaluate import MatchPolicy, score_triples

    aliases = {"Northwind Logistics Inc.": ["NWL", "Northwind"]}
    predicted = [("NWL", "acquires", "Cascade")]
    gold = [("Northwind Logistics Inc.", "acquires", "Cascade")]

    without = score_triples(predicted, gold, policy=MatchPolicy(use_aliases=False))
    with_ = score_triples(predicted, gold, policy=MatchPolicy(use_aliases=True), aliases=aliases)

    assert without.matched == []
    assert len(with_.matched) == 1


def test_fuzzy_threshold_is_recorded_as_a_loosening():
    from kgx.evaluate import MatchPolicy, score_triples

    policy = MatchPolicy(use_aliases=False, fuzzy_threshold=0.80)
    s = score_triples([("Halcyon Semiconducter", "produces", "chips")],
                      [("Halcyon Semiconductor", "produces", "chips")], policy=policy)

    assert len(s.matched) == 1
    assert len(s.loosened) == 1          # the audit trail matters more than the match
    assert "fuzzy" in s.loosened[0][2]


def test_undirected_policy_accepts_a_reversed_triple():
    from kgx.evaluate import MatchPolicy, score_triples

    predicted = [("B", "competes_with", "A")]
    gold = [("A", "competes_with", "B")]

    assert score_triples(predicted, gold).matched == []
    assert len(score_triples(predicted, gold, policy=MatchPolicy(directed=False)).matched) == 1


def test_empty_predictions_and_empty_gold_do_not_divide_by_zero():
    from kgx.evaluate import score_triples

    assert score_triples([], [("A", "r", "B")]).precision == 0.0
    assert score_triples([("A", "r", "B")], []).recall == 0.0
    both = score_triples([], [])
    assert (both.precision, both.recall, both.f1) == (0.0, 0.0, 0.0)


def test_graph_triples_filters_on_confidence_and_support():
    from kgx.evaluate import graph_triples

    entities = {
        "company:a": CanonicalEntity("company:a", "company", "A"),
        "company:b": CanonicalEntity("company:b", "company", "B"),
    }
    strong = GraphEdge("company:a", "acquires", "company:b",
                       [Evidence("d1", (0, 1), (2, 3), 0.9, ""),
                        Evidence("d2", (0, 1), (2, 3), 0.8, "")])
    weak = GraphEdge("company:a", "supplies", "company:b",
                     [Evidence("d1", (0, 1), (2, 3), 0.3, "")])
    graph = KnowledgeGraph(entities, [strong, weak])

    assert len(graph_triples(graph)) == 2
    assert graph_triples(graph, min_support=2) == [("A", "acquires", "B")]
    assert graph_triples(graph, min_confidence=0.5) == [("A", "acquires", "B")]
    assert graph_triples(graph, drop_relations=["supplies"]) == [("A", "acquires", "B")]


# ---------------------------------------------------------------------------
# kgx.llm -- response parsing and caching. Never shells out.
# ---------------------------------------------------------------------------


def test_parse_json_response_handles_every_shape_models_emit():
    from kgx.llm import parse_json_response

    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json_response('```\n{"a": 3}\n```') == {"a": 3}
    assert parse_json_response('Sure! Here you go: {"a": 4} — hope that helps') == {"a": 4}
    assert parse_json_response("[1, 2, 3]") == [1, 2, 3]
    # a closing brace inside a string must not end the scan early
    assert parse_json_response('{"a": {"b": "}"}}') == {"a": {"b": "}"}}


def test_parse_json_response_raises_rather_than_guessing():
    from kgx.llm import parse_json_response

    with pytest.raises(ValueError):
        parse_json_response("I'm afraid I can't help with that.")


def test_claude_cli_replays_from_cache_without_the_binary(tmp_path):
    """A cached prompt must not need `claude` on PATH -- that is what makes a
    committed notebook reproducible on a machine with no CLI."""
    import json as _json
    from kgx.llm import ClaudeCLI, LLMUnavailable

    llm = ClaudeCLI(model="haiku", cache_dir=tmp_path, binary="definitely-not-installed")
    assert not llm.available

    path = llm._cache_path("the prompt")
    path.write_text(_json.dumps({"text": '{"ok": true}', "model": "haiku",
                                 "cost_usd": 0.0003, "duration_ms": 800}))

    result = llm.complete("the prompt")
    assert result.cached is True
    assert result.json() == {"ok": True}
    assert llm.n_cached == 1

    with pytest.raises(LLMUnavailable):
        llm.complete("a prompt that was never cached")


def test_cache_key_separates_models_and_system_prompts(tmp_path):
    from kgx.llm import ClaudeCLI

    a = ClaudeCLI(model="haiku", cache_dir=tmp_path, system_prompt="X")
    b = ClaudeCLI(model="sonnet", cache_dir=tmp_path, system_prompt="X")
    c = ClaudeCLI(model="haiku", cache_dir=tmp_path, system_prompt="Y")

    paths = {a._cache_path("p"), b._cache_path("p"), c._cache_path("p")}
    assert len(paths) == 3


def test_llm_extractor_drops_output_the_ontology_forbids():
    """The model may return anything; only ontology-legal output survives."""
    from kgx.llm import LLMExtractor

    onto = Ontology(
        name="tiny",
        entities=(EntityType("person"), EntityType("company")),
        relations=(RelationType("works_for", ("person",), ("company",)),),
    )
    payload = {
        "entities": [
            {"text": "Ada", "type": "person"},
            {"text": "Acme", "type": "company"},
            {"text": "Tuesday", "type": "weekday"},       # type not in the ontology
        ],
        "relations": [
            {"head": "Ada", "type": "works_for", "tail": "Acme"},
            {"head": "Ada", "type": "befriends", "tail": "Acme"},   # relation not in ontology
            {"head": "Ada", "type": "works_for", "tail": "Nobody"},  # dangling endpoint
            {"head": "Ada", "type": "works_for", "tail": "Ada"},     # self loop
        ],
    }
    graph = LLMExtractor.__new__(LLMExtractor)
    graph.context_width = 40
    doc = graph._to_doc_graph(payload, "Ada works for Acme on Tuesday.", "d1", onto, 0.0, {})

    assert sorted(m.type for m in doc.mentions) == ["company", "person"]
    assert [(doc.mention(e.head).text, e.type, doc.mention(e.tail).text) for e in doc.edges] == [
        ("Ada", "works_for", "Acme")
    ]


def test_llm_extractor_recovers_character_offsets():
    from kgx.llm import LLMExtractor

    onto = Ontology(name="t", entities=(EntityType("company"),), relations=())
    text = "Yesterday Acme Corp announced results."
    graph = LLMExtractor.__new__(LLMExtractor)
    graph.context_width = 40
    doc = graph._to_doc_graph({"entities": [{"text": "Acme Corp", "type": "company"}]},
                              text, "d1", onto, 0.0, {})

    (mention,) = doc.mentions
    assert text[mention.start:mention.end] == "Acme Corp"


# ---------------------------------------------------------------------------
# kgx.gazetteer -- dictionary linking against a controlled vocabulary
# ---------------------------------------------------------------------------


def _airports():
    from kgx.gazetteer import Gazetteer

    return Gazetteer({
        "LHR": {"name": "London Heathrow", "kind": "airport", "aliases": ["Heathrow"]},
        "JFK": {"name": "John F. Kennedy International", "kind": "airport",
                "aliases": ["New York JFK", "JFK Airport"]},
        "MR": {"name": "Meridian Air", "kind": "airline", "aliases": ["Meridian"]},
    })


def test_gazetteer_resolves_a_code_no_string_metric_could_reach():
    """LHR vs London Heathrow is Jaro-Winkler 0.48; the dictionary makes it exact."""
    gz = _airports()

    assert gz.lookup("LHR") == ("LHR", "code", 1.0)
    assert gz.lookup("London Heathrow")[:2] == ("LHR", "name")
    assert gz.lookup("Heathrow")[:2] == ("LHR", "alias")


def test_gazetteer_reads_the_parenthetical_code_convention():
    gz = _airports()
    assert gz.lookup("London Heathrow (LHR)") == ("LHR", "code", 1.0)


def test_a_miss_is_a_miss_by_default():
    """Silently snapping an unknown code to the nearest known one destroys the signal."""
    gz = _airports()

    assert gz.lookup("Gatwick") is None
    assert gz.lookup("Heathrw") is None                      # fuzzy off by default
    assert gz.lookup("Heathrw", fuzzy=0.85)[:2] == ("LHR", "fuzzy")


def test_kind_constraint_stops_a_cross_category_link():
    """A city mention must not link to an airline that shares a name."""
    from kgx.extract import Mention

    gz = _airports()
    mentions = [Mention("d:1", "d", "city", "Meridian", 0, 8)]

    unconstrained, _ = gz.resolve(mentions, types=["city"])
    constrained, missed = gz.resolve(mentions, types=["city"], kinds={"city": "airport"})

    assert unconstrained[0].key == "MR"
    assert constrained == [] and len(missed) == 1


def test_resolve_splits_matched_from_unmatched_and_records_the_rule():
    from kgx.extract import Mention

    gz = _airports()
    mentions = [
        Mention("d:1", "d", "airport", "LHR", 0, 3),
        Mention("d:2", "d", "airport", "Heathrow", 0, 8),
        Mention("d:3", "d", "airport", "Gatwick", 0, 7),
        Mention("d:4", "d", "disruption", "LHR", 0, 3),      # excluded by `types`
    ]

    matched, unmatched = gz.resolve(mentions, types=["airport"])

    assert {m.mention_id: m.how for m in matched} == {"d:1": "code", "d:2": "alias"}
    assert [m.mention_id for m in unmatched] == ["d:3", "d:4"]
    assert all(m.key == "LHR" for m in matched)


def test_canonical_map_gives_corpus_stable_ids():
    """Ids come from the vocabulary key, not from a cluster, so they survive a rerun."""
    from kgx.extract import Mention

    gz = _airports()
    matched, _ = gz.resolve(
        [Mention("d:1", "d", "airport", "LHR", 0, 3),
         Mention("d:2", "d", "airport", "London Heathrow", 0, 15)],
        types=["airport"])

    assert gz.canonical_map(matched, prefix="airport:") == {
        "d:1": "airport:LHR", "d:2": "airport:LHR",
    }


def test_coverage_reports_what_the_vocabulary_could_not_account_for():
    from kgx.extract import Mention

    gz = _airports()
    report = gz.coverage(
        [Mention("d:1", "d", "airport", "LHR", 0, 3),
         Mention("d:2", "d", "airport", "Gatwick", 0, 7)],
        types=["airport"])

    assert report["linked"] == 1
    assert report["coverage"] == 0.5
    assert report["by_rule"] == {"code": 1}
    assert report["unlinked_surfaces"] == ["Gatwick"]
