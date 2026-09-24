"""Tests for ``kgx.decide`` that never load the model.

Every stage talks to GLiNER2.5-Decide only through ``judge.probabilities``, so a
scripted fake stands in for it. What is tested is everything around the model,
which is where the bugs live: the schemas each stage builds (labels, task
names, instructions), the windows it sends, how answers map back to ontology
names, the gate rule, aggregation onto canonical edges, and the clone
bookkeeping that lets a second type reach blocking.

The schemas are real ``gliner2.classification.ClassificationSchema`` objects --
constructing them needs no weights -- so a label the library would reject
fails here, not twenty minutes into a notebook.
"""

from __future__ import annotations

import pytest

from kgx.decide import (
    FACT,
    MODALITY_LABELS,
    NO_RELATION_LABEL,
    EdgeCheck,
    TypeOpinion,
    aggregate_checks,
    check_edges,
    clean,
    edge_window,
    fold_clones,
    judge_sentences,
    modality_schema,
    relation_label,
    relation_schema,
    same_referent,
    second_opinion,
    select_relations,
    sentence_spans,
    type_label,
    type_mentions,
    type_schema,
)
from kgx.extract import DocGraph, Edge, Mention
from kgx.ontology import BUSINESS_NEWS
from kgx.typesafe import NO_RELATION


class FakeJudge:
    """Records what it was asked; answers with a scripted function.

    ``answer(text, task, labels, instruction) -> {label: prob}``. The default
    puts all the mass on the first label.
    """

    def __init__(self, answer=None):
        self.answer = answer or (lambda text, task, labels, instruction: {labels[0]: 1.0})
        self.calls: list[tuple[list[str], list]] = []

    def probabilities(self, texts, schemas):
        texts = list(texts)
        per_text = schemas if isinstance(schemas, (list, tuple)) else [schemas] * len(texts)
        self.calls.append((texts, list(per_text)))
        out = []
        for text, schema in zip(texts, per_text):
            row = {}
            for spec in schema.task_specs:
                labels = list(spec.label_names)
                given = self.answer(text, spec.name, labels, spec.instruction)
                row[spec.name] = {l: float(given.get(l, 0.0)) for l in labels}
            out.append(row)
        return out


TEXT = (
    "Northwind Logistics agreed to acquire Cascade Freight. "
    "Priya Raman is chief executive of Northwind Logistics. "
    "The company may face supply disruptions."
)


def _m(doc, i, text, type_, context=""):
    start = TEXT.index(text)
    return Mention(f"{doc}:e{i}", doc, type_, text, start, start + len(text), 0.9, context or text)


@pytest.fixture
def graph():
    mentions = [
        _m("d1", 0, "Northwind Logistics", "company"),
        _m("d1", 1, "Cascade Freight", "company"),
        _m("d1", 2, "Priya Raman", "person"),
        _m("d1", 3, "supply disruptions", "risk_factor"),
    ]
    edges = [
        Edge("d1", "acquires", "d1:e0", "d1:e1", 0.9),
        Edge("d1", "officer_of", "d1:e2", "d1:e0", 0.8),
        Edge("d1", "partners_with", "d1:e1", "d1:e0", 0.6, derived=True),
    ]
    return DocGraph("d1", TEXT, mentions, edges)


# -- labels and schemas ------------------------------------------------------


def test_clean_strips_what_the_classifier_would_reject():
    assert clean("Halcyon (NYSE: HLCN)") == "Halcyon NYSE: HLCN"
    assert clean("[L] odd [DESCRIPTION] input") == "odd input"
    assert clean("()") == "?"                      # never an empty label


def test_labels_are_plain_words():
    assert type_label("business_segment") == "business segment"
    assert relation_label("officer_of") == "officer of"


def test_modality_schema_has_no_instruction_and_the_five_labels():
    spec = modality_schema().task_spec("modality")
    assert spec.label_names == MODALITY_LABELS
    assert spec.instruction is None
    assert spec.is_exclusive


def test_relation_schema_offers_no_relation_and_maps_back_to_names():
    rels = [BUSINESS_NEWS.relation("acquires"), BUSINESS_NEWS.relation("partners_with")]
    schema, names = relation_schema("Northwind (NWL)", "Cascade", rels)
    spec = schema.task_spec("relation")
    assert spec.label_names == ("acquires", "partners with", NO_RELATION_LABEL)
    assert names == {"acquires": "acquires", "partners with": "partners_with",
                     NO_RELATION_LABEL: NO_RELATION}
    assert "Northwind NWL" in spec.instruction and "Cascade" in spec.instruction


def test_type_schema_covers_every_entity_type():
    schema, names = type_schema(BUSINESS_NEWS)
    assert sorted(names.values()) == sorted(e.name for e in BUSINESS_NEWS.entities)
    assert "financial metric" in schema.task_spec("type").label_names


def test_type_schema_takes_everyday_words_and_falls_back_for_the_rest():
    schema, names = type_schema(BUSINESS_NEWS, labels={"geography": "place (city)"})
    shown = schema.task_spec("type").label_names
    assert names["place city"] == "geography"       # scrubbed, and mapped back
    assert "financial metric" in shown              # not overridden -> type_label


def test_type_schema_rejects_labels_that_collide():
    with pytest.raises(ValueError):
        type_schema(BUSINESS_NEWS, labels={"geography": "company"})


# -- windows -----------------------------------------------------------------


def test_sentence_spans_cover_the_text():
    spans = sentence_spans(TEXT)
    assert len(spans) == 3
    assert TEXT[spans[1][0]:spans[1][1]].startswith("Priya Raman")


def test_edge_window_is_the_sentences_the_edge_spans(graph):
    acquires, officer = graph.edges[0], graph.edges[1]
    assert edge_window(graph, acquires) == "Northwind Logistics agreed to acquire Cascade Freight."
    # Priya Raman (sentence 2) -> Northwind Logistics (first seen in sentence 1)
    assert edge_window(graph, officer).startswith("Northwind Logistics agreed")
    assert edge_window(graph, officer).endswith("Northwind Logistics.")
    assert edge_window(graph, acquires, pad=1).endswith("Northwind Logistics.")


# -- edge checks -------------------------------------------------------------


def _script(modality, relation_for):
    def answer(text, task, labels, instruction):
        if task == "modality":
            return {modality: 1.0}
        pick = relation_for(instruction)
        return {pick: 0.8, labels[-1]: 0.2} if pick in labels else {labels[-1]: 1.0}
    return answer


def test_check_edges_asks_two_questions_per_edge_in_one_batch(graph):
    judge = FakeJudge(_script("fact", lambda ins: "acquires"))
    checks = check_edges(judge, [graph], BUSINESS_NEWS)
    assert len(judge.calls) == 1                     # one batched call
    texts, schemas = judge.calls[0]
    assert len(texts) == 2 * len(checks) == 4        # derived mirror skipped
    assert texts[0] == texts[1]                      # both tasks see the same window
    assert schemas[0].task_order == ("modality",)
    assert schemas[1].task_order == ("relation",)
    assert "Northwind Logistics" in schemas[1].task_spec("relation").instruction


def test_check_keys_are_unique_per_edge_even_when_surfaces_repeat(graph):
    # a second "Northwind Logistics" mention, and the same surface triple from it
    again = _m("d1", 4, "Northwind Logistics", "company")
    again = type(again)(again.mention_id, "d1", "company", again.text,
                        TEXT.rindex("Northwind Logistics"), TEXT.rindex("Northwind Logistics") + 19)
    g = DocGraph("d1", TEXT, graph.mentions + [again],
                 [Edge("d1", "officer_of", "d1:e2", "d1:e0"), Edge("d1", "officer_of", "d1:e2", "d1:e4")])
    a, b = check_edges(FakeJudge(), [g], BUSINESS_NEWS)
    assert (a.head, a.relation, a.tail) == (b.head, b.relation, b.tail)
    assert a.key() != b.key()
    assert a.window != b.window


def test_check_edges_can_include_derived_edges(graph):
    checks = check_edges(FakeJudge(), [graph], BUSINESS_NEWS, include_derived=True)
    assert [c.relation for c in checks] == ["acquires", "officer_of", "partners_with"]


def test_gate_needs_fact_and_agreement(graph):
    judge = FakeJudge(_script("fact", lambda ins: "acquires"))
    acquires, officer = check_edges(judge, [graph], BUSINESS_NEWS)
    assert acquires.agrees and acquires.asserted and acquires.keep
    assert acquires.p_relation == pytest.approx(0.8)
    assert officer.relation_choice == NO_RELATION    # "acquires" is not legal person->company
    assert not officer.keep

    hedged = check_edges(FakeJudge(_script("possibility", lambda ins: "acquires")),
                         [graph], BUSINESS_NEWS)[0]
    assert hedged.agrees and not hedged.asserted and not hedged.keep


def test_relation_probabilities_are_keyed_by_ontology_name(graph):
    judge = FakeJudge(_script("fact", lambda ins: "officer of"))
    officer = check_edges(judge, [graph], BUSINESS_NEWS)[1]
    assert officer.relation_choice == "officer_of"
    assert set(officer.relation_probabilities) >= {"officer_of", NO_RELATION}


def _check(doc, head_id, tail_id, modality, agrees, p_fact):
    return EdgeCheck(doc_id=doc, head="h", relation="acquires", tail="t",
                     head_id=head_id, tail_id=tail_id, modality=modality,
                     modality_probabilities={FACT: p_fact},
                     relation_choice="acquires" if agrees else NO_RELATION)


def test_aggregate_prefers_a_kept_check_over_a_confident_hedge():
    canon = {"d1:a": "A", "d1:b": "B", "d2:a": "A", "d2:b": "B"}
    hedged = _check("d1", "d1:a", "d1:b", "possibility", True, 0.45)
    stated = _check("d2", "d2:a", "d2:b", FACT, True, 0.40)
    best = aggregate_checks([hedged, stated], canon)
    assert best[("A", "acquires", "B")] is stated
    assert aggregate_checks([stated, hedged], canon)[("A", "acquires", "B")] is stated


def test_aggregate_skips_unresolved_endpoints():
    assert aggregate_checks([_check("d1", "x", "y", FACT, True, 0.9)], {}) == {}


def test_judge_sentences_reports_the_distribution():
    judge = FakeJudge(lambda text, task, labels, ins: {"denial": 0.7, FACT: 0.3})
    row = judge_sentences(judge, ["We do not expect to acquire."])[0]
    assert row["modality"] == "denial" and row["p_fact"] == pytest.approx(0.3)


# -- typing --------------------------------------------------------------------


def test_type_mentions_sends_bare_surfaces_once_each(graph):
    mentions = graph.mentions + [_m("d2", 0, "Northwind Logistics", "company")]
    judge = FakeJudge(lambda text, task, labels, ins:
                      {"person": 1.0} if text == "Priya Raman" else {"company": 0.6, "security": 0.4})
    opinions = type_mentions(judge, mentions, BUSINESS_NEWS)
    texts, _ = judge.calls[0]
    assert sorted(texts) == sorted({m.text for m in mentions})     # deduplicated, no context
    assert len(opinions) == len(mentions)
    by_text = {o.text: o for o in opinions}
    assert by_text["Priya Raman"].type == "person" and not by_text["Priya Raman"].changed
    assert by_text["supply disruptions"].type == "company" and by_text["supply disruptions"].changed
    assert by_text["Cascade Freight"].alternatives(0.25) == ["security"]


def _opinion(m, type_, probs):
    return TypeOpinion(m.mention_id, m.doc_id, m.text, m.type, type_, probs[type_], probs)


def test_second_opinion_keeps_the_extractor_type_and_adds_a_clone(graph):
    hlcn = graph.mentions[0]
    ops = [_opinion(hlcn, "security", {"security": 0.6, "company": 0.3, "product": 0.1})]
    out, clones = second_opinion(graph.mentions, ops)
    assert out[0] is hlcn                                   # original untouched
    assert clones == {f"{hlcn.mention_id}~security": hlcn.mention_id}
    clone = next(m for m in out if m.mention_id in clones)
    assert clone.type == "security" and clone.text == hlcn.text

    _, more = second_opinion(graph.mentions, ops, min_prob=0.1)
    assert set(more) == {f"{hlcn.mention_id}~security", f"{hlcn.mention_id}~product"}


def test_second_opinion_adds_nothing_when_the_models_agree(graph):
    m = graph.mentions[2]
    out, clones = second_opinion(graph.mentions, [_opinion(m, "person", {"person": 1.0})])
    assert clones == {} and out == graph.mentions


def test_second_opinion_clones_fold_back_onto_originals(graph):
    hlcn = graph.mentions[0]
    _, clones = second_opinion(graph.mentions, [_opinion(hlcn, "security", {"security": 1.0})])
    clone_id = next(iter(clones))
    # the clone landed in cluster Z with some other mention; the original was in X
    mapping = {hlcn.mention_id: "X", clone_id: "Z", "other": "Z"}
    folded = fold_clones(mapping, clones)
    assert clone_id not in folded
    assert folded[hlcn.mention_id] == folded["other"]


# -- negative-result helpers ---------------------------------------------------


def test_select_relations_asks_over_the_pair_window(graph):
    judge = FakeJudge(lambda text, task, labels, ins: {labels[0]: 0.9, labels[-1]: 0.1})
    picks = select_relations(judge, [graph], BUSINESS_NEWS, scope="document")
    assert picks and all(p.relation != NO_RELATION for p in picks)
    texts, _ = judge.calls[0]
    assert all(t in TEXT for t in texts)


def test_same_referent_asks_over_both_contexts():
    a = Mention("d1:e0", "d1", "product", "Aurora 14 (2025)", 0, 9, context="The Aurora 14 is light.")
    b = Mention("d2:e0", "d2", "product", "Aurora 14 Pro", 0, 13, context="The Pro adds a GPU.")
    judge = FakeJudge(lambda text, task, labels, ins: {"no": 0.9, "yes": 0.1})
    assert same_referent(judge, [(a, b)]) == [pytest.approx(0.1)]
    texts, schemas = judge.calls[0]
    assert texts[0] == "The Aurora 14 is light.\n\nThe Pro adds a GPU."
    assert "Aurora 14 2025" in schemas[0].task_spec("answer").instruction
