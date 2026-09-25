"""Tests for ``kgx.alignment`` and ``kgx.data.beer`` that never load the model or the network.

What is tested is the part a notebook run would not catch quickly: the
cookbook's routing rule at its boundaries, the score and confidence
arithmetic, the schema the cookbook's questions compile to (real
``ClassificationSchema`` objects, so a label the library would reject fails
here), how answers map back onto levels and properties, the threshold-free
AUC, the fitted cuts, and the order the benchmark pairs get their ids in.
"""

from __future__ import annotations

import json

import pytest

import kgx.alignment as ka
from kgx.data.beer import pairs_from_tables


class FakeJudge:
    """Answers every task with a scripted ``{label: prob}``; records the texts it saw."""

    def __init__(self, answer):
        self.answer = answer
        self.texts: list[str] = []

    def probabilities(self, texts, schemas):
        texts = list(texts)
        self.texts += texts
        per_text = schemas if isinstance(schemas, (list, tuple)) else [schemas] * len(texts)
        return [{spec.name: self.answer(text, spec.name, list(spec.label_names))
                 for spec in schema.task_specs}
                for text, schema in zip(texts, per_text)]


def _pair(pid="c000", name_b="Fat Tire Amber Ale"):
    return {"id": pid,
            "entity_a": {"name": "Fat Tire Amber Ale", "brewery": "New Belgium Brewing",
                         "style": "American Amber / Red Ale", "abv": "5.20 %"},
            "entity_b": {"name": name_b, "brewery": "New Belgium Brewing Company",
                         "style": "Amber Ale", "abv": "5.2 %"},
            "known_same_as": True, "split": "train"}


# --- the cookbook's decision -------------------------------------------------


@pytest.mark.parametrize("score, outcome", [
    (0.0, "leave unlinked"), (0.49, "leave unlinked"),
    (0.5, "curator queue"), (1.0, "curator queue"), (1.49, "curator queue"),
    (1.5, "assert sameAs"), (2.0, "assert sameAs"),
])
def test_route_is_the_nearest_level(score, outcome):
    assert ka.route(score) == outcome


def test_level_score_is_the_probability_weighted_position():
    assert ka.level_score([0.0, 0.0, 1.0]) == 2.0
    assert ka.level_score([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(1.0)
    assert ka.level_score([0.4, 0.36, 0.24]) == pytest.approx(0.84)


def test_confidence_is_one_on_a_point_mass_and_zero_when_uniform():
    assert ka.confidence([0.0, 1.0, 0.0]) == 1.0
    assert ka.confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert ka.confidence([0.4, 0.36, 0.24]) == pytest.approx(0.1)


def test_cookbook_schema_carries_the_levels_and_questions_verbatim():
    schema = ka.cookbook_schema()
    assert schema.task_order == (ka.LINK_TASK, *ka.NOULS)
    link = schema.task_spec(ka.LINK_TASK)
    assert link.is_exclusive and link.instruction == ka.INSTRUCTION
    assert link.label_names == ka.LEVEL_LABELS
    assert tuple(label.description for label in link.labels) == ka.LEVELS
    for qid, question in ka.NOULS.items():
        spec = schema.task_spec(qid)
        assert spec.label_names == ("yes", "no") and spec.instruction == question
    assert ka.cookbook_schema(nouls=False).task_order == (ka.LINK_TASK,)


def test_cookbook_schema_compiles():
    from gliner2.classification.compiler import compile_schema

    compile_schema(ka.cookbook_schema())


def test_state_text_is_the_pair_as_published_and_nothing_else():
    pair = _pair(name_b="TrÃ ¶ egs Nugget Nectar")
    text = ka.state_text(pair)
    assert json.loads(text) == {"entity_a": pair["entity_a"], "entity_b": pair["entity_b"]}
    assert "TrÃ ¶ egs" in text          # not escaped to \u sequences
    assert "known_same_as" not in text and "c000" not in text


# --- asking Decide -----------------------------------------------------------


def test_align_maps_levels_in_order_and_reports_yes_tasks_as_properties():
    def answer(text, task, labels):
        if task == ka.LINK_TASK:
            return dict(zip(labels, [0.2, 0.3, 0.5]))
        return {"yes": 0.9 if task == "same_brewery" else 0.1, "no": 0.0}

    judge = FakeJudge(answer)
    [j] = ka.align(judge, [_pair()])
    assert judge.texts == [ka.state_text(_pair())]
    assert j.levels == [0.2, 0.3, 0.5] and j.p_top == 0.5
    assert j.score == pytest.approx(1.3) and j.outcome == "curator queue"
    assert j.properties == {"same_name": 0.1, "same_brewery": 0.9, "same_style": 0.1}


def test_align_takes_a_variant_task_and_level_order():
    from gliner2.classification import ClassificationSchema

    schema = ClassificationSchema().single("answer", ["yes", "no"], instruction="Same product?")
    judge = FakeJudge(lambda text, task, labels: {"yes": 0.7, "no": 0.3})
    [j] = ka.align(judge, [_pair()], schema, task="answer", labels=["no", "yes"],
                   render=lambda p: p["entity_b"]["name"])
    assert judge.texts == ["Fat Tire Amber Ale"]
    assert j.probabilities == {"no": 0.3, "yes": 0.7} and j.p_top == 0.7
    assert j.properties == {}


# --- measurement -------------------------------------------------------------


def test_auc_orders_not_places():
    labels = [False, False, True, True]
    assert ka.auc([0.1, 0.2, 0.3, 0.4], labels) == 1.0
    assert ka.auc([0.4, 0.3, 0.2, 0.1], labels) == 0.0
    assert ka.auc([0.20, 0.21, 0.22, 0.23], labels) == 1.0     # compressed, still perfectly ordered
    assert ka.auc([0.5, 0.5, 0.5, 0.5], labels) == 0.5         # ties count half
    assert ka.auc([0.1, 0.3, 0.2, 0.4], labels) == 0.75


def test_auc_needs_both_classes():
    with pytest.raises(ValueError):
        ka.auc([0.1, 0.2], [True, True])


def test_fit_cuts_merges_nothing_a_training_non_match_reached():
    values = [0.10, 0.20, 0.45, 0.25, 0.35, 0.40, 0.50]
    labels = [False, False, False, True, True, True, True]
    low, high = ka.fit_cuts(values, labels, max_missed=0.25)
    assert high == 0.45                                     # the highest non-match
    assert low == 0.35                                      # one of four matches may fall below
    assert [ka.route_by_cuts(v, low, high) for v in values] == [
        "leave unlinked", "leave unlinked", "curator queue",
        "leave unlinked", "curator queue", "curator queue", "assert sameAs"]


def test_fit_cuts_without_overlap_has_no_curator_band():
    # every match above every non-match: the lower cut would land above the
    # upper one, so it is clamped down to it rather than inverting the band
    low, high = ka.fit_cuts([0.1, 0.2, 0.3, 0.4], [False, False, True, True], max_missed=0.5)
    assert low == high == 0.2
    assert [ka.route_by_cuts(v, low, high) for v in (0.1, 0.3)] == ["leave unlinked", "assert sameAs"]


# --- the benchmark pairs -----------------------------------------------------


def test_pairs_get_ids_by_position_across_splits_in_order():
    row = lambda i, name: {"id": str(i), "Beer_Name": name, "Brew_Factory_Name": f"brewery {i}",
                           "Style": "Amber Ale", "ABV": "5 %"}
    table_a = [row(0, "A0"), row(1, "A1")]
    table_b = [row(0, "B0"), row(1, "B1")]
    labelled = [("train", [{"ltable_id": "1", "rtable_id": "0", "label": "0"}]),
                ("valid", [{"ltable_id": "0", "rtable_id": "1", "label": "1"},
                           {"ltable_id": "1", "rtable_id": "1", "label": "0"}])]
    pairs = pairs_from_tables(table_a, table_b, labelled)
    assert [p["id"] for p in pairs] == ["c000", "c001", "c002"]
    assert [p["split"] for p in pairs] == ["train", "valid", "valid"]
    assert [p["known_same_as"] for p in pairs] == [False, True, False]
    assert pairs[0]["entity_a"] == {"name": "A1", "brewery": "brewery 1", "style": "Amber Ale", "abv": "5 %"}
    assert pairs[1]["entity_b"]["name"] == "B1"
    pairs[0]["entity_a"]["name"] = "changed"                 # entities are copies, not shared rows
    assert pairs[2]["entity_a"]["name"] == "A1"


# --- Jev, replayed from a seeded cache: never the network ---------------------


@pytest.fixture
def cached_client(tmp_path, monkeypatch):
    from kgx.typesafe import CachedTypeSafe

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    return CachedTypeSafe(model="jev-1.12", cache_dir=tmp_path)


def _seed(client, pair, *, legend_order=(0, 1, 2), probs=(0.02, 0.22, 0.76), score=1.73):
    import msgspec
    import typesafe_sdk as ts

    questions = ka.jev_questions()
    answers = {
        ka.LINK_TASK: ts.ScoreAnswer(
            score=score, confidence=0.6,
            legend={k: ka.LEVELS[level] for k, level in enumerate(legend_order)},
            probabilities=dict(enumerate(probs))),
        "same_name": ts.NoulAnswer(noul=0.97),
        "same_brewery": ts.NoulAnswer(noul=0.99),
        "same_style": ts.NoulAnswer(noul=0.81),
    }
    client._cache_path(ka.state(pair), questions).write_text(msgspec.json.encode({
        "model": "jev-1.12", "usage": ts.Usage(input_tokens=300, output_tokens=12),
        "answers": answers, "duration_ms": 120}).decode())


def test_jev_questions_are_the_cookbooks_verbatim():
    questions = ka.jev_questions()
    assert list(questions) == [ka.LINK_TASK, *ka.NOULS]
    link = questions[ka.LINK_TASK]
    assert link.instructions == ka.INSTRUCTION and list(link.criteria) == list(ka.LEVELS)
    assert {qid: questions[qid].instructions for qid in ka.NOULS} == ka.NOULS


def test_ask_jev_replays_and_keeps_the_reported_score(cached_client):
    pair = _pair()
    _seed(cached_client, pair)
    [j] = ka.ask_jev(cached_client, [pair])
    assert j.levels == [0.02, 0.22, 0.76]
    assert j.score == 1.73                       # reported, not recomputed (1.74)
    assert ka.level_score(j.levels) == pytest.approx(1.74)
    assert j.confidence == 0.6 and j.outcome == "assert sameAs"
    assert j.properties == {"same_name": 0.97, "same_brewery": 0.99, "same_style": 0.81}
    assert (j.input_tokens, j.output_tokens) == (300, 12)
    assert cached_client.n_calls == 0 and cached_client.n_cached == 1


def test_ask_jev_maps_levels_by_legend_text_not_position(cached_client):
    pair = _pair()
    # legend lists the levels same, related, different: key 0 is `same product`
    _seed(cached_client, pair, legend_order=(2, 1, 0), probs=(0.76, 0.22, 0.02))
    [j] = ka.ask_jev(cached_client, [pair])
    assert j.probabilities == {"different product": 0.02, "related product": 0.22, "same product": 0.76}


def test_ask_jev_refuses_rather_than_calls_without_a_key(cached_client):
    from kgx.typesafe import TypeSafeUnavailable

    with pytest.raises(TypeSafeUnavailable):
        ka.ask_jev(cached_client, [_pair()])
