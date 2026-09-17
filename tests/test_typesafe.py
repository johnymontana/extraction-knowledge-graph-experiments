"""Tests for ``kgx.typesafe`` that never touch the network.

The API key path is exercised by the notebook; what is tested here is everything
around it, which is where the bugs actually live: the cache key, the replay
path, question construction, the outcome rule, and the union-find in
:func:`~kgx.typesafe.apply_verdicts`.

Live calls are made impossible rather than mocked -- ``TYPESAFE_API_KEY`` is
cleared and ``cache_dir`` points at a tmp_path -- so a test that accidentally
reaches the network fails with :class:`~kgx.typesafe.TypeSafeUnavailable`
instead of silently billing someone.
"""

from __future__ import annotations

import json

import msgspec
import pytest
import typesafe_sdk as ts

from kgx.extract import DocGraph, Edge, Mention
from kgx.ontology import BUSINESS_NEWS
from kgx.resolve import CanonicalEntity, Resolution, ScoredPair
from kgx.typesafe import (
    ASSERTION_LEVELS,
    LINK_LEVELS,
    CachedTypeSafe,
    EdgeJudgment,
    PairVerdict,
    TypeSafeUnavailable,
    adjudicate_pairs,
    apply_verdicts,
    edge_questions,
    judge_edges,
    pair_questions,
)


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    """No test in this module may make a live call."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


@pytest.fixture
def client(tmp_path):
    return CachedTypeSafe(cache_dir=tmp_path)


def seed(client, state, questions, answers, *, model="jev-latest", duration_ms=120):
    """Write a cache entry so :meth:`CachedTypeSafe.ask` can replay it."""
    path = client._cache_path(state, questions)
    path.write_text(
        msgspec.json.encode(
            {
                "model": model,
                "usage": ts.Usage(input_tokens=100, output_tokens=8),
                "answers": answers,
                "duration_ms": duration_ms,
                "asked_model": model,
                "n_questions": len(questions),
            }
        ).decode()
    )
    return path


def mention(mention_id, type_, text, *, doc_id="d1", start=0, context=""):
    return Mention(
        mention_id=mention_id,
        doc_id=doc_id,
        type=type_,
        text=text,
        start=start,
        end=start + len(text),
        context=context,
    )


# ---------------------------------------------------------------------------
# client: availability and caching
# ---------------------------------------------------------------------------


def test_unavailable_without_key_or_cache(client):
    with pytest.raises(TypeSafeUnavailable, match="TYPESAFE_API_KEY"):
        client.ask("hello", {"q": ts.Noul(instructions="Is this a greeting?")})


def test_check_without_key_names_the_env_var(client):
    assert client.available is False
    with pytest.raises(TypeSafeUnavailable, match="TYPESAFE_API_KEY"):
        client.check()


def test_empty_questions_rejected(client):
    with pytest.raises(ValueError, match="non-empty"):
        client.ask("hello", {})


def test_cache_hit_replays_a_real_response(client):
    state = {"document": "Northwind acquired Cascade."}
    questions = {"q": ts.Noul(instructions="Did Northwind acquire Cascade?")}
    seed(client, state, questions, {"q": ts.NoulAnswer(noul=0.97)})

    assert client.cached(state, questions) is True
    response = client.ask(state, questions)

    # A replay is a real SystemOneResponse, so downstream code cannot tell.
    assert isinstance(response, ts.SystemOneResponse)
    assert response.nouls["q"].noul == pytest.approx(0.97)
    assert response.model == "jev-latest"
    assert client.n_cached == 1 and client.n_calls == 0


def test_cache_hit_replays_a_score_answer(client):
    """Regression: JSON object keys are strings, ScoreAnswer keys them by int.

    Converting a ``json.loads`` dict loses that and every Score replay fails --
    which stays invisible until a notebook is run twice.
    """
    state, questions = "s", {"link": ts.Score(instructions="i", criteria=["lo", "mid", "hi"])}
    seed(client, state, questions, {
        "link": ts.ScoreAnswer(score=1.87, confidence=0.9,
                               legend={0: "lo", 1: "mid", 2: "hi"},
                               probabilities={0: 0.01, 1: 0.11, 2: 0.88}),
    })
    answer = client.ask(state, questions).scores["link"]

    assert answer.score == pytest.approx(1.87)
    assert answer.legend == {0: "lo", 1: "mid", 2: "hi"}          # int keys, not "0"
    assert answer.probabilities[2] == pytest.approx(0.88)


def test_every_answer_type_survives_a_round_trip(client):
    """One of each, in one entry, decoded through the tagged union."""
    state = {"a": 1}
    questions = {
        "n": ts.Noul(instructions="i"),
        "c": ts.Choice(instructions="i", criteria={"x": None, "y": None}),
        "s": ts.Score(instructions="i", criteria=["lo", "hi"]),
    }
    seed(client, state, questions, {
        "n": ts.NoulAnswer(noul=0.42),
        "c": ts.ChoiceAnswer(choice="y", confidence=0.6, probabilities={"x": 0.4, "y": 0.6}),
        "s": ts.ScoreAnswer(score=0.7, confidence=0.7, legend={0: "lo", 1: "hi"},
                            probabilities={0: 0.3, 1: 0.7}),
    })
    r = client.ask(state, questions)
    assert r.nouls["n"].noul == pytest.approx(0.42)
    assert r.choices["c"].choice == "y"
    assert r.scores["s"].legend[1] == "hi"


def test_replay_reports_original_latency_and_tokens(client):
    state, questions = "s", {"q": ts.Noul(instructions="i")}
    seed(client, state, questions, {"q": ts.NoulAnswer(noul=0.5)}, duration_ms=250)
    client.ask(state, questions)

    stats = client.stats()
    # Nothing was billed this run, but the numbers are not zero or NaN.
    assert stats["input_tokens_billed"] == 0
    assert stats["input_tokens_observed"] == 100
    assert stats["ms_spent"] == 0
    assert stats["ms_per_request_observed"] == 250.0


def test_cache_key_depends_on_state_questions_and_model(tmp_path):
    a = CachedTypeSafe(cache_dir=tmp_path)
    q = {"q": ts.Noul(instructions="i")}
    base = a._cache_path("s", q)

    assert a._cache_path("other", q) != base
    assert a._cache_path("s", {"q": ts.Noul(instructions="different")}) != base
    assert a._cache_path("s", {"other_id": ts.Noul(instructions="i")}) != base
    assert CachedTypeSafe(cache_dir=tmp_path, model="jev-1.13")._cache_path("s", q) != base
    # Stable across instances and insensitive to dict ordering.
    assert CachedTypeSafe(cache_dir=tmp_path)._cache_path("s", q) == base
    two = {"a": ts.Noul(instructions="x"), "b": ts.Noul(instructions="y")}
    assert a._cache_path("s", two) == a._cache_path("s", dict(reversed(list(two.items()))))


def test_cache_key_survives_equivalent_question_objects(client):
    """Keying on the wire form, not on identity or repr."""
    as_object = {"q": ts.Choice(instructions="i", criteria={"yes": None, "no": None})}
    as_dict = {"q": {"type": "choice", "instructions": "i",
                     "criteria": {"yes": None, "no": None}}}
    assert client._cache_path("s", as_object) == client._cache_path("s", as_dict)


# ---------------------------------------------------------------------------
# question construction
# ---------------------------------------------------------------------------


def test_edge_questions_name_the_edge_in_the_instructions():
    """Question ids are not sent, so the instructions must carry the meaning."""
    questions = edge_questions("Northwind", "acquires", "Cascade")
    wire = json.loads(msgspec.json.encode(questions))

    assert set(wire) == {"supported", "status"}
    assert wire["supported"]["type"] == "noul"
    assert wire["status"]["type"] == "choice"
    for q in wire.values():
        assert "Northwind" in q["instructions"] and "Cascade" in q["instructions"]
        assert "acquires" in q["instructions"]
    assert set(wire["status"]["criteria"]) == set(ASSERTION_LEVELS)


def test_edge_questions_carry_the_ontology_description():
    rel = BUSINESS_NEWS.relation("supplies")
    questions = edge_questions("Torrent", "supplies", "Northwind",
                               relation_description=rel.description)
    assert rel.description in str(questions["status"].instructions)


def test_pair_questions_are_one_score_and_three_nouls():
    wire = json.loads(msgspec.json.encode(pair_questions("company")))
    assert wire["link_state"]["type"] == "score"
    assert wire["link_state"]["criteria"] == LINK_LEVELS
    assert [k for k, v in wire.items() if v["type"] == "noul"] == [
        "same_name", "same_context", "abbreviation"
    ]
    assert "company" in wire["link_state"]["instructions"]


# ---------------------------------------------------------------------------
# judging edges
# ---------------------------------------------------------------------------


def _doc_graph(n_edges=3):
    mentions, edges = [], []
    for i in range(n_edges + 1):
        mentions.append(mention(f"m{i}", "company", f"Company {i}", context=f"ctx {i}"))
    for i in range(n_edges):
        edges.append(Edge(doc_id="d1", type="competes_with", head=f"m{i}",
                          tail=f"m{i + 1}", confidence=0.9))
    return DocGraph(doc_id="d1", text="Some business news.", mentions=mentions, edges=edges)


def test_judge_edges_batches_questions_per_document(client):
    graph = _doc_graph(n_edges=3)
    state = {"document": graph.text}

    questions = {}
    for i, edge in enumerate(graph.edges):
        for name, q in edge_questions(f"Company {i}", "competes_with",
                                      f"Company {i + 1}").items():
            questions[f"e{i}_{name}"] = q
    answers = {}
    for i in range(3):
        answers[f"e{i}_supported"] = ts.NoulAnswer(noul=0.9)
        answers[f"e{i}_status"] = ts.ChoiceAnswer(
            choice="asserted", confidence=0.8,
            probabilities={k: 0.25 for k in ASSERTION_LEVELS},
        )
    seed(client, state, questions, answers)

    judgments = judge_edges(client, [graph], max_questions=24)

    # Six questions, one request: the fan-out is the point.
    assert client.n_cached == 1
    assert client.n_questions == 6
    assert [j.triple() for j in judgments] == [
        ("Company 0", "competes_with", "Company 1"),
        ("Company 1", "competes_with", "Company 2"),
        ("Company 2", "competes_with", "Company 3"),
    ]
    assert all(j.keep for j in judgments)


def test_judge_edges_chunks_at_max_questions(client, monkeypatch):
    graph = _doc_graph(n_edges=4)
    seen = []

    def fake_ask(state, questions, *, refresh=False):
        seen.append(len(questions))
        answers = {}
        for key in questions:
            tag, _, kind = key.rpartition("_")
            answers[key] = (
                ts.NoulAnswer(noul=0.9) if kind == "supported"
                else ts.ChoiceAnswer(choice="asserted", confidence=0.7, probabilities={})
            )
        return ts.SystemOneResponse(model="jev-latest", usage=ts.Usage(), answers=answers)

    monkeypatch.setattr(client, "ask", fake_ask)
    judgments = judge_edges(client, [graph], max_questions=4)

    assert seen == [4, 4]      # 4 edges x 2 questions, chunked two edges at a time
    assert len(judgments) == 4


def test_judge_edges_skips_derived_mirror_edges(client, monkeypatch):
    graph = _doc_graph(n_edges=1)
    graph.edges.append(Edge(doc_id="d1", type="competes_with", head="m1", tail="m0",
                            confidence=0.9, derived=True))
    asked = []

    def fake_ask(state, questions, *, refresh=False):
        asked.append(sorted(questions))
        return ts.SystemOneResponse(
            model="m", usage=ts.Usage(),
            answers={
                "e0_supported": ts.NoulAnswer(noul=0.9),
                "e0_status": ts.ChoiceAnswer(choice="asserted", confidence=0.7,
                                             probabilities={}),
            },
        )

    monkeypatch.setattr(client, "ask", fake_ask)
    assert len(judge_edges(client, [graph])) == 1
    assert asked == [["e0_status", "e0_supported"]]


@pytest.mark.parametrize(
    "supported,status,keep",
    [
        (0.99, "asserted", True),
        (0.99, "hypothetical", False),   # the modality trap
        (0.99, "negated", False),
        (0.99, "forward_looking", False),
        (0.10, "asserted", False),       # nothing in the document connects them
    ],
)
def test_edge_gate_needs_support_and_assertion(supported, status, keep):
    judgment = EdgeJudgment(doc_id="d1", head="a", relation="acquires", tail="b",
                            supported=supported, status=status)
    assert judgment.keep is keep


# ---------------------------------------------------------------------------
# adjudicating pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level,outcome",
    [
        (0.0, "reject"), (0.25, "reject"), (0.49, "reject"),
        (0.5, "review"), (1.0, "review"), (1.49, "review"),
        (1.5, "merge"), (2.0, "merge"),
    ],
)
def test_outcome_is_the_nearest_level(level, outcome):
    """No fitted threshold: the levels are the outcomes."""
    assert PairVerdict(a="a", b="b", a_text="A", b_text="B", type="company",
                       resolver_score=0.8, level=level).outcome == outcome


def test_adjudicate_pairs_sends_context_not_just_the_surface(client, monkeypatch):
    mentions = {
        "m0": mention("m0", "company", "NWL", context="NWL reported revenue growth"),
        "m1": mention("m1", "company", "Northwind", context="Northwind reported revenue"),
    }
    pairs = [ScoredPair(a="m0", b="m1", type="company", block="acronym", score=0.87)]
    captured = {}

    def fake_ask(state, questions, *, refresh=False):
        captured.update(state)
        return ts.SystemOneResponse(
            model="m", usage=ts.Usage(),
            answers={
                "link_state": ts.ScoreAnswer(score=1.9, confidence=0.88, legend={},
                                             probabilities={0: 0.0, 1: 0.1, 2: 0.9}),
                "same_name": ts.NoulAnswer(noul=0.3),
                "same_context": ts.NoulAnswer(noul=0.95),
                "abbreviation": ts.NoulAnswer(noul=0.99),
            },
        )

    monkeypatch.setattr(client, "ask", fake_ask)
    verdicts = adjudicate_pairs(client, pairs, mentions)

    assert captured["mention_a"]["context"] == "NWL reported revenue growth"
    assert captured["mention_b"]["type"] == "company"
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.outcome == "merge"
    assert v.resolver_score == pytest.approx(0.87)   # what kgx thought, kept alongside
    assert v.abbreviation == pytest.approx(0.99)
    assert v.as_record()["a"] == "NWL"


# ---------------------------------------------------------------------------
# applying verdicts
# ---------------------------------------------------------------------------


def _resolution(mapping):
    entities = {}
    for mid, cid in mapping.items():
        entities.setdefault(
            cid, CanonicalEntity(canon_id=cid, type="company", canonical=cid)
        ).mentions.append(mid)
    return Resolution(entities=entities, mention_to_canon=dict(mapping), pairs=[],
                      threshold=0.9)


def _verdict(a, b, level):
    return PairVerdict(a=a, b=b, a_text=a, b_text=b, type="company",
                       resolver_score=0.8, level=level)


def test_apply_verdicts_merges_only_accepted_outcomes():
    resolution = _resolution({"m0": "c0", "m1": "c1", "m2": "c2"})
    merged = apply_verdicts(
        resolution, [_verdict("m0", "m1", 2.0), _verdict("m1", "m2", 1.0)]
    )
    assert merged["m0"] == merged["m1"]       # merge
    assert merged["m2"] != merged["m0"]       # review, not accepted by default


def test_apply_verdicts_can_accept_the_review_band_too():
    resolution = _resolution({"m0": "c0", "m1": "c1"})
    merged = apply_verdicts(resolution, [_verdict("m0", "m1", 1.0)],
                            accept=("merge", "review"))
    assert merged["m0"] == merged["m1"]


def test_apply_verdicts_is_transitive():
    resolution = _resolution({"m0": "c0", "m1": "c1", "m2": "c2"})
    merged = apply_verdicts(
        resolution, [_verdict("m0", "m1", 2.0), _verdict("m1", "m2", 2.0)]
    )
    assert len({merged["m0"], merged["m1"], merged["m2"]}) == 1


def test_apply_verdicts_is_order_independent():
    pairs = [_verdict("m0", "m1", 2.0), _verdict("m2", "m3", 2.0), _verdict("m1", "m2", 2.0)]
    mapping = {"m0": "c0", "m1": "c1", "m2": "c2", "m3": "c3"}
    first = apply_verdicts(_resolution(mapping), pairs)
    second = apply_verdicts(_resolution(mapping), list(reversed(pairs)))
    assert first == second


def test_apply_verdicts_does_not_mutate_the_resolution():
    resolution = _resolution({"m0": "c0", "m1": "c1"})
    before = dict(resolution.mention_to_canon)
    apply_verdicts(resolution, [_verdict("m0", "m1", 2.0)])
    assert resolution.mention_to_canon == before


def test_apply_verdicts_keeps_untouched_mentions():
    resolution = _resolution({"m0": "c0", "m1": "c0", "m2": "c2"})
    merged = apply_verdicts(resolution, [])
    assert merged == {"m0": "c0", "m1": "c0", "m2": "c2"}
