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
import os

import msgspec
import pytest
import typesafe_sdk as ts

from kgx.extract import DocGraph, Edge, Mention
from kgx.ontology import BUSINESS_NEWS
from kgx.resolve import CanonicalEntity, Resolution, ScoredPair
from kgx.typesafe import (
    ASSERTION_LEVELS,
    LINK_LEVELS,
    NO_RELATION,
    CachedTypeSafe,
    EdgeJudgment,
    JevAlternatives,
    PairVerdict,
    RelationPick,
    TypeSafeUnavailable,
    TypeVerdict,
    adjudicate_pairs,
    alternatives_questions,
    apply_types,
    apply_verdicts,
    candidate_pairs,
    edge_questions,
    fold_clones,
    judge_edges,
    judge_sentences,
    legal_relations,
    pair_questions,
    pair_state,
    picks_to_graphs,
    relation_questions,
    retype_mentions,
    select_relations,
    sentence_questions,
    soft_typed,
    type_questions,
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


def _shown_cache_dir(client):
    shown = repr(client)
    assert "cache_dir=" in shown and "key=unset" in shown
    return shown.split("cache_dir=")[1].split(",")[0].strip("'")


def test_repr_shows_a_relative_cache_path(client, tmp_path):
    """A notebook prints this; it must not bake an absolute home path into an output."""
    import os
    from pathlib import Path

    inner = _shown_cache_dir(client)
    assert not os.path.isabs(inner)
    assert Path(inner).resolve() == tmp_path.resolve()


def test_repr_from_a_sibling_directory_is_the_notebook_case(tmp_path, monkeypatch):
    """cwd = notebooks/, cache = ../output/typesafe_cache: no absolute prefix at all."""
    cache = tmp_path / "output" / "typesafe_cache"
    here = tmp_path / "notebooks"
    here.mkdir()
    monkeypatch.chdir(here)

    inner = _shown_cache_dir(CachedTypeSafe(cache_dir=cache))
    assert inner == os.path.join("..", "output", "typesafe_cache")
    assert str(tmp_path) not in inner


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


def test_sentence_questions_name_the_denial_trap():
    """The factual Noul must say which reading of a denial to take."""
    wire = json.loads(msgspec.json.encode(sentence_questions()))
    assert wire["status"]["type"] == "choice"
    assert set(wire["status"]["criteria"]) == set(ASSERTION_LEVELS)
    assert wire["factual"]["type"] == "noul"
    assert "NOT" in wire["factual"]["instructions"]          # the denial warning
    assert "denies" in wire["factual"]["criteria"]["false"]


def test_judge_sentences_accepts_an_override(client, monkeypatch):
    """The notebook replays the first, badly worded version through this."""
    sent = []

    def fake_ask(state, questions, *, refresh=False):
        sent.append(questions)
        return ts.SystemOneResponse(
            model="m", usage=ts.Usage(),
            answers={"status": ts.ChoiceAnswer(choice="negated", confidence=0.9,
                                               probabilities={}),
                     "factual": ts.NoulAnswer(noul=0.7)},
        )

    monkeypatch.setattr(client, "ask", fake_ask)
    naive = dict(sentence_questions())
    naive["factual"] = ts.Noul(instructions="first attempt")
    rows = judge_sentences(client, ["a sentence"], questions=naive)

    assert sent[0]["factual"].instructions == "first attempt"
    assert rows[0]["status"] == "negated" and rows[0]["factual"] == pytest.approx(0.7)


def test_pair_state_sends_context_by_name():
    a = mention("m0", "company", "NWL", context="NWL reported revenue growth")
    b = mention("m1", "security", "Northwind", context="shares of Northwind fell")
    state = pair_state(a, b)
    assert state == {
        "mention_a": {"text": "NWL", "type": "company", "context": "NWL reported revenue growth"},
        "mention_b": {"text": "Northwind", "type": "security", "context": "shares of Northwind fell"},
    }


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


# ---------------------------------------------------------------------------
# guided pair questions
# ---------------------------------------------------------------------------


def test_pair_questions_without_a_guideline_are_unchanged():
    """Cache compatibility: no description means the exact old wire form."""
    wire = json.loads(msgspec.json.encode(pair_questions("company")))
    assert "In this domain" not in wire["link_state"]["instructions"]
    assert wire == json.loads(msgspec.json.encode(pair_questions("company", "")))


def test_pair_questions_carry_the_ontology_guideline():
    desc = BUSINESS_NEWS.entity("security").description
    wire = json.loads(msgspec.json.encode(pair_questions("security", desc)))
    assert f"In this domain, a security is: {desc}" in wire["link_state"]["instructions"]
    # The diagnostics are unaffected.
    assert wire["same_name"] == json.loads(msgspec.json.encode(pair_questions("security")))["same_name"]


def test_adjudicate_pairs_passes_both_descriptions_for_a_cross_type_pair(client, monkeypatch):
    mentions = {"m0": mention("m0", "security", "HLCN", context="shares of HLCN"),
                "m1": mention("m1", "company", "Halcyon", context="Halcyon reported")}
    pairs = [ScoredPair(a="m0", b="m1", type="security or company", block="never", score=0.0)]
    seen = {}

    def fake_ask(state, questions, *, refresh=False):
        seen["instr"] = questions["link_state"].instructions
        return ts.SystemOneResponse(
            model="m", usage=ts.Usage(),
            answers={"link_state": ts.ScoreAnswer(score=1.9, confidence=0.9, legend={},
                                                  probabilities={}),
                     "same_name": ts.NoulAnswer(noul=0.5),
                     "same_context": ts.NoulAnswer(noul=0.5),
                     "abbreviation": ts.NoulAnswer(noul=0.9)})

    monkeypatch.setattr(client, "ask", fake_ask)
    adjudicate_pairs(client, pairs, mentions, ontology=BUSINESS_NEWS)
    assert BUSINESS_NEWS.entity("security").description in seen["instr"]
    assert BUSINESS_NEWS.entity("company").description in seen["instr"]


# ---------------------------------------------------------------------------
# re-typing
# ---------------------------------------------------------------------------


def test_type_questions_use_the_ontology_descriptions_as_criteria():
    wire = json.loads(msgspec.json.encode(type_questions("HLCN", BUSINESS_NEWS)))["type"]
    assert wire["type"] == "choice"
    assert wire["criteria"] == {e.name: e.description for e in BUSINESS_NEWS.entities}
    assert "HLCN" in wire["instructions"] and "stand for" in wire["instructions"]


def _typed_graph():
    ms = [mention("m0", "security", "HLCN", context="a"),
          mention("m1", "company", "Halcyon", context="b"),
          mention("m2", "security", "HLCN", context="c")]      # same surface as m0
    return DocGraph(doc_id="d1", text="HLCN and Halcyon and HLCN.", mentions=ms)


def test_retype_mentions_dedupes_surfaces_and_fans_the_answer_out(client, monkeypatch):
    seen = []

    def fake_ask(state, questions, *, refresh=False):
        seen.append(sorted(questions))
        return ts.SystemOneResponse(
            model="m", usage=ts.Usage(),
            answers={"s0": ts.ChoiceAnswer(choice="security", confidence=0.6,
                                           probabilities={"security": 0.6, "company": 0.4}),
                     "s1": ts.ChoiceAnswer(choice="company", confidence=1.0,
                                           probabilities={"company": 1.0})})

    monkeypatch.setattr(client, "ask", fake_ask)
    verdicts = retype_mentions(client, [_typed_graph()], BUSINESS_NEWS)

    assert seen == [["s0", "s1"]]                       # two surfaces, one request
    assert [v.mention_id for v in verdicts] == ["m0", "m2", "m1"]
    assert verdicts[0].alternatives(0.25) == ["company"]
    assert verdicts[1].probabilities == verdicts[0].probabilities   # copied to m2
    assert not any(v.changed for v in verdicts)


def test_retype_mentions_chunks_at_max_questions(client, monkeypatch):
    ms = [mention(f"m{i}", "company", f"Co{i}") for i in range(5)]
    graph = DocGraph(doc_id="d1", text="x", mentions=ms)
    sizes = []

    def fake_ask(state, questions, *, refresh=False):
        sizes.append(len(questions))
        return ts.SystemOneResponse(
            model="m", usage=ts.Usage(),
            answers={k: ts.ChoiceAnswer(choice="company", confidence=1.0, probabilities={})
                     for k in questions})

    monkeypatch.setattr(client, "ask", fake_ask)
    assert len(retype_mentions(client, [graph], BUSINESS_NEWS, max_questions=2)) == 5
    assert sizes == [2, 2, 1]


def _type_verdict(mid, old, new, probs):
    return TypeVerdict(mention_id=mid, doc_id="d1", text=mid, extractor_type=old, type=new,
                       confidence=max(probs.values()), probabilities=probs)


def test_apply_types_swaps_the_argmax_in_and_keeps_ids():
    ms = [mention("m0", "security", "HLCN"), mention("m1", "company", "Halcyon")]
    out = apply_types(ms, [_type_verdict("m0", "security", "company", {"company": 0.9})])
    assert [(m.mention_id, m.type) for m in out] == [("m0", "company"), ("m1", "company")]
    assert out[0].text == "HLCN"


def test_soft_typed_clones_only_alternatives_with_mass():
    ms = [mention("m0", "security", "HLCN"), mention("m1", "company", "Halcyon")]
    verdicts = [_type_verdict("m0", "security", "security", {"security": 0.6, "company": 0.4}),
                _type_verdict("m1", "company", "company", {"company": 0.99, "person": 0.01})]
    out, clones = soft_typed(ms, verdicts, min_prob=0.25)

    assert [(m.mention_id, m.type) for m in out] == [
        ("m0", "security"), ("m0~company", "company"), ("m1", "company")]
    assert clones == {"m0~company": "m0"}


def test_fold_clones_unites_the_original_with_wherever_its_clone_landed():
    # m0 (security) sat alone; its company clone was clustered with m1.
    mapping = {"m0": "c-sec", "m0~company": "c-halcyon", "m1": "c-halcyon"}
    folded = fold_clones(mapping, {"m0~company": "m0"})
    assert set(folded) == {"m0", "m1"}                   # clones are gone
    assert folded["m0"] == folded["m1"]


def test_fold_clones_is_a_no_op_without_clones():
    mapping = {"m0": "c0", "m1": "c1"}
    assert fold_clones(mapping, {}) == mapping


# ---------------------------------------------------------------------------
# relation selection
# ---------------------------------------------------------------------------


def test_legal_relations_follow_the_ontology():
    legal = legal_relations(BUSINESS_NEWS)
    names = {r.name for r in legal[("company", "company")]}
    assert {"acquires", "subsidiary_of", "supplies", "partners_with", "competes_with"} <= names
    assert ("person", "company") in legal and ("geography", "company") not in legal


def _pair_graph():
    text = "Northwind acquires Cascade.\n\nPriya Raman runs Northwind. Diesel rose."
    ms = [mention("m0", "company", "Northwind", start=0),
          mention("m1", "company", "Cascade", start=19),
          mention("m2", "person", "Priya Raman", start=29),
          mention("m3", "company", "Northwind", start=46),
          mention("m4", "commodity", "Diesel", start=57)]
    return DocGraph(doc_id="d1", text=text, mentions=ms)


def test_candidate_pairs_respect_scope_legality_and_dedupe():
    g = _pair_graph()
    para = {(a.text, b.text) for a, b in candidate_pairs(g, BUSINESS_NEWS, scope="paragraph")}
    # Same paragraph, legal both ways.
    assert ("Northwind", "Cascade") in para and ("Cascade", "Northwind") in para
    # Person -> company is legal (officer_of); company -> person is not.
    assert ("Priya Raman", "Northwind") in para and ("Northwind", "Priya Raman") not in para
    # Cross-paragraph pairs are excluded at paragraph scope ...
    assert ("Priya Raman", "Cascade") not in para
    # ... and included at document scope.
    doc = {(a.text, b.text) for a, b in candidate_pairs(g, BUSINESS_NEWS, scope="document")}
    assert ("Priya Raman", "Cascade") in doc
    # Diesel -> Northwind is legal (supplies); no pair is repeated.
    assert ("Diesel", "Northwind") in doc
    allpairs = [(a.text, b.text) for a, b in candidate_pairs(g, BUSINESS_NEWS, scope="document")]
    assert len(allpairs) == len(set(allpairs))
    # A surface paired with itself is never a candidate.
    assert ("Northwind", "Northwind") not in doc


def test_relation_questions_always_offer_none():
    legal = legal_relations(BUSINESS_NEWS)
    wire = json.loads(msgspec.json.encode(
        relation_questions("Northwind", "Cascade", legal[("company", "company")])))["relation"]
    assert wire["type"] == "choice"
    assert NO_RELATION in wire["criteria"]
    assert wire["criteria"]["acquires"] == BUSINESS_NEWS.relation("acquires").description
    assert "Northwind" in wire["instructions"] and "Cascade" in wire["instructions"]


def test_select_relations_batches_per_document_and_builds_edges(client, monkeypatch):
    g = _pair_graph()
    seen = []

    def fake_ask(state, questions, *, refresh=False):
        seen.append(len(questions))
        answers = {}
        for key, q in questions.items():
            labels = list(q.criteria)
            pick = "acquires" if "acquires" in labels and "Northwind" in q.instructions.split("to")[0] \
                else NO_RELATION
            probs = {l: 0.0 for l in labels}; probs[pick] = 0.9
            answers[key] = ts.ChoiceAnswer(choice=pick, confidence=0.9, probabilities=probs)
        return ts.SystemOneResponse(model="m", usage=ts.Usage(), answers=answers)

    monkeypatch.setattr(client, "ask", fake_ask)
    picks = select_relations(client, [g], BUSINESS_NEWS, scope="document", max_questions=3)

    n_pairs = len(candidate_pairs(g, BUSINESS_NEWS, scope="document"))
    assert sum(seen) == n_pairs and all(s <= 3 for s in seen)
    assert len(picks) == n_pairs
    assert any(p.picked for p in picks) and any(not p.picked for p in picks)

    graphs = picks_to_graphs([g], picks, min_prob=0.5)
    edges = graphs[0].edges
    assert edges and all(e.type == "acquires" for e in edges)
    assert all(e.confidence == pytest.approx(0.9) for e in edges)
    assert graphs[0].validate(BUSINESS_NEWS) == []        # legal by construction
    assert graphs[0].mentions == g.mentions               # mentions untouched


def test_alternatives_questions_name_the_relation():
    wire = json.loads(msgspec.json.encode(alternatives_questions("npm", "pnpm", relation="prefers")))
    assert wire["alternatives"]["type"] == "noul"
    assert "prefers" in wire["alternatives"]["instructions"]
    assert set(wire["kind"]["criteria"]) == {"alternatives", "complementary", "same", "unrelated"}


def test_jev_alternatives_is_a_thresholded_symmetric_callable_with_a_log(client, monkeypatch):
    asked = []

    def fake_ask(state, questions, *, refresh=False):
        asked.append((state["a"], state["b"]))
        p = 0.9 if {state["a"], state["b"]} == {"npm", "pnpm"} else 0.1
        return ts.SystemOneResponse(
            model="m", usage=ts.Usage(),
            answers={"alternatives": ts.NoulAnswer(noul=p),
                     "kind": ts.ChoiceAnswer(choice="alternatives" if p > 0.5 else "complementary",
                                             confidence=0.9, probabilities={})})

    monkeypatch.setattr(client, "ask", fake_ask)
    fn = JevAlternatives(client, threshold=0.5, relation="prefers")

    assert fn("npm", "pnpm") is True
    assert fn("pnpm", "npm") is True            # same answer, no second request
    assert fn("npm", "Berlin") is False
    assert asked == [("npm", "pnpm"), ("Berlin", "npm")]   # sorted before sending
    assert [d["decided"] for d in fn.decisions] == [True, True, False]
    assert fn.frame().iloc[2]["kind"] == "complementary"


def test_jev_alternatives_plugs_into_the_temporal_graph(client, monkeypatch):
    from kgx.temporal import Fact, TemporalGraph

    def fake_ask(state, questions, *, refresh=False):
        p = 0.95 if {state["a"], state["b"]} == {"npm", "pnpm"} else 0.05
        return ts.SystemOneResponse(
            model="m", usage=ts.Usage(),
            answers={"alternatives": ts.NoulAnswer(noul=p),
                     "kind": ts.ChoiceAnswer(choice="alternatives", confidence=1.0, probabilities={})})

    monkeypatch.setattr(client, "ask", fake_ask)
    memory = TemporalGraph(alternative_fn=JevAlternatives(client, relation="prefers"))

    def fact(tail, when):
        return Fact(head="user", relation="prefers", tail=tail, head_name="Priya", tail_name=tail,
                    tail_type="tool", confidence=0.9, episode_id=when, valid_from=when)

    memory.assert_fact(fact("npm", "2026-01-01"))
    memory.assert_fact(fact("Python", "2026-01-02"))       # complementary: both stay current
    killed = memory.assert_fact(fact("pnpm", "2026-02-01"))  # alternative: supersedes npm

    assert [f.tail for f in killed] == ["npm"]
    assert sorted(f.tail for f in memory.current(relation="prefers")) == ["Python", "pnpm"]
    assert [(o.tail, n.tail) for o, n in memory.contradictions()] == [("npm", "pnpm")]


def test_picks_to_graphs_applies_min_prob():
    g = _pair_graph()
    pick = RelationPick(doc_id="d1", head="m0", tail="m1", head_text="Northwind", tail_text="Cascade",
                        head_type="company", tail_type="company", relation="acquires",
                        confidence=0.4, probabilities={"acquires": 0.4, NO_RELATION: 0.6})
    assert picks_to_graphs([g], [pick], min_prob=0.5)[0].edges == []
    assert len(picks_to_graphs([g], [pick], min_prob=0.3)[0].edges) == 1
