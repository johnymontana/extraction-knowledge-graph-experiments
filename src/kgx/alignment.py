"""Entity alignment in three outcomes: TypeSafe's cookbook decision, asked of GLiNER2.5-Decide.

`TypeSafe's entity-alignment cookbook
<https://docs.typesafe.ai/cookbooks/entity_alignment>`_ decides, for each
candidate pair of entities from two catalogues, one of three outcomes: leave
them unlinked, send them to a curator, or assert ``sameAs``. The decision is
one ``Score`` over three worded levels, and the routing rule is to round the
score to the nearest level. No threshold is fitted anywhere: the wording of the
levels is the decision. Three ``Noul`` questions (same name? same brewery? same
style?) ride along in the same request to tell the curator which field
disagrees.

This module keeps that decision as plain data -- :data:`LEVELS`,
:data:`OUTCOME`, :data:`INSTRUCTION`, :data:`NOULS`, :func:`route` -- with no SDK
import, and translates the questions for GLiNER2.5-Decide as literally as its
API allows (:func:`cookbook_schema`):

**Score** becomes a single-label task over three labels, the cookbook's level
wordings attached as label descriptions and its instruction as the task
instruction. Decide needs label *names* where TypeSafe's levels have none, so
each level is named by the product relation it describes. The score is then the
same quantity a TypeSafe ``Score`` returns: the probability-weighted position on
the levels (:func:`level_score`), 0 to 2.

**Noul** becomes a ``yes``/``no`` task with the question as its instruction --
the model card's "question over a passage" shape. P(yes) stands in for the Noul.

**One request, four questions** becomes one schema with four tasks: one
forward pass per pair. **State** is the same JSON (:func:`state_text`), passed
as the text.

The Jev side is here too, so both models answer from one definition:
:func:`jev_questions` is the cookbook's ``QUESTIONS`` verbatim as TypeSafe SDK
objects, and :func:`ask_jev` sends them through
:class:`kgx.typesafe.CachedTypeSafe` and returns the same :class:`PairJudgment`
Decide's answers come back as. The SDK is imported only inside those two
functions: importing this module needs no key and no ``typesafe_sdk``.

What happens then is notebook 16's subject. In short: Decide's distribution
over the three levels is nearly flat on every pair, so the probability-weighted
score never leaves the middle level and :func:`route` sends every pair to the
curator. The *ordering* of pairs by P(same) carries real signal, so a threshold
fitted on labelled pairs (:func:`fit_cuts`) can route -- which is the step the
cookbook's design exists to avoid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "LEVELS",
    "LEVEL_LABELS",
    "OUTCOME",
    "INSTRUCTION",
    "NOULS",
    "LINK_TASK",
    "state",
    "state_text",
    "cookbook_schema",
    "jev_questions",
    "ask_jev",
    "level_score",
    "confidence",
    "route",
    "PairJudgment",
    "align",
    "auc",
    "fit_cuts",
    "route_by_cuts",
]

#: The cookbook's three levels, verbatim, lowest first. Each level is one outcome.
LEVELS: tuple[str, ...] = (
    "They describe two different products.",
    "They describe closely related products that may or may not be the same one: "
    "a variant, a special edition, or a name that could plausibly refer to either.",
    "They describe one and the same product.",
)

#: The names Decide is shown for the levels. TypeSafe's levels are unnamed
#: criteria; a Decide label must have a name, and the name is what it reads most.
LEVEL_LABELS: tuple[str, ...] = ("different product", "related product", "same product")

OUTCOME: dict[int, str] = {0: "leave unlinked", 1: "curator queue", 2: "assert sameAs"}

INSTRUCTION = "How do the two entity descriptions relate as products?"

#: The cookbook's three Nouls, keyed by its question ids. Alcohol content has
#: none: comparing two numbers is arithmetic.
NOULS: dict[str, str] = {
    "same_name": "Do the two entities state the same beer name?",
    "same_brewery": "Are the two entities from the same brewery?",
    "same_style": "Do the two entities describe the same beer style?",
}

#: The cookbook's id for the Score question. In TypeSafe a question id is never
#: sent to the model; in GLiNER2 the task name is encoded into the prompt, so
#: this string is part of what Decide reads.
LINK_TASK = "link_state"


def state(pair: Mapping[str, Any]) -> dict[str, Any]:
    """The cookbook's state: both entities in one object, and nothing else --
    not the id, not the benchmark's answer."""
    return {"entity_a": pair["entity_a"], "entity_b": pair["entity_b"]}


def state_text(pair: Mapping[str, Any]) -> str:
    """:func:`state` as JSON: Decide's text.

    ``ensure_ascii=False`` so the published text reaches the model as published,
    mis-decoded characters included.
    """
    return json.dumps(state(pair), ensure_ascii=False)


def cookbook_schema(*, nouls: bool = True, task: str = LINK_TASK):
    """The cookbook's four questions as one Decide schema: one forward pass per pair.

    ``nouls=False`` asks the Score question alone. GLiNER2 encodes every task in
    one prompt, so the Nouls riding along can move the Score's answer; the
    notebook measures by how much.
    """
    from gliner2.classification import ClassificationSchema

    schema = ClassificationSchema().single(
        task, dict(zip(LEVEL_LABELS, LEVELS)), instruction=INSTRUCTION)
    if nouls:
        for qid, question in NOULS.items():
            schema.single(qid, ["yes", "no"], instruction=question)
    return schema


def level_score(probabilities: Sequence[float]) -> float:
    """Probability-weighted position on the levels, lowest level 0.

    The number a TypeSafe ``Score`` returns as ``score``: all the mass on
    ``same product`` gives 2.0, a uniform distribution over three levels 1.0.
    """
    return float(sum(i * p for i, p in enumerate(probabilities)))


def confidence(probabilities: Sequence[float]) -> float:
    """Concentration of a distribution: 1 on one option, 0 when uniform.

    ``(n * max - 1) / (n - 1)``, the approximation the TypeSafe confidence
    docs give for their ``confidence`` field. TypeSafe's own definition may
    differ in detail; this is for reading Decide's distributions on the same
    scale, not for reproducing TypeSafe's numbers.
    """
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (n * max(probabilities) - 1) / (n - 1)))


def route(score_value: float) -> str:
    """The cookbook's whole decision rule: the nearest level names the outcome."""
    return OUTCOME[min(int(score_value + 0.5), len(LEVELS) - 1)]


@dataclass
class PairJudgment:
    """One model's answers for one candidate pair.

    ``probabilities`` is over the levels, lowest first; ``properties`` holds the
    per-field answers -- P(yes) for Decide's yes/no tasks, the ``noul`` for
    Jev's -- keyed by question id.

    A model that reports its own score and confidence (TypeSafe) is taken at
    its word through ``reported_score`` and ``reported_confidence``: Jev's
    probabilities come back rounded to two places, so recomputing the score from
    them can land a hundredth off, which is enough to move a pair across a cut
    point. For Decide both are computed from the distribution.
    """

    pair_id: str
    probabilities: dict[str, float]
    properties: dict[str, float] = field(default_factory=dict)
    reported_score: float | None = None
    reported_confidence: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def levels(self) -> list[float]:
        return list(self.probabilities.values())

    @property
    def score(self) -> float:
        if self.reported_score is not None:
            return self.reported_score
        return level_score(self.levels)

    @property
    def confidence(self) -> float:
        if self.reported_confidence is not None:
            return self.reported_confidence
        return confidence(self.levels)

    @property
    def p_top(self) -> float:
        """Probability of the highest level -- P(same product) for the cookbook's levels."""
        return self.levels[-1]

    @property
    def outcome(self) -> str:
        """:func:`route` applied to :attr:`score`. Meaningful for three levels only."""
        return route(self.score)

    def as_record(self) -> dict[str, Any]:
        return {"id": self.pair_id, "score": round(self.score, 3),
                "confidence": round(self.confidence, 3), "outcome": self.outcome,
                **{label: round(p, 3) for label, p in self.probabilities.items()},
                **{qid: round(p, 3) for qid, p in self.properties.items()}}


def align(
    judge: Any,
    pairs: Sequence[Mapping[str, Any]],
    schema: Any = None,
    *,
    task: str = LINK_TASK,
    labels: Sequence[str] = LEVEL_LABELS,
    render: Callable[[Mapping[str, Any]], str] = state_text,
) -> list[PairJudgment]:
    """Ask every pair the same schema, in one batch.

    ``schema`` defaults to :func:`cookbook_schema`. ``task`` and ``labels`` say
    which task holds the levels and in what order, lowest first, so that a
    variant schema (other label words, two levels, a yes/no question with
    ``labels=("no", "yes")``) goes through the same code. Every other task with
    a ``yes`` label is reported as a property.
    """
    schema = cookbook_schema() if schema is None else schema
    out = judge.probabilities([render(p) for p in pairs], schema)
    judgments = []
    for pair, answers in zip(pairs, out):
        dist = answers[task]
        judgments.append(PairJudgment(
            pair_id=pair["id"],
            probabilities={label: dist[label] for label in labels},
            properties={name: probs["yes"] for name, probs in answers.items()
                        if name != task and "yes" in probs},
        ))
    return judgments


# ---------------------------------------------------------------------------
# Jev -- the cookbook's own model, through the repo's cached client
# ---------------------------------------------------------------------------


def jev_questions() -> dict[str, Any]:
    """The cookbook's ``QUESTIONS``, verbatim, as TypeSafe SDK objects."""
    from typesafe_sdk import Noul, Score

    return {
        LINK_TASK: Score(instructions=INSTRUCTION, criteria=list(LEVELS)),
        **{qid: Noul(instructions=question) for qid, question in NOULS.items()},
    }


def ask_jev(
    client: Any, pairs: Sequence[Mapping[str, Any]], questions: Mapping[str, Any] | None = None
) -> list[PairJudgment]:
    """One System One request per pair -- state and questions as the cookbook sends them.

    ``client`` is a :class:`kgx.typesafe.CachedTypeSafe` (anything with its
    ``ask(state, questions)``), so a pair asked once is replayed from disk after
    that. Requests go one at a time: the client's counters are not guarded for
    threads, and the requests cached by notebooks 10-14 took a median 0.16 s
    each, so a sequential pass over the 450 pairs is a minute or two.

    The Score's ``legend`` names each level by its criterion text; the levels
    are matched back to :data:`LEVEL_LABELS` by that text rather than by
    position, so a reordered legend cannot silently swap two levels.
    """
    questions = jev_questions() if questions is None else questions
    label_for = dict(zip(LEVELS, LEVEL_LABELS))
    judgments = []
    for pair in pairs:
        response = client.ask(state(pair), questions)
        link = response.answers[LINK_TASK]
        by_label = {label_for[link.legend[k]]: float(p) for k, p in link.probabilities.items()}
        judgments.append(PairJudgment(
            pair_id=pair["id"],
            probabilities={label: by_label[label] for label in LEVEL_LABELS},
            properties={qid: float(response.answers[qid].noul)
                        for qid in questions if qid != LINK_TASK},
            reported_score=float(link.score),
            reported_confidence=float(link.confidence),
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
        ))
    return judgments


# ---------------------------------------------------------------------------
# Measurement -- what the cookbook does not need, and Decide does
# ---------------------------------------------------------------------------


def auc(values: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve: the chance a random match outranks a random non-match.

    Mann-Whitney, ties counted as half. Threshold-free, so it measures whether a
    signal *orders* the pairs, independently of where its values sit.
    """
    ranked = sorted(zip(values, labels), key=lambda vl: vl[0])
    ranks = [0.0] * len(ranked)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(1 for _, label in ranked if label)
    n_neg = len(ranked) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("auc needs at least one match and one non-match")
    rank_sum = sum(r for r, (_, label) in zip(ranks, ranked) if label)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_cuts(
    values: Sequence[float], labels: Sequence[bool], *, max_missed: float = 0.05
) -> tuple[float, float]:
    """Two cut points fitted to labelled pairs, with the cookbook's asymmetry.

    A wrong merge is the expensive mistake, so the upper cut is the highest
    value any non-match reached: :func:`route_by_cuts` merges only above it,
    which is zero false merges on the fitting data. A missed match only leaves a
    duplicate, so the lower cut lets ``max_missed`` of the matches fall below it
    into ``leave unlinked``. Everything between goes to the curator. If the
    lower cut lands above the upper one there is no curator band.
    """
    pos = sorted(v for v, label in zip(values, labels) if label)
    neg = [v for v, label in zip(values, labels) if not label]
    if not pos or not neg:
        raise ValueError("fit_cuts needs at least one match and one non-match")
    high = max(neg)
    low = pos[min(int(max_missed * len(pos)), len(pos) - 1)]
    return min(low, high), high


def route_by_cuts(value: float, low: float, high: float) -> str:
    """Route on fitted cuts: below ``low`` unlinked, above ``high`` merged."""
    if value > high:
        return OUTCOME[2]
    if value < low:
        return OUTCOME[0]
    return OUTCOME[1]
