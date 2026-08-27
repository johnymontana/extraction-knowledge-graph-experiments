"""LLM-as-teacher distillation: turn Claude's output into GLiNER2.5 training data.

The premise of the "distil a frontier model into a small encoder" pipeline is
that labels are the expensive part and inference is the cheap part, so you pay
the LLM once per training example and never again. Everything here is the
plumbing between those two halves:

``synth_prompt`` / ``parse_documents``
    The sandbox ships ten documents and a 47-triple answer key, which is a test
    set, not a training set. Fine-tuning on those ten documents and then scoring
    against their gold facts would measure memorisation. So the unlabelled pool
    is manufactured: the teacher writes a *disjoint* fictional universe -- new
    companies, new people, new tickers -- in the same register. That keeps the
    ten real documents genuinely held out.

``chunk_sentences``
    Training units are one to three sentences, not whole documents. GLiNER's
    relation recall degrades past a few hundred words (see
    :meth:`kgx.extract.GlinerExtractor.extract_long`), and short units give the
    teacher less room to hallucinate a link across half a page.

``label_prompt`` / ``parse_label_batch``
    The teacher labels *many chunks per call*. Per-call overhead dominates the
    token cost through the CLI, so one call carrying twenty short texts costs
    about what one call carrying one costs.

``occurrence`` / ``snap_span`` / ``to_examples``
    The part that decides whether training works at all. An LLM label that is
    not a token-aligned span of the text is not supervision, and GLiNER2.5's
    boundary preprocessor raises on it *inside the training loop*, several
    minutes in -- ``validate_data=True`` does not catch it, because the
    validator's rule (case-insensitive substring) is looser than the
    preprocessor's (token alignment). Filtering here, loudly and with counts, is
    the difference between "the fine-tune did nothing" and knowing why.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .ontology import Ontology

__all__ = [
    "SEEDS",
    "Chunk",
    "synth_prompt",
    "parse_documents",
    "chunk_sentences",
    "label_prompt",
    "parse_label_batch",
    "to_examples",
    "occurrence",
    "snap_span",
    "quiet_train",
    "curve",
    "LabelStats",
]


# ---------------------------------------------------------------------------
# 1. an unlabelled pool that does not overlap the benchmark
# ---------------------------------------------------------------------------

SEEDS: list[dict[str, Any]] = [
    {
        "sector": "regional airlines and aircraft leasing",
        "companies": ["Kestrel Air Group", "Bluewater Aviation Holdings", "Pinnacle Lease Partners"],
        "people": ["Dana Okonjo", "Ruben Alvarez", "Hilde Brandt"],
        "geography": ["Denver", "Reykjavik", "Queensland"],
    },
    {
        "sector": "speciality chemicals and industrial coatings",
        "companies": ["Verdant Chemical Works", "Orinoco Polymers", "Steadfast Coatings Ltd"],
        "people": ["Ingrid Vasari", "Tomas Bergqvist", "Nadia Rahmani"],
        "geography": ["Rotterdam", "Baton Rouge", "Ulsan"],
    },
    {
        "sector": "grocery retail and cold-chain distribution",
        "companies": ["Marrow & Fields", "Glacier Cold Logistics", "Bramble Markets"],
        "people": ["Cecily Aduba", "Gregor Malik", "Yuki Tanabe"],
        "geography": ["Manchester", "Sao Paulo", "Winnipeg"],
    },
    {
        "sector": "medical devices and diagnostics",
        "companies": ["Corvus Medical Systems", "Lumen Diagnostics", "Fairhaven Surgical"],
        "people": ["Amara Diallo", "Piotr Zielinski", "Rosa Ferrante"],
        "geography": ["Galway", "Minneapolis", "Hyderabad"],
    },
    {
        "sector": "utilities and grid infrastructure",
        "companies": ["Thistledown Power", "Ardent Grid Services", "Mesa Renewables Corp"],
        "people": ["Kwame Boateng", "Elin Sorensen", "Marisol Reyes"],
        "geography": ["Cardiff", "Albuquerque", "Gdansk"],
    },
    {
        "sector": "asset management and payments infrastructure",
        "companies": ["Harrowgate Capital", "Tessellate Payments", "Blue Anchor Trust"],
        "people": ["Julian Osei", "Freya Lindqvist", "Sanjay Kulkarni"],
        "geography": ["Zurich", "Toronto", "Singapore"],
    },
    {
        "sector": "shipbuilding, ports and marine services",
        "companies": ["Tidewater Yards", "Consolidated Quay Holdings", "Petrel Marine Services"],
        "people": ["Anneke de Vries", "Omar Haddad", "Beatriz Cardoso"],
        "geography": ["Gothenburg", "Mobile", "Busan"],
    },
    {
        "sector": "agricultural inputs and food processing",
        "companies": ["Redstone Agriscience", "Copper Kettle Foods", "Highfield Grain Cooperative"],
        "people": ["Louis Traore", "Petra Novak", "Emeka Nwosu"],
        "geography": ["Saskatoon", "Bordeaux", "Nakuru"],
    },
]


def _type_lines(ontology: Ontology) -> tuple[str, str]:
    entity_lines = "\n".join(f"  - {e.name}: {e.description}" for e in ontology.entities)
    relation_lines = "\n".join(
        f"  - {r.name}({'|'.join(r.head)} -> {'|'.join(r.tail)}): {r.description}"
        for r in ontology.relations
    )
    return entity_lines, relation_lines


def synth_prompt(seed: Mapping[str, Any], ontology: Ontology, n: int = 5) -> str:
    """Ask the teacher for ``n`` short, fictional, in-domain news items."""
    _, relation_lines = _type_lines(ontology)
    return f"""Write {n} short business-news items about FICTIONAL companies in {seed['sector']}.

Use these invented names for the main actors and invent nothing that resembles a real firm:
  companies: {', '.join(seed['companies'])}
  people:    {', '.join(seed['people'])}
  places:    {', '.join(seed['geography'])}

Requirements:
- 110 to 170 words each. Four to seven sentences.
- Vary the register across the {n} items: a press release, a wire story, an SEC filing excerpt, an analyst note, an earnings-call summary.
- Between them the items must state facts of these kinds:
{relation_lines}
- Give each item at least one sentence that is hypothetical, negated, conditional or forward-looking ("would", "does not expect to", "if approved", "may face"), so the text is not a list of bare assertions.
- Use full company names on first mention and short forms afterwards.
- Do not reuse sentence structures across items.

Respond with JSON of exactly this shape and nothing else:
{{"documents": [{{"title": "<headline>", "text": "<the item>"}}]}}"""


def parse_documents(payload: Any, seed_index: int) -> list[dict[str, str]]:
    """``{"documents": [...]}`` (or a bare list) into ``doc_id``-stamped records."""
    items = payload.get("documents") if isinstance(payload, Mapping) else payload
    docs = []
    for i, item in enumerate(items or [], start=1):
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text", "")).strip()
        if len(text.split()) < 40:
            continue
        docs.append({
            "doc_id": f"s{seed_index:02d}d{i:02d}",
            "title": str(item.get("title", "")).strip(),
            "text": text,
        })
    return docs


# ---------------------------------------------------------------------------
# 2. chunking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    """One training unit: a short span of text with a stable id."""

    chunk_id: str
    doc_id: str
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


_SENT = re.compile(r"(?<=[.!?\"])\s+(?=[A-Z\"'(])")


def chunk_sentences(
    docs: Sequence[Mapping[str, Any]],
    *,
    min_words: int = 12,
    max_words: int = 55,
    text_key: str = "text",
    id_key: str = "doc_id",
) -> list[Chunk]:
    """Split documents into one-to-three-sentence chunks.

    Sentences are merged forward until the chunk clears ``min_words``; a chunk is
    never allowed past ``max_words`` unless a single sentence is already longer,
    in which case it is emitted whole rather than cut mid-clause.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        doc_id = doc.get(id_key, "doc")
        buf: list[str] = []
        n = 0
        for para in str(doc[text_key]).split("\n"):
            for sent in _SENT.split(para.strip()):
                sent = sent.strip()
                if not sent:
                    continue
                if buf and len(" ".join(buf + [sent]).split()) > max_words:
                    chunks.append(Chunk(f"{doc_id}c{n:02d}", doc_id, " ".join(buf)))
                    n, buf = n + 1, []
                buf.append(sent)
                if len(" ".join(buf).split()) >= min_words:
                    chunks.append(Chunk(f"{doc_id}c{n:02d}", doc_id, " ".join(buf)))
                    n, buf = n + 1, []
        if buf and len(" ".join(buf).split()) >= 6:
            chunks.append(Chunk(f"{doc_id}c{n:02d}", doc_id, " ".join(buf)))
    return chunks


# ---------------------------------------------------------------------------
# 3. batched teacher labelling
# ---------------------------------------------------------------------------

def label_prompt(chunks: Sequence[Chunk], ontology: Ontology) -> str:
    """One prompt that labels every chunk in ``chunks``.

    The instructions are the same ones :func:`kgx.llm._ontology_prompt` gives the
    teacher for a whole document, so the labels are the same function of the text
    that the teacher-row in the scoreboard is measuring -- only the batching
    differs.
    """
    entity_lines, relation_lines = _type_lines(ontology)
    body = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))
    return f"""Extract a knowledge graph from EACH numbered text below, using ONLY the entity types and relation types defined in this ontology.

ENTITY TYPES:
{entity_lines}

RELATION TYPES (head_types -> tail_types):
{relation_lines}

RULES:
- Use only the listed types. Do not invent new ones.
- Every relation's head and tail must be entities you also list for that text, and their types must satisfy the relation's declared head/tail types.
- Copy the exact surface string as it appears in that text for each entity. Character-for-character; do not normalise, expand or shorten it.
- Extract only what the text states or directly implies. Do not add outside knowledge.
- Do not extract from a sentence that is hypothetical, negated, conditional or forward-looking.
- A text with no extractable facts gets an entry with empty lists. Do not skip it.

Respond with JSON of exactly this shape and nothing else, one entry per numbered text, in order:
{{"results": [{{"id": 1, "entities": [{{"text": "<exact surface string>", "type": "<entity type>"}}], "relations": [{{"head": "<entity surface string>", "type": "<relation type>", "tail": "<entity surface string>"}}]}}]}}

TEXTS:
{body}"""


def parse_label_batch(payload: Any, chunks: Sequence[Chunk]) -> dict[str, dict[str, list]]:
    """Map a batched response back onto chunk ids, by ``id`` and by position.

    Models drop or renumber entries. Anything that cannot be matched to a chunk
    is discarded rather than guessed at, and the caller sees the shortfall in the
    returned dictionary's length.
    """
    items = payload.get("results") if isinstance(payload, Mapping) else payload
    out: dict[str, dict[str, list]] = {}
    for pos, item in enumerate(items or []):
        if not isinstance(item, Mapping):
            continue
        try:
            idx = int(item.get("id", pos + 1)) - 1
        except (TypeError, ValueError):
            idx = pos
        if not 0 <= idx < len(chunks):
            continue
        out[chunks[idx].chunk_id] = {
            "entities": [e for e in (item.get("entities") or []) if isinstance(e, Mapping)],
            "relations": [r for r in (item.get("relations") or []) if isinstance(r, Mapping)],
        }
    return out


# ---------------------------------------------------------------------------
# 4. labels -> InputExample
# ---------------------------------------------------------------------------

_ROLE_WORDS = {
    "chief", "the", "a", "an", "its", "their", "his", "her", "company", "group",
    "corporation", "inc", "ltd", "plc", "holdings", "partners", "board", "president",
    "officer", "executive", "director", "management", "firm", "business", "unit",
}


def occurrence(surface: str, text: str) -> str | None:
    """The literal slice of ``text`` equal to ``surface`` at token boundaries.

    ``TrainingDataset.validate`` accepts a plain case-insensitive *substring*.
    The boundary preprocessor does not: it aligns gold mentions to tokens, so
    ``"Denver"`` inside ``"Denver-based"`` passes validation and then raises
    ``ValueError: entity 'geography' was not found in sample N`` inside the
    collator, mid-epoch.

    The exclusion is hyphens only, which was measured rather than assumed: a
    mention immediately followed by ``'s`` collates fine eight times out of
    eight, a mention inside a hyphenated compound fails eight times out of
    eight.
    """
    if not surface:
        return None
    m = re.search(r"(?<![\w-])" + re.escape(surface) + r"(?![\w-])", text, re.I)
    return text[m.start():m.end()] if m else None


def snap_span(label: str, text: str) -> str | None:
    """Snap a teacher label back onto a literal span of ``text``, or give up.

    The dominant failure mode of LLM labelling is not hallucination, it is
    *canonicalisation*: the text says "Bluewater" and the label says "Bluewater
    Aviation Holdings", because the model resolved the short form on the way out.
    That is the right answer for a knowledge graph and the wrong answer for span
    supervision, so the longest word-prefix of the label that does occur verbatim
    is used instead.

    Deliberately conservative: the prefix must start with a capitalised,
    non-role word, run to at least five characters, and not end on an
    abbreviating full stop -- so "acquisition" does not become "acquired",
    "Chief Financial Officer" does not become "Chief", and "U.S. Securities and
    Exchange Commission" does not become "U.S.".
    """
    words = label.split()
    if not words:
        return None
    for k in range(len(words) - 1, 0, -1):
        cand = " ".join(words[:k])
        if (len(cand) < 5 or cand.endswith(".") or not cand[0].isupper()
                or cand.split()[0].strip(".,").lower() in _ROLE_WORDS):
            return None      # "U.S." out of "U.S. Securities and Exchange Commission" is not a name
        if not any(ch.isalnum() for ch in words[k - 1]):
            continue                      # never end a span on "&", "-" or a bare comma
        hit = occurrence(cand, text)
        if hit is not None:
            return hit
    return None


@dataclass
class LabelStats:
    """What survived the trip from teacher output to trainable supervision."""

    chunks: int = 0
    entities_in: int = 0
    entities_kept: int = 0
    unknown_type: int = 0
    span_not_in_text: int = 0
    span_repaired: int = 0
    relations_in: int = 0
    relations_kept: int = 0
    unknown_relation: int = 0
    dangling_endpoint: int = 0
    ontology_violation: int = 0
    empty_chunks: int = 0

    def as_record(self) -> dict[str, Any]:
        return {
            "chunks labelled": self.chunks,
            "chunks with no facts": self.empty_chunks,
            "entity labels": self.entities_in,
            "entity kept": self.entities_kept,
            "entity repaired by snapping": self.span_repaired,
            "entity dropped: unknown type": self.unknown_type,
            "entity dropped: span not in text": self.span_not_in_text,
            "relation labels": self.relations_in,
            "relation kept": self.relations_kept,
            "relation dropped: unknown name": self.unknown_relation,
            "relation dropped: endpoint gone": self.dangling_endpoint,
            "relation dropped: ontology violation": self.ontology_violation,
        }


def to_examples(
    chunks: Sequence[Chunk],
    labels: Mapping[str, Mapping[str, list]],
    ontology: Ontology,
    *,
    negatives: int | None = None,
    seed: int = 0,
    keep_empty: bool = True,
    repair_spans: bool = True,
    repairs: list | None = None,
) -> tuple[list[Any], list[dict[str, Any]], LabelStats]:
    """Convert teacher labels into ``InputExample`` objects.

    Returns ``(examples, records, stats)``. ``records`` carries the surviving
    surface mentions and triples per chunk, which is what a distillation-fidelity
    score is computed against.

    ``negatives`` is how many *absent* entity types to declare with an empty
    mention list on each example. GLiNER2.5 turns those into real queries with
    zero gold spans, which is the only way the fine-tune ever sees a type it is
    supposed to stay silent about. ``None`` means every absent type; ``0``
    disables negatives entirely.

    ``repair_spans`` runs :func:`snap_span` on labels that are not literal
    substrings; pass a list as ``repairs`` to collect what it rewrote.
    """
    from gliner2.training import InputExample, Relation

    valid_entities = list(ontology.entity_names)
    valid_set = set(valid_entities)
    rng = random.Random(seed)
    stats = LabelStats()
    examples, records = [], []
    repairs = repairs if repairs is not None else []

    for chunk in chunks:
        label = labels.get(chunk.chunk_id)
        if label is None:
            continue
        stats.chunks += 1

        by_type: dict[str, list[str]] = {}
        surface_type: dict[str, str] = {}
        rewritten: dict[str, str] = {}       # teacher's string -> the span it snapped to
        for item in label.get("entities", []):
            surface = str(item.get("text", "")).strip()
            etype = str(item.get("type", "")).strip()
            stats.entities_in += 1
            if etype not in valid_set:
                stats.unknown_type += 1
                continue
            # A label that is not a token-aligned span of this chunk is not
            # trainable supervision: GLiNER2.5's boundary preprocessor raises on
            # it mid-epoch. Snap what can be snapped, drop the rest.
            literal = occurrence(surface, chunk.text)
            if literal is None:
                snapped = snap_span(surface, chunk.text) if repair_spans else None
                if snapped is None:
                    stats.span_not_in_text += 1
                    continue
                repairs.append((chunk.chunk_id, surface, snapped))
                rewritten[surface.casefold()] = snapped
                surface = snapped
                stats.span_repaired += 1
            else:
                surface = literal
            bucket = by_type.setdefault(etype, [])
            if surface not in bucket:
                bucket.append(surface)
            surface_type.setdefault(surface.casefold(), etype)
            stats.entities_kept += 1

        relations, triples = [], []
        seen: set[tuple[str, str, str]] = set()
        for item in label.get("relations", []):
            rel = str(item.get("type", "")).strip()
            head = str(item.get("head", "")).strip()
            tail = str(item.get("tail", "")).strip()
            stats.relations_in += 1
            head = rewritten.get(head.casefold(), head)
            tail = rewritten.get(tail.casefold(), tail)
            try:
                ontology.relation(rel)
            except KeyError:
                stats.unknown_relation += 1
                continue
            htype = surface_type.get(head.casefold())
            ttype = surface_type.get(tail.casefold())
            if not htype or not ttype or head.casefold() == tail.casefold():
                stats.dangling_endpoint += 1
                continue
            if not ontology.permits(htype, rel, ttype):
                stats.ontology_violation += 1
                continue
            key = (head.casefold(), rel, tail.casefold())
            if key in seen:
                continue
            seen.add(key)
            relations.append(Relation(rel, head=head, tail=tail))
            triples.append((head, rel, tail))
            stats.relations_kept += 1

        if not by_type and not keep_empty:
            continue
        if not by_type and not relations:
            stats.empty_chunks += 1

        absent = [t for t in valid_entities if t not in by_type]
        if negatives is None:
            chosen = absent
        elif negatives > 0:
            chosen = rng.sample(absent, min(negatives, len(absent)))
        else:
            chosen = []
        entities = {**by_type, **{t: [] for t in chosen}}

        examples.append(InputExample(text=chunk.text, entities=entities, relations=relations))
        records.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
                        "text": chunk.text, "triples": triples,
                        "mentions": [(m, t) for t, ms in by_type.items() for m in ms]})
    return examples, records, stats


# ---------------------------------------------------------------------------
# 5. running the trainer inside a notebook
# ---------------------------------------------------------------------------

_INTERESTING = (
    "Dropped relation", "Validation", "invalid", "Truncat", "truncat",
    "capacity", "Froze", "LoRA enabled", "New best", "Total optimization steps",
)


def quiet_train(trainer: Any, train_data: Any, eval_data: Any = None) -> tuple[dict, list[str]]:
    """Train with the progress bars swallowed, returning ``(results, log)``.

    ``ExtractorTrainer`` writes a tqdm bar per step and a logger line per epoch
    and per checkpoint; in a notebook that is a screen of carriage returns per
    run, and it buries the two messages you actually need: ``"Validation: Found
    N invalid record(s)"`` and ``"Dropped relation auxiliary loss after ..."``
    -- the second means relation supervision was switched off and the run is
    entity-only.

    ``contextlib.redirect_stderr`` alone is not enough. The trainer calls
    ``logging.basicConfig``, so the root handler holds a reference to whichever
    stream was current when it was installed, and ``tqdm.auto`` in a kernel does
    not write to ``sys.stderr`` at all. Both have to be intercepted directly.
    """
    import contextlib
    import io
    import logging

    from gliner2.training import trainer as _trainer_mod

    buf = io.StringIO()
    root = logging.getLogger()
    muted = [(h, h.level) for h in root.handlers]
    sink = logging.StreamHandler(buf)
    sink.setLevel(logging.INFO)
    real_tqdm = _trainer_mod.tqdm

    def silent_tqdm(*args: Any, **kwargs: Any) -> Any:
        kwargs["disable"] = True
        return real_tqdm(*args, **kwargs)

    for handler, _ in muted:
        handler.setLevel(logging.CRITICAL + 1)
    root.addHandler(sink)
    _trainer_mod.tqdm = silent_tqdm
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            results = trainer.train(train_data=train_data, eval_data=eval_data)
    finally:
        _trainer_mod.tqdm = real_tqdm
        root.removeHandler(sink)
        for handler, level in muted:
            handler.setLevel(level)

    log = [ln for ln in buf.getvalue().splitlines() if any(k in ln for k in _INTERESTING)]
    return results, log


def curve(results: Mapping[str, Any]) -> Any:
    """Per-epoch train and eval loss out of a ``train()`` result, as a frame."""
    import pandas as pd

    train = pd.DataFrame(results["train_metrics_history"])
    per_epoch = (train.assign(ep=train["epoch"].astype(int) + 1)
                      .groupby("ep")["loss"].mean().rename("train_loss"))
    ev = pd.DataFrame(results["eval_metrics_history"])
    ev = ev.assign(ep=range(1, len(ev) + 1)).set_index("ep")["eval_loss"]
    return pd.concat([per_epoch, ev], axis=1)
