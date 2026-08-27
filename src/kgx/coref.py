"""Making conversations legible to an extractor that has no coreference model.

This is the step people skip, and it is the step that decides whether GLiNER
looks broken on chat logs. Half the facts in a real transcript are stated about
"I", "he", or "the migration". An encoder-only pipeline has no escape hatch for
that: there is no pronoun resolution anywhere in GLiNER2.5, so a mention of
``"I"`` is either extracted as a ``user`` span with no name attached, or missed.

Rather than pretend otherwise, three deterministic layers run *before*
extraction, each inspectable and each individually ablatable:

1. **Speaker attribution + first-person substitution.** Every turn is prefixed
   with the speaker's real name, and first-person pronouns in a user turn are
   rewritten to that name. Rule, not a model. This single layer recovers most
   agent-memory edges -- the notebook measures the delta.
2. **Bare first-name expansion.** ``"Dev"`` becomes ``"Dev Shah"`` when exactly
   one person in the session-local roster has that given name. Two candidates
   means the mention is genuinely ambiguous, and it is left alone for entity
   resolution to flag rather than guessed at.
3. **Recency-bound third-person pronouns.** ``"he"`` / ``"she"`` / ``"they"`` at
   the head of a clause bind to the most recently mentioned non-user person,
   within a window of turns, and only when that antecedent is unambiguous.

Layer 3 is crude and will be wrong sometimes. It is included because it is
honest, cheap and auditable -- every rewrite is recorded in
:class:`Rewrite` so its accuracy can actually be checked instead of assumed.

A fourth layer, :class:`NeuralCorefPreprocessor`, runs a *trained* coreference
model on top of the rules. The dependency is optional (``fastcoref``, MIT, or
``stanza``, Apache-2.0) and nothing here imports it until you call
:func:`load_coref_engine`. It is an addition to layer 1, never a replacement for
it: none of the four engines tested can bind a bare "I" to a speaker, because
there is no antecedent in the text until layer 1 writes the ``Name:`` prefix the
model then binds to. That is not an OntoNotes artefact -- stanza's default
package is CorefUD-trained and fails the same way -- it is a property of running
coreference on a transcript whose speakers are metadata rather than text.

Notebook 08 measures the whole thing end to end. Stacked on top of the rules the
model recovered no gold fact the rules missed, and its entire contribution was
twelve fewer spurious canonical edges. Used *instead of* the first-person layer
it does gain one fact (``Dev Shah works_at Northwind Logistics``) and loses
several more, scoring below the rules alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Rewrite", "RenderedSession", "ConversationPreprocessor", "USER_CANON_ID",
    "NeuralCorefPreprocessor", "FastCorefEngine", "StanzaCorefEngine",
    "load_coref_engine", "substitute_clusters", "cluster_representative", "ALL_PRONOUNS",
]

USER_CANON_ID = "user:__self__"

FIRST_PERSON_SUBJECT = {"i"}
FIRST_PERSON_OBJECT = {"me"}
FIRST_PERSON_POSS = {"my", "mine"}
FIRST_PERSON_REFLEX = {"myself"}
FIRST_PERSON_PLURAL = {"we", "us", "our", "ours"}
THIRD_PERSON = {"he", "she", "they", "him", "her", "them", "his", "their", "hers", "theirs"}
POSSESSIVE_THIRD = {"his", "her", "their", "hers", "theirs"}


@dataclass
class Rewrite:
    """One audited text substitution, so the layers can be evaluated not trusted."""

    turn_index: int
    layer: str
    original: str
    replacement: str
    reason: str = ""

    def as_record(self) -> dict[str, Any]:
        return {
            "turn": self.turn_index,
            "layer": self.layer,
            "from": self.original,
            "to": self.replacement,
            "reason": self.reason,
        }


@dataclass
class RenderedSession:
    """A session flattened to text, with the rewrite log that produced it."""

    session_id: str
    text: str
    turn_texts: list[str] = field(default_factory=list)
    rewrites: list[Rewrite] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def rewrites_frame(self):
        import pandas as pd

        return pd.DataFrame([r.as_record() for r in self.rewrites])


class ConversationPreprocessor:
    """Rewrite a chat transcript into text an extractor can attribute correctly.

    Parameters
    ----------
    user_name:
        The real name of the human the assistant serves. First-person mentions in
        user turns are rewritten to this.
    assistant_name:
        Label used for assistant turns. Assistant turns are included by default
        because they carry facts the user confirmed, but they are never treated
        as first-person about the user.
    roster:
        Known ``person`` full names, used by layer 2. It can be grown across
        sessions -- pass the canonical registry's person names back in.
    layers:
        Any subset of ``{"speaker", "first_person", "first_name", "pronoun"}``.
        Ablating a layer here is how the notebook measures what each one is
        worth.
    """

    def __init__(
        self,
        user_name: str,
        *,
        assistant_name: str = "Assistant",
        roster: Iterable[str] = (),
        layers: Iterable[str] = ("speaker", "first_person", "first_name", "pronoun"),
        pronoun_window: int = 3,
        include_assistant: bool = True,
    ) -> None:
        self.user_name = user_name
        self.assistant_name = assistant_name
        self.roster = list(dict.fromkeys(roster))
        self.layers = set(layers)
        self.pronoun_window = pronoun_window
        self.include_assistant = include_assistant

    # -- roster ----------------------------------------------------------

    def learn_roster(self, names: Iterable[str]) -> "ConversationPreprocessor":
        """Add known full names (e.g. from the canonical registry)."""
        self.roster = list(dict.fromkeys([*self.roster, *names]))
        return self

    def _first_name_map(self) -> dict[str, str]:
        """Given name -> full name, only where the given name is unambiguous."""
        buckets: dict[str, set[str]] = {}
        for full in self.roster:
            parts = full.split()
            if len(parts) < 2:
                continue
            buckets.setdefault(parts[0].casefold(), set()).add(full)
            # tolerate diacritics being dropped in casual typing
            plain = _strip_accents(parts[0]).casefold()
            buckets.setdefault(plain, set()).add(full)
        return {k: next(iter(v)) for k, v in buckets.items() if len(v) == 1}

    # -- layers ----------------------------------------------------------

    def _sub_first_person(self, text: str, idx: int, log: list[Rewrite]) -> str:
        """Rewrite first-person references to the user's name.

        Possessives become ``"Priya Raman's"``; plurals ("we", "our") are left
        alone -- they usually mean the team, and rewriting them to the user
        invents facts.
        """
        owner = f"{self.user_name}'s"

        def repl(match: re.Match[str]) -> str:
            word = match.group(0)
            low = word.casefold()
            if low in FIRST_PERSON_POSS:
                out = owner
            elif low in FIRST_PERSON_REFLEX:
                out = self.user_name
            elif low in FIRST_PERSON_SUBJECT | FIRST_PERSON_OBJECT:
                out = self.user_name
            else:
                return word
            log.append(Rewrite(idx, "first_person", word, out, "user turn, first person"))
            return out

        # (?!['\u2019]\w) keeps contractions intact: "I'm" must not become
        # "Priya Raman'm". \b alone matches between the I and the apostrophe.
        pattern = (
            r"\b("
            + "|".join(sorted(FIRST_PERSON_SUBJECT | FIRST_PERSON_OBJECT
                              | FIRST_PERSON_POSS | FIRST_PERSON_REFLEX))
            + r")\b(?![\'\u2019]\w)"
        )
        return re.sub(pattern, repl, text, flags=re.IGNORECASE)

    def _expand_first_names(
        self, text: str, idx: int, name_map: Mapping[str, str], log: list[Rewrite]
    ) -> str:
        for given, full in name_map.items():
            if full.casefold() in text.casefold():
                continue  # already fully qualified in this turn

            def repl(match: re.Match[str], full=full, given=given) -> str:
                log.append(
                    Rewrite(idx, "first_name", match.group(0), full,
                            f"unique given name in roster ({given})")
                )
                return full

            # Case-SENSITIVE on the initial letter. Many given names are also
            # ordinary words -- "Dev", "Mark", "Bill", "Grace", "Rose" -- and a
            # case-insensitive rewrite turns "no PII in dev environments" into
            # "no PII in Dev Shah environments", inventing a fact about a person.
            # Requiring the capital costs a rewrite on a lowercased mention and
            # buys immunity to the whole class.
            #
            # sorted(), and the re-check inside the loop, so that expanding the
            # accent-stripped spelling does not then re-expand the accented one
            # inside the name it just wrote.
            for spelling in sorted({given, _strip_accents(given)}):
                if full.casefold() in text.casefold():
                    break
                pattern = rf"\b{re.escape(spelling[:1].upper() + spelling[1:])}\b"
                text = re.sub(pattern, repl, text)
        return text

    def _bind_pronouns(
        self, text: str, idx: int, recent: Sequence[str], log: list[Rewrite]
    ) -> str:
        """Bind a leading third-person pronoun to an unambiguous recent person.

        Deliberately conservative: only the first pronoun of a sentence, only
        when exactly one person was mentioned in the recent window, and never
        when the antecedent would be the user (that is layer 1's job).
        """
        if len(set(recent)) != 1:
            return text
        antecedent = recent[0]

        def repl(match: re.Match[str]) -> str:
            lead, word = match.group(1), match.group(2)
            out = f"{antecedent}'s" if word.casefold() in POSSESSIVE_THIRD else antecedent
            log.append(
                Rewrite(idx, "pronoun", word, out,
                        f"sentence-initial pronoun, unique antecedent in last "
                        f"{self.pronoun_window} turns")
            )
            return f"{lead}{out}"

        pattern = r"(^|(?<=[.!?]\s))(" + "|".join(sorted(THIRD_PERSON)) + r")\b"
        return re.sub(pattern, repl, text, flags=re.IGNORECASE | re.MULTILINE)

    # -- driver ----------------------------------------------------------

    def render(self, session: Mapping[str, Any]) -> RenderedSession:
        """Flatten one session into extraction-ready text.

        ``session`` is ``{"session_id", "turns": [{"speaker", "text"}], ...}``.
        """
        name_map = self._first_name_map() if "first_name" in self.layers else {}
        log: list[Rewrite] = []
        out_turns: list[str] = []
        recent_people: list[tuple[int, str]] = []

        for idx, turn in enumerate(session.get("turns", [])):
            speaker = str(turn.get("speaker", "user")).lower()
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            if speaker != "user" and not self.include_assistant:
                continue

            is_user = speaker == "user"
            if is_user and "first_person" in self.layers:
                text = _expand_contractions(text, self.user_name)
                text = self._sub_first_person(text, idx, log)
            if "first_name" in self.layers:
                text = self._expand_first_names(text, idx, name_map, log)
            # Record people named in THIS turn before binding its pronouns: in
            # "Dev Shah is stuck. He's blocked on the driver docs." the
            # antecedent is in the same turn.
            for full in self.roster:
                if full.casefold() in text.casefold() and full != self.user_name:
                    recent_people.append((idx, full))

            if "pronoun" in self.layers:
                window = [n for i, n in recent_people if idx - i <= self.pronoun_window]
                text = self._bind_pronouns(text, idx, window, log)

            if "speaker" in self.layers:
                label = self.user_name if is_user else self.assistant_name
                out_turns.append(f"{label}: {text}")
            else:
                out_turns.append(text)

        return RenderedSession(
            session_id=str(session.get("session_id", "session")),
            text="\n".join(out_turns),
            turn_texts=out_turns,
            rewrites=log,
            meta={
                k: v for k, v in session.items() if k not in {"turns", "session_id"}
            },
        )

    def render_all(self, sessions: Iterable[Mapping[str, Any]]) -> list[RenderedSession]:
        return [self.render(s) for s in sessions]

    def episodes(
        self,
        sessions: Iterable[Mapping[str, Any]],
        *,
        window: int = 4,
        stride: int | None = None,
        drop_speaker_mentions: bool = True,
    ) -> list[dict[str, Any]]:
        """Split sessions into overlapping windows of turns -- the extraction unit.

        Feeding a whole transcript to the model in one call reads like the
        obvious thing to do and is a mistake. Relation recall collapses on long
        conversational text: the model has to decide how many instances of each
        of fifteen relations exist across the entire document, and on a rambling
        500-word transcript the answer it settles on is frequently zero. The
        notebook measures this directly.

        Short overlapping windows are also what temporal memory systems actually
        do -- Graphiti's "episode", mem0's message batch. Each window keeps the
        turn's own context, and the overlap means a fact stated across a turn
        boundary is still seen whole by at least one window.

        Returns dicts ready for :meth:`~kgx.extract.GlinerExtractor.extract_batch`,
        each carrying ``session_id`` and ``timestamp`` so provenance survives.
        """
        stride = stride if stride is not None else max(1, window - 1)
        out: list[dict[str, Any]] = []
        for session in sessions:
            rendered = self.render(session)
            turns = rendered.turn_texts
            if not turns:
                continue
            starts = list(range(0, max(len(turns) - window + 1, 1), stride))
            # a session whose length is not a multiple of the stride would
            # otherwise lose its final turns entirely
            if starts and starts[-1] + window < len(turns):
                starts.append(max(0, len(turns) - window))
            for start in starts:
                chunk = turns[start : start + window]
                if not chunk:
                    continue
                out.append(
                    {
                        "doc_id": f"{rendered.session_id}:w{start:02d}",
                        "text": "\n".join(chunk),
                        "session_id": rendered.session_id,
                        "turn_start": start,
                        "timestamp": rendered.meta.get("timestamp") or rendered.meta.get("date"),
                        "drop_speakers": (
                            [self.assistant_name] if drop_speaker_mentions else []
                        ),
                    }
                )
        return out

    def clean_mentions(self, graphs: Iterable[Any]) -> int:
        """Drop mentions that are just a speaker label the renderer inserted.

        Prefixing turns with ``"Assistant: ..."`` helps the model attribute
        statements, but it also puts the literal string "Assistant" in the text,
        and the extractor duly types it as a person. Those are artefacts of the
        rendering, not entities. Mutates the graphs in place and returns the
        number removed.
        """
        labels = {self.assistant_name.casefold(), "assistant", "user"}
        removed = 0
        for graph in graphs:
            keep_ids = {
                m.mention_id
                for m in graph.mentions
                if m.text.strip().casefold() not in labels
            }
            removed += len(graph.mentions) - len(keep_ids)
            graph.mentions = [m for m in graph.mentions if m.mention_id in keep_ids]
            graph.edges = [
                e for e in graph.edges if e.head in keep_ids and e.tail in keep_ids
            ]
            graph.reindex()
        return removed


# First-person contractions, rewritten to third person with the user's name in
# one step. Splitting "I'm" on the word boundary would emit "<name>'m", and
# expanding it first would emit "<name> am" -- both wrong, and the second is the
# kind of ungrammatical input that quietly costs recall.
_CONTRACTIONS = [
    (re.compile(r"\bI['\u2019]m\b", re.I), "{name} is"),
    (re.compile(r"\bI['\u2019]ve\b", re.I), "{name} has"),
    (re.compile(r"\bI['\u2019]ll\b", re.I), "{name} will"),
    (re.compile(r"\bI['\u2019]d\b", re.I), "{name} would"),
]


def _expand_contractions(text: str, name: str) -> str:
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement.format(name=name), text)
    return text


def _strip_accents(text: str) -> str:
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def user_mentions(mentions: Iterable[Any], user_name: str) -> set[str]:
    """Mention ids that refer to the assistant's own user.

    After layer 1, first-person references carry the user's name, so this is a
    name match rather than a pronoun heuristic. Everything matching is pinned to
    :data:`USER_CANON_ID` so the user is one node across every episode, whatever
    the resolver would otherwise have decided.
    """
    target = user_name.casefold()
    out = set()
    for m in mentions:
        text = getattr(m, "text", "").casefold()
        if text == target or text in {"i", "me", "my", "myself", "user"}:
            out.add(getattr(m, "mention_id"))
        elif getattr(m, "type", "") == "user" and target.split()[0] in text:
            out.add(getattr(m, "mention_id"))
    return out


# ---------------------------------------------------------------------------
# Optional layer: a trained coreference model
# ---------------------------------------------------------------------------
#
# Everything above is rules. This section wires a real coreference resolver into
# the same interface, so the two can be compared on the same corpus with the
# same extractor. The dependency is OPTIONAL -- ``kgx`` does not require it, and
# nothing in this module imports it until you ask for an engine by name:
#
#     uv add fastcoref          # MIT, ~700 MB of weights, ~0.2 s / 500 words
#     uv add stanza peft        # Apache-2.0, ~2.3 GB, ~1.2 s / 500 words
#
# Notebook 08 measures whether it is worth it. The short version: on chat
# transcripts these models resolve third-person and second-person pronouns well
# and cannot resolve "I" to a speaker at all unless a literal ``Name:`` prefix is
# already sitting in the token stream -- which is layer 1's job, not theirs.

_PRONOUN_SUBJ = {"i", "we", "you", "he", "she", "it", "they"}
_PRONOUN_OBJ = {"me", "us", "him", "her", "them"}
_PRONOUN_POSS = {"my", "our", "your", "his", "her", "its", "their",
                 "mine", "ours", "yours", "hers", "theirs"}
_PRONOUN_REFL = {"myself", "ourselves", "yourself", "yourselves",
                 "himself", "herself", "itself", "themselves"}
ALL_PRONOUNS = _PRONOUN_SUBJ | _PRONOUN_OBJ | _PRONOUN_POSS | _PRONOUN_REFL

# Unambiguously determiner-possessive. "her" and "your" are handled by lookahead
# because "told her" and "her team" need different substitutions.
_ALWAYS_POSS = {"my", "our", "his", "its", "their", "mine", "ours", "yours",
                "hers", "theirs"}
_MAYBE_POSS = {"her", "your"}

_DEFINITE_HEADS = {"the", "that", "this", "those", "these"}
_NAME_STOPWORDS = {"the", "of", "and", "a", "an", "de", "van"}

# A coref model's mention span for "I'm" is the single character "I", so a naive
# span substitution emits "Priya Raman'm". Swallow the clitic and conjugate.
_CLITIC = re.compile(r"^['\u2019](m|ve|ll|re|d)\b", re.IGNORECASE)
_PLURAL_PRONOUNS = {"we", "they", "you", "us", "them"}
_CLITIC_FORMS = {
    "m": ("is", "are"), "ve": ("has", "have"), "ll": ("will", "will"),
    "re": ("is", "are"), "d": ("would", "would"),
}


def _is_pronoun(text: str) -> bool:
    return text.strip().casefold() in ALL_PRONOUNS


def _is_proper(text: str) -> bool:
    """Does this mention look like a proper name (every content token capitalised)?"""
    tokens = [t for t in re.findall(r"[\w'’-]+", text) if t.casefold() not in _NAME_STOPWORDS]
    return bool(tokens) and all(t[:1].isupper() for t in tokens)


def cluster_representative(mentions: Sequence[str], *, max_words: int = 6) -> str | None:
    """Pick the string a cluster should be rewritten to, or ``None`` for no idea.

    A coreference model returns a set of co-referring spans and no opinion about
    which one is the name. Choosing badly is how a coref layer makes text worse:
    rewrite every mention to the first span and half the pronouns in a transcript
    become "it".

    The preference order is proper name, then longest non-pronoun noun phrase
    under ``max_words``. A cluster with no non-pronoun mention -- an orphan
    ``['I', 'I', 'me']`` chain, which every engine tested produces on long
    transcripts -- returns ``None`` and is left alone.
    """
    # Strip the possessive clitic BEFORE ranking. "Elena Vasquez's" and
    # "Elena Vasquez" are the same name, and ranking first means the possessive
    # form can win the tie-break and get substituted into a nominative slot --
    # "Elena Vasquez's's the consultant". Stripping first also collapses the
    # duplicate, so the tie-break never sees both.
    candidates = [
        re.sub(r"['\u2019]s?$", "", m.strip())
        for m in mentions
        if m.strip() and not _is_pronoun(m)
    ]
    candidates = [c for c in dict.fromkeys(candidates) if c]
    if not candidates:
        return None
    proper = [c for c in candidates if _is_proper(c) and len(c.split()) <= max_words]
    pool = proper or [c for c in candidates if len(c.split()) <= max_words]
    if not pool:
        return None
    # longest wins: "Order Graph Migration" over "Migration", "Priya Raman" over "Priya"
    return max(pool, key=lambda c: (len(c.split()), len(c)))


def substitute_clusters(
    text: str,
    clusters: Sequence[Sequence[tuple[int, int]]],
    *,
    resolve_definites: bool = False,
    pronouns: Iterable[str] = ALL_PRONOUNS,
    turn_of: Any = None,
) -> tuple[str, list[Rewrite]]:
    """Rewrite pronoun mentions to their cluster's representative string.

    ``clusters`` are character spans, half-open, as produced by
    :class:`FastCorefEngine` / :class:`StanzaCorefEngine`. Substitutions are
    applied right-to-left so earlier offsets stay valid.

    Possessive pronouns become ``"Name's"``. ``her`` and ``your`` are possessive
    only when a word follows them, so ``"told her"`` and ``"her team"`` come out
    differently. With ``resolve_definites=True``, definite descriptions ("the
    migration", "that repo") are rewritten too -- off by default because the
    engines put far more junk in those spans than in pronoun spans.

    Contractions are the trap. Every engine's mention span for ``"I'm"`` is the
    single character ``I``, so substituting the span alone yields
    ``"Priya Raman'm"`` -- ungrammatical input handed straight to the extractor.
    The clitic is swallowed and conjugated instead: ``I'm`` -> ``Priya Raman is``.

    ``pronouns`` restricts which pronoun classes are rewritten at all. Narrowing
    it to ``ALL_PRONOUNS - FIRST_PERSON_PLURAL`` is the interesting ablation:
    "we" almost always means the team rather than any entity in the cluster.

    ``turn_of`` is an optional callable mapping a character offset to a turn
    index, used only to label the returned :class:`Rewrite` records.
    """
    targets = {p.casefold() for p in pronouns}
    edits: list[tuple[int, int, str, str]] = []
    for cluster in clusters:
        spans = [(s, e) for s, e in cluster if 0 <= s < e <= len(text)]
        if len(spans) < 2:
            continue
        rep = cluster_representative([text[s:e] for s, e in spans])
        if rep is None:
            continue
        for s, e in spans:
            surface = text[s:e]
            low = surface.casefold()
            if "\n" in surface:          # a mention that straddles a turn boundary is a bug
                continue
            if low == rep.casefold():
                continue
            is_pron = low in targets
            is_def = bool(low.split()) and low.split()[0] in _DEFINITE_HEADS
            if not is_pron and not (resolve_definites and is_def and not _is_pronoun(surface)):
                continue
            end = e
            clitic = _CLITIC.match(text[e:e + 4]) if is_pron else None
            if clitic:
                singular, plural = _CLITIC_FORMS[clitic.group(1).casefold()]
                verb = plural if low in _PLURAL_PRONOUNS else singular
                replacement = f"{rep} {verb}"
                end = e + clitic.end()
            elif is_pron and (low in _ALWAYS_POSS or
                              (low in _MAYBE_POSS and re.match(r"\s+\w", text[e:e + 2] or ""))):
                replacement = f"{rep}'s"
            else:
                replacement = rep
            edits.append((s, end, text[s:end], replacement))

    log: list[Rewrite] = []
    out = text
    for s, e, surface, replacement in sorted(edits, key=lambda x: -x[0]):
        out = out[:s] + replacement + out[e:]
        log.append(Rewrite(turn_of(s) if turn_of else -1, "coref", surface, replacement,
                           "coreference cluster"))
    log.reverse()
    return out, log


class FastCorefEngine:
    """``fastcoref`` (MIT), either ``FCoref`` or ``LingMessCoref``.

    ``FCoref`` is a distilroberta model: ~700 MB of weights, roughly 3000 words/s
    on a laptop CPU, under 1 GB resident. ``LingMessCoref`` is longformer-large,
    six times bigger and slower, and on this repo's dialogue it is *worse* -- it
    never proposes the speaker-prefix name as a mention. Notebook 08 has the
    output.

    ``LingMessCoref`` additionally needs an eager-attention shim: transformers
    >= 4.48 defaults to sdpa and Longformer has no sdpa kernel, so construction
    raises ``ValueError``. :meth:`load` applies it.
    """

    license = "MIT"

    def __init__(self, variant: str = "fcoref", device: str = "cpu") -> None:
        self.variant = variant
        self.device = device
        self.name = variant
        self.model: Any = None

    def load(self) -> "FastCorefEngine":
        import logging

        from transformers import AutoConfig

        if not getattr(AutoConfig.from_pretrained, "_kgx_eager_shim", False):
            original = AutoConfig.from_pretrained

            @classmethod  # type: ignore[misc]
            def patched(cls, *args, **kwargs):
                kwargs.setdefault("attn_implementation", "eager")
                return original(*args, **kwargs)

            patched.__func__._kgx_eager_shim = True
            AutoConfig.from_pretrained = patched

        try:  # fastcoref pipes inputs through datasets.Dataset.map()
            import datasets

            datasets.disable_progress_bars()
        except Exception:
            pass
        logging.getLogger("fastcoref").setLevel(logging.ERROR)

        from fastcoref import FCoref, LingMessCoref

        cls = FCoref if self.variant == "fcoref" else LingMessCoref
        self.model = cls(device=self.device, enable_progress_bar=False)
        return self

    def clusters(self, text: str) -> list[list[tuple[int, int]]]:
        if self.model is None:
            self.load()
        result = self.model.predict(texts=[text])[0]
        return [[(int(s), int(e)) for s, e in c] for c in result.get_clusters(as_strings=False)]


class StanzaCorefEngine:
    """Stanza's coreference processor (Apache-2.0).

    Heavier than fastcoref -- ~2.3 GB of weights, ~5 GB resident, about 450
    words/s -- and the only engine tested that keeps one clean chain per speaker
    across a long multi-party transcript. Needs ``peft`` installed: the coref
    head is a LoRA adapter and the import fails without it.

    Stanza emits singletons; they are dropped here so the output matches the
    other engines. ``package`` selects a non-default coref model, e.g.
    ``"gum-speakers_roberta-large-lora"``.
    """

    license = "Apache-2.0"

    def __init__(self, package: str | None = None) -> None:
        self.package = package
        self.name = "stanza" if not package else f"stanza[{package.split('_')[0]}]"
        self.nlp: Any = None

    def load(self) -> "StanzaCorefEngine":
        import logging

        import stanza

        logging.getLogger("stanza").setLevel(logging.ERROR)
        kw = {"package": {"coref": self.package}} if self.package else {}
        self.nlp = stanza.Pipeline("en", processors="tokenize,coref", verbose=False, **kw)
        return self

    def clusters(self, text: str) -> list[list[tuple[int, int]]]:
        if self.nlp is None:
            self.load()
        doc = self.nlp(text)
        out: list[list[tuple[int, int]]] = []
        for chain in doc.coref:
            if len(chain.mentions) < 2:
                continue
            spans = []
            for m in chain.mentions:
                words = doc.sentences[m.sentence].words[m.start_word:m.end_word]
                if words:
                    spans.append((int(words[0].start_char), int(words[-1].end_char)))
            if len(spans) >= 2:
                out.append(spans)
        return out


def load_coref_engine(name: str = "fcoref", **kwargs: Any):
    """Instantiate one of the engines by name; the import happens here, not on import of kgx."""
    if name in {"fcoref", "lingmess"}:
        return FastCorefEngine(name, **kwargs).load()
    if name == "stanza":
        return StanzaCorefEngine(**kwargs).load()
    if name.startswith("stanza:"):
        return StanzaCorefEngine(package=name.split(":", 1)[1], **kwargs).load()
    raise ValueError(f"unknown coref engine {name!r}; try 'fcoref', 'lingmess' or 'stanza'")


class NeuralCorefPreprocessor(ConversationPreprocessor):
    """:class:`ConversationPreprocessor` with a trained coref model as a final layer.

    The rule layers run first (whatever subset ``layers`` names), the session is
    flattened to one string, the engine is run over that whole string, and every
    pronoun mention in a multi-mention cluster is rewritten to its cluster's
    representative. The turn boundaries survive because the substitution never
    crosses a newline.

    Order matters and is not negotiable: coref runs *after* the speaker layer,
    because the ``"Priya Raman: "`` prefix is the only antecedent in the text
    that a first-person pronoun can bind to. Run it on raw turns instead and the
    ``I`` chains come back orphaned, with no name in them, and
    :func:`cluster_representative` correctly refuses to rewrite them.

    Combining this with the ``first_person`` rule layer is not redundant: the
    rule is exact and the model is not, and the model's contribution is the
    third-person and second-person pronouns the rule does not touch.

    Parameters
    ----------
    engine:
        Anything with ``.clusters(text) -> [[(start, end), ...], ...]``; pass
        :func:`load_coref_engine`'s result.
    resolve_definites:
        Also rewrite "the migration" / "that repo". Off by default.
    """

    def __init__(self, user_name: str, *, engine: Any, resolve_definites: bool = False,
                 pronouns: Iterable[str] = ALL_PRONOUNS, cache: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(user_name, **kwargs)
        self.engine = engine
        self.resolve_definites = resolve_definites
        self.pronouns = frozenset(pronouns)
        # render() is called repeatedly on the same sessions -- episodes(), the
        # rewrite log, an ablation loop -- and a coref forward pass is four
        # orders of magnitude more expensive than the regex layers it sits on
        # top of. Without this, stanza re-resolves the corpus three times per
        # measurement.
        self._cache: dict[str, RenderedSession] | None = {} if cache else None

    def render(self, session: Mapping[str, Any]) -> RenderedSession:
        key = str(session.get("session_id", ""))
        if self._cache is not None and key in self._cache:
            return self._cache[key]

        base = super().render(session)
        if self.engine is None or not base.text.strip():
            return base

        line_starts: list[int] = []
        pos = 0
        for line in base.turn_texts:
            line_starts.append(pos)
            pos += len(line) + 1

        def turn_of(offset: int) -> int:
            i = 0
            for j, start in enumerate(line_starts):
                if offset >= start:
                    i = j
            return i

        clusters = self.engine.clusters(base.text)
        text, log = substitute_clusters(
            base.text, clusters, resolve_definites=self.resolve_definites,
            pronouns=self.pronouns, turn_of=turn_of,
        )
        out = RenderedSession(
            session_id=base.session_id,
            text=text,
            turn_texts=text.split("\n"),
            rewrites=[*base.rewrites, *log],
            meta=base.meta,
        )
        if self._cache is not None and key:
            self._cache[key] = out
        return out
