"""Running three third-party KG-construction frameworks on one LLM and one ontology.

``neo4j-graphrag``'s ``SimpleKGPipeline``, LangChain's ``LLMGraphTransformer`` and
LlamaIndex's ``PropertyGraphIndex`` all do the same job: prompt a model with a
schema, parse a graph out of the reply, write it to Neo4j. Comparing them
requires holding three things constant that they each want to own.

**The model.** None of them ships an adapter for a subprocess CLI, and all three
type-check the object you hand them, so each needs a small subclass of a
different base class. :func:`graphrag_llm`, :func:`langchain_llm` and
:func:`llamaindex_llm` build those three wrappers around one
:class:`~kgx.llm.ClaudeCLI`. Crucially none of them needs function calling: each
framework has a prompt-and-parse-JSON fallback, and the wrapper's job is partly
to make sure that fallback is the path taken.

**The ontology.** Each framework has its own opinion about label casing --
``Company`` for neo4j-graphrag, whatever the model emits for LangChain,
``COMPANY`` (mandatory, see :func:`schema_dialects`) for LlamaIndex. The same
:class:`~kgx.ontology.Ontology` compiles to all three dialects here.

**The scoring.** Each framework writes a *different* schema into Neo4j -- a
different key property, different lexical nodes, different secondary labels. The
comparison is only worth as much as the function that reads those three graphs
back into one comparable shape, which is :func:`read_back`. It is deliberately
blunt: keep every node that carries an ontology label, keep every edge between
two such nodes whose type maps to an ontology relation, and count everything it
throws away so the discards can be audited rather than assumed harmless.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .llm import ClaudeCLI
from .ontology import Ontology

__all__ = [
    "CallLog",
    "run_coroutine",
    "run_off_loop",
    "LEXICAL_LABELS",
    "LEXICAL_RELATIONSHIPS",
    "label_key",
    "canonical_type",
    "canonical_relation",
    "schema_dialects",
    "stabilise_prompt",
    "graphrag_llm",
    "langchain_llm",
    "llamaindex_llm",
    "FrameworkOutput",
    "read_back",
    "wipe",
    "duplication_report",
    "resolve_names",
]


# Edges every framework writes to tie entities back to the text they came from.
# They are provenance, not knowledge, and none of them is in any ontology.
LEXICAL_RELATIONSHIPS = frozenset({
    "FROM_CHUNK", "FROM_DOCUMENT", "NEXT_CHUNK", "MENTIONS", "MENTIONED_IN",
    "SOURCE", "PART_OF", "HAS_SOURCE",
})

# Node labels for the *lexical* graph -- the documents and chunks the entities
# were extracted from. Not ontology types, and not the framework inventing a
# type either, so they are excluded from the violation count rather than
# charged against it. Anything beginning with ``__`` is a framework's internal
# marker label (``__Entity__``, ``__Node__``, ``__KGBuilder__``).
LEXICAL_LABELS = frozenset({"Document", "Chunk", "Node", "TextChunk", "Entity"})


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------

class CallLog:
    """A counting proxy in front of :class:`~kgx.llm.ClaudeCLI`.

    Wall-clock time is a bad measure of an LLM pipeline that runs against a
    disk cache -- the second run of this notebook would report every framework
    as infinitely fast. Every :class:`~kgx.llm.LLMResult` carries the cost and
    duration of the call *when it was really made*, cache hit or not, so the log
    records those alongside the observed wall time and both can be reported.
    """

    def __init__(self, llm: ClaudeCLI) -> None:
        self.llm = llm
        self.calls: list[dict[str, Any]] = []

    def reset(self) -> "CallLog":
        self.calls = []
        return self

    def _record(self, prompt: str, result: Any) -> Any:
        self.calls.append({
            "prompt_chars": len(prompt),
            "reply_chars": len(result.text),
            "cost_usd": result.cost_usd,
            "model_ms": result.duration_ms,
            "cached": result.cached,
        })
        return result

    def complete(self, prompt: str) -> str:
        return self._record(prompt, self.llm.complete(prompt)).text

    # -- ClaudeCLI-compatible surface, so this can also stand in front of
    #    kgx.llm.LLMExtractor and account for the in-house baseline the same way.

    def complete_json(self, prompt: str, **kwargs: Any) -> Any:
        from .llm import parse_json_response

        return parse_json_response(self.complete(prompt))

    def batch(self, prompts: Sequence[str], *, workers: int = 4) -> list[Any]:
        from concurrent.futures import ThreadPoolExecutor

        def one(prompt: str) -> Any:
            return self._record(prompt, self.llm.complete(prompt))

        if workers <= 1:
            return [one(p) for p in prompts]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, prompts))

    def summary(self) -> dict[str, Any]:
        """Totals. ``model_seconds`` is the sum of the *original* call durations."""
        return {
            "llm_calls": len(self.calls),
            "cache_hits": sum(1 for c in self.calls if c["cached"]),
            "prompt_chars": sum(c["prompt_chars"] for c in self.calls),
            "reply_chars": sum(c["reply_chars"] for c in self.calls),
            "cost_usd": round(sum(c["cost_usd"] for c in self.calls), 4),
            "model_seconds": round(sum(c["model_ms"] for c in self.calls) / 1000, 1),
        }


# ---------------------------------------------------------------------------
# label dialects
# ---------------------------------------------------------------------------

def run_off_loop(fn):
    """Call ``fn()`` on a worker thread that has no running event loop.

    Two of the three frameworks call ``asyncio.run`` internally --
    ``PropertyGraphIndex._insert_nodes`` does it unconditionally when
    ``use_async`` is on -- and ``asyncio.run`` raises ``RuntimeError:
    asyncio.run() cannot be called from a running event loop`` inside a Jupyter
    kernel, which always has one. A fresh worker thread does not, so the same
    call works there in both a notebook and a script.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn).result()


def run_coroutine(factory):
    """Run an async factory to completion from sync code, notebook or script.

    ``SimpleKGPipeline`` exposes only ``run_async``, and top-level ``await``
    works in Jupyter but not in a script. Running ``asyncio.run`` on a worker
    thread (see :func:`run_off_loop`) works in both -- and the coroutine has to
    be *created* inside that thread, hence the factory rather than a coroutine.
    """
    import asyncio

    return run_off_loop(lambda: asyncio.run(factory()))


def label_key(label: str) -> str:
    """Casing- and separator-insensitive key for a label or relationship type.

    ``business_segment``, ``BusinessSegment``, ``BUSINESS_SEGMENT`` and
    ``Business segment`` all collapse to ``businesssegment``. Every framework
    invents its own casing convention and two of them rewrite the model's answer
    before storing it; matching on this key is what makes the read-back
    framework-agnostic without hand-maintaining three lookup tables.
    """
    return re.sub(r"[^0-9a-z]+", "", label.lower())


def canonical_type(label: str, ontology: Ontology) -> str | None:
    """Map a stored node label back to an ontology entity type, or ``None``."""
    key = label_key(label)
    for name in ontology.entity_names:
        if label_key(name) == key:
            return name
    return None


def canonical_relation(rel_type: str, ontology: Ontology) -> str | None:
    """Map a stored relationship type back to an ontology relation, or ``None``."""
    key = label_key(rel_type)
    for name in ontology.relation_names:
        if label_key(name) == key:
            return name
    return None


def _pascal(name: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[^0-9A-Za-z]+", name) if p)


def schema_dialects(ontology: Ontology) -> dict[str, Any]:
    """One ontology, three schema encodings.

    ``graphrag``
        ``{"node_types": [...], "relationship_types": [...], "patterns": [...]}``
        with ``PascalCase`` labels and ``UPPER_SNAKE`` relations.
    ``langchain``
        ``allowed_nodes`` (``PascalCase``) plus ``allowed_relationships`` as
        ``(head, REL, tail)`` 3-tuples, which is what makes ``strict_mode``
        filter on direction rather than on relation name alone.
    ``llamaindex``
        ``UPPER_SNAKE`` for *both*, which is not a style choice --
        ``SchemaLLMPathExtractor`` uppercases every type the model returns
        before validating it against your ``Literal``, so a ``PascalCase``
        entity list silently matches nothing.
    """
    ents = [(e.name, e.description) for e in ontology.entities]
    rels = list(ontology.relations)
    patterns = sorted({(h, r.name, t) for r in rels for h in r.head for t in r.tail})

    return {
        "graphrag": {
            "node_types": [
                {"label": _pascal(n), "description": d,
                 "properties": [{"name": "name", "type": "STRING"}]}
                for n, d in ents
            ],
            "relationship_types": [
                {"label": r.name.upper(), "description": r.description} for r in rels
            ],
            "patterns": [(_pascal(h), r.upper(), _pascal(t)) for h, r, t in patterns],
        },
        "langchain": {
            "allowed_nodes": [_pascal(n) for n, _ in ents],
            "allowed_relationships": [(_pascal(h), r.upper(), _pascal(t))
                                      for h, r, t in patterns],
        },
        "llamaindex": {
            "entities": [n.upper() for n, _ in ents],
            "relations": [r.name.upper() for r in rels],
            "validation_schema": [(h.upper(), r.upper(), t.upper())
                                  for h, r, t in patterns],
        },
        "patterns": patterns,
    }


# ---------------------------------------------------------------------------
# one LLM, three interfaces
# ---------------------------------------------------------------------------

def graphrag_llm(log: CallLog, model_name: str = "claude-cli-haiku"):
    """A ``neo4j_graphrag.llm.LLMBase`` backed by :class:`CallLog`.

    ``supports_structured_output = False`` is the load-bearing line: the builder
    reads it to decide between the structured-output extractor and the
    prompt-and-parse one, and only the second works without function calling.
    ``ainvoke`` hands the blocking subprocess to a thread so that the
    extractor's ``max_concurrency=5`` actually buys concurrency.
    """
    import asyncio

    from neo4j_graphrag.llm import LLMBase, LLMResponse

    def flatten(value: Any, system_instruction: str | None = None) -> str:
        if isinstance(value, str):
            body = value
        elif isinstance(value, list):
            body = "\n".join(f"{m['role']}: {m['content']}" if isinstance(m, dict)
                             else str(m) for m in value)
        else:
            body = str(value)
        return f"{system_instruction}\n\n{body}" if system_instruction else body

    class ClaudeCLILLM(LLMBase):
        supports_structured_output: bool = False

        def invoke(self, input, message_history=None, system_instruction=None,
                   response_format=None, **kwargs) -> "LLMResponse":
            return LLMResponse(content=log.complete(flatten(input, system_instruction)))

        async def ainvoke(self, input, message_history=None, system_instruction=None,
                          response_format=None, **kwargs) -> "LLMResponse":
            text = await asyncio.to_thread(log.complete, flatten(input, system_instruction))
            return LLMResponse(content=text)

    return ClaudeCLILLM(model_name=model_name)


_UPPER_LIST = re.compile(r"\['[A-Z][A-Z0-9_]*'(?:, '[A-Z][A-Z0-9_]*')*\]")


def stabilise_prompt(prompt: str, order: str = "sorted") -> str:
    """Sort any ``['UPPER_SNAKE', ...]`` list literal inside a rendered prompt.

    ``langchain_neo4j.graph_transformers.llm.create_unstructured_prompt`` builds
    the allowed-relationship line as ``str(list({item[1] for item in rel_types}))``
    -- a **set**, rendered directly into the prompt. Python randomises string
    hashing per process, so the prompt text, the model's answer and any
    content-addressed cache key all change from one run to the next.

    Sorting the enumeration back into a stable order does not change what the
    prompt asks for, and it is the difference between a benchmark that replays
    from cache and one that re-bills and re-scores on every execution.

    ``order="reversed"`` gives a second, equally stable ordering. It exists so
    that the cost of the reordering can be *measured* rather than asserted.
    """
    def rewrite(match: "re.Match[str]") -> str:
        names = sorted(match.group(0)[1:-1].split(", "), reverse=(order == "reversed"))
        return "[" + ", ".join(names) + "]"

    return _UPPER_LIST.sub(rewrite, prompt)


def langchain_llm(log: CallLog, model_name: str = "claude-cli-haiku",
                  *, stabilise: bool = True, order: str = "sorted"):
    """A ``langchain_core`` ``BaseChatModel`` backed by :class:`CallLog`.

    Deliberately does **not** override ``bind_tools``. ``BaseChatModel``'s
    ``with_structured_output`` raises ``NotImplementedError`` when it sees the
    base implementation still in place, and ``LLMGraphTransformer`` catches that
    to fall back to its JSON-prompt path. Implementing ``bind_tools`` here would
    silently move the whole pipeline onto a code path a CLI cannot serve.
    """
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ClaudeCLIChat(BaseChatModel):
        model: str = model_name

        @property
        def _llm_type(self) -> str:
            return "claude-cli"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> "ChatResult":
            prompt = "\n\n".join(f"[{m.type}]\n{m.content}" for m in messages)
            if stabilise:
                prompt = stabilise_prompt(prompt, order)
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(content=log.complete(prompt)))
            ])

    return ClaudeCLIChat()


def llamaindex_llm(log: CallLog, model_name: str = "claude-cli-haiku"):
    """A ``llama_index.core.llms.CustomLLM`` backed by :class:`CallLog`.

    ``is_function_calling_model=False`` routes ``astructured_predict`` through
    ``LLMTextCompletionProgram`` -- the JSON schema is appended to the prompt as
    text and the reply is parsed -- instead of ``FunctionCallingProgram``.
    """
    from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
    from llama_index.core.llms.callbacks import llm_completion_callback

    class ClaudeCLILLM(CustomLLM):
        model: str = model_name

        @property
        def metadata(self) -> "LLMMetadata":
            return LLMMetadata(
                context_window=200_000, num_output=8192, model_name=self.model,
                is_chat_model=True, is_function_calling_model=False,
            )

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs):
            return CompletionResponse(text=log.complete(prompt))

        @llm_completion_callback()
        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs):
            text = log.complete(prompt)
            yield CompletionResponse(text=text, delta=text)

    return ClaudeCLILLM()


# ---------------------------------------------------------------------------
# reading three different schemas back into one shape
# ---------------------------------------------------------------------------

@dataclass
class FrameworkOutput:
    """What one framework left in Neo4j, normalised to ontology terms."""

    framework: str
    n_nodes: int = 0
    n_relationships: int = 0
    node_labels: dict[str, int] = field(default_factory=dict)
    relationship_types: dict[str, int] = field(default_factory=dict)
    entities: list[tuple[str, str]] = field(default_factory=list)     # (name, ontology type)
    typed_triples: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    dropped_lexical: int = 0
    dropped_unmapped_rel: dict[str, int] = field(default_factory=dict)
    dropped_unmapped_node: dict[str, int] = field(default_factory=dict)
    lexical_labels: dict[str, int] = field(default_factory=dict)
    unnamed_entities: int = 0
    wall_seconds: float = 0.0
    llm: dict[str, Any] = field(default_factory=dict)

    @property
    def triples(self) -> list[tuple[str, str, str]]:
        return [(h, r, t) for h, _, r, t, _ in self.typed_triples]

    def illegal_edges(self, ontology: Ontology) -> list[tuple[str, str, str, str, str]]:
        """Edges whose relation exists but whose endpoint types are not legal."""
        return [t for t in self.typed_triples if not ontology.permits(t[1], t[2], t[4])]

    def canonicalised(self, mapping: dict[str, str]) -> list[tuple[str, str, str]]:
        """Triples with every name replaced by ``mapping``'s canonical form."""
        return [(mapping.get(h, h), r, mapping.get(t, t)) for h, _, r, t, _ in self.typed_triples]

    def violations(self, ontology: Ontology) -> dict[str, int]:
        """Ontology breaches, split by kind.

        ``off_ontology_labels``
            node labels the framework wrote that are not in the ontology at all.
        ``off_ontology_relations``
            relationship types not in the ontology.
        ``illegal_endpoints``
            edges whose relation exists but whose head/tail types are not a
            legal pattern for it -- ``(person)-[:PRODUCES]->(product)``.
        """
        illegal = sum(
            1 for _, ht, r, _, tt in self.typed_triples
            if not ontology.permits(ht, r, tt)
        )
        return {
            "off_ontology_labels": len(self.dropped_unmapped_node),
            "off_ontology_relations": len(self.dropped_unmapped_rel),
            "illegal_endpoints": illegal,
        }


def read_back(
    driver,
    ontology: Ontology,
    framework: str,
    *,
    database: str = "neo4j",
    lexical_relationships: Iterable[str] = LEXICAL_RELATIONSHIPS,
) -> FrameworkOutput:
    """Read whatever is in the database into a :class:`FrameworkOutput`.

    The rules, in order, because an unfair normalisation invalidates the whole
    comparison:

    1. A node is an **entity** if any of its labels maps to an ontology entity
       type by :func:`label_key`. That covers ``Company`` / ``COMPANY`` /
       ``company`` without a per-framework table, and it excludes ``Document``,
       ``Chunk``, ``__Entity__``, ``__Node__`` and ``__KGBuilder__`` -- none of
       which is an ontology type -- without naming them.
    2. Its **name** is ``n.name`` if present, else ``n.id``. neo4j-graphrag and
       LlamaIndex key on ``name``; LangChain keys on ``id``.
    3. An edge counts if both endpoints are entities *and* its type maps to an
       ontology relation. Provenance edges are counted separately, not silently
       dropped.

    No entity resolution, normalisation or deduplication happens here. Names go
    to the scorer exactly as the framework stored them.
    """
    import neo4j

    lexical = {t.upper() for t in lexical_relationships}
    out = FrameworkOutput(framework=framework)

    records, _, _ = driver.execute_query(
        "MATCH (n) RETURN labels(n) AS labels, count(*) AS n",
        database_=database, routing_=neo4j.RoutingControl.READ)
    for r in records:
        for label in r["labels"]:
            out.node_labels[label] = out.node_labels.get(label, 0) + r["n"]
        out.n_nodes += r["n"]
        if not any(canonical_type(l, ontology) for l in r["labels"]):
            for label in r["labels"]:
                if label.startswith("__"):
                    continue
                bucket = (out.lexical_labels if label in LEXICAL_LABELS
                          else out.dropped_unmapped_node)
                bucket[label] = bucket.get(label, 0) + r["n"]

    records, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n",
        database_=database, routing_=neo4j.RoutingControl.READ)
    for r in records:
        out.relationship_types[r["t"]] = r["n"]
        out.n_relationships += r["n"]

    # Named properties only -- `properties(n)` would drag chunk embeddings back
    # over the wire for every row.
    records, _, _ = driver.execute_query(
        "MATCH (n) RETURN labels(n) AS labels, coalesce(n.name, n.id) AS name",
        database_=database, routing_=neo4j.RoutingControl.READ)
    for r in records:
        etype = next((t for t in (canonical_type(l, ontology) for l in r["labels"]) if t), None)
        if etype is None:
            continue
        if not r["name"]:
            out.unnamed_entities += 1
            continue
        out.entities.append((r["name"], etype))

    records, _, _ = driver.execute_query(
        """
        MATCH (a)-[r]->(b)
        RETURN labels(a) AS hl, coalesce(a.name, a.id) AS hn,
               type(r) AS rel,
               labels(b) AS tl, coalesce(b.name, b.id) AS tn
        """,
        database_=database, routing_=neo4j.RoutingControl.READ)
    for r in records:
        rel_type = r["rel"]
        if rel_type.upper() in lexical:
            out.dropped_lexical += 1
            continue
        htype = next((t for t in (canonical_type(l, ontology) for l in r["hl"]) if t), None)
        ttype = next((t for t in (canonical_type(l, ontology) for l in r["tl"]) if t), None)
        if htype is None or ttype is None or not r["hn"] or not r["tn"]:
            out.dropped_lexical += 1
            continue
        rel = canonical_relation(rel_type, ontology)
        if rel is None:
            out.dropped_unmapped_rel[rel_type] = out.dropped_unmapped_rel.get(rel_type, 0) + 1
            continue
        out.typed_triples.append((r["hn"], htype, rel, r["tn"], ttype))

    return out


def wipe(driver, database: str = "neo4j") -> dict[str, Any]:
    """Delete every node, index and constraint in the database.

    Neo4j Community has exactly one database, so three frameworks that all write
    ``__Entity__`` cannot be kept apart by anything except time.
    neo4j-graphrag's entity resolver in particular runs
    ``MATCH (entity:__Entity__)`` with no scoping of any kind, so leaving a
    previous framework's nodes in place would let it merge them.
    """
    import neo4j

    from .neo4j_io import clear_database

    deleted = clear_database(driver, database)

    records, _, _ = driver.execute_query(
        "SHOW CONSTRAINTS YIELD name RETURN name", database_=database)
    constraints = [r["name"] for r in records]
    for name in constraints:
        driver.execute_query(f"DROP CONSTRAINT {name} IF EXISTS", database_=database)

    records, _, _ = driver.execute_query(
        "SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name",
        database_=database)
    indexes = [r["name"] for r in records]
    for name in indexes:
        driver.execute_query(f"DROP INDEX {name} IF EXISTS", database_=database)

    return {"nodes_deleted": deleted, "constraints_dropped": len(constraints),
            "indexes_dropped": len(indexes)}


# ---------------------------------------------------------------------------
# entity resolution, or the lack of it
# ---------------------------------------------------------------------------

def duplication_report(
    entities: Sequence[tuple[str, str]],
    aliases: dict[str, Sequence[str]],
) -> dict[str, Any]:
    """How many stored nodes refer to the same real entity, per the alias table.

    The corpus ships a hand-written clustering of every surface form onto a
    canonical name. Two stored nodes that fall in the same alias group are two
    nodes for one thing -- which is exactly what an entity-resolution pass would
    have collapsed and what none of these frameworks does.
    """
    from .resolve import normalize

    lookup: dict[str, str] = {}
    for canonical, variants in aliases.items():
        for variant in [canonical, *variants]:
            lookup[normalize(variant).key] = canonical

    groups: dict[str, set[str]] = {}
    for name, _type in entities:
        canonical = lookup.get(normalize(name).key)
        if canonical is None:
            continue
        groups.setdefault(canonical, set()).add(name)

    split = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    covered = sum(len(v) for v in groups.values())
    return {
        "nodes": len(entities),
        "distinct_names": len({n for n, _ in entities}),
        "nodes_in_alias_table": covered,
        "real_entities_covered": len(groups),
        "split_entities": len(split),
        "surplus_nodes": covered - len(groups),
        "detail": split,
    }


def resolve_names(
    entities: Sequence[tuple[str, str]],
    *,
    corpus_texts: Iterable[str] = (),
    threshold: float = 0.90,
) -> dict[str, str]:
    """Run this repo's resolver over a framework's node names.

    Returns ``{stored name: canonical name}``. Synthesising one
    :class:`~kgx.extract.Mention` per stored node is legitimate here because the
    resolver only ever reads ``text`` and ``type`` off a mention for scoring
    (plus ``context``, which these frameworks do not keep) -- the point is to
    measure what the missing stage would have been worth, on the frameworks'
    own output.
    """
    from .extract import Mention
    from .resolve import EntityResolver

    mentions = [
        Mention(mention_id=f"m{i}", doc_id="corpus", type=etype, text=name,
                start=0, end=len(name), confidence=1.0, context="")
        for i, (name, etype) in enumerate(entities)
    ]
    resolver = EntityResolver(threshold=threshold)
    if corpus_texts:
        resolver.learn_aliases(corpus_texts)
    resolution = resolver.resolve(mentions)
    return {
        m.text: resolution.entities[resolution.canon_for(m.mention_id)].canonical
        for m in mentions
    }
