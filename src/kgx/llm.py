"""An LLM backend for the comparison notebooks, with no API key.

Every notebook up to now runs entirely locally. Comparing a 194M encoder against
a frontier model breaks that, and requiring an API key would make the most
important notebook in the repo the one nobody can run.

The way out is the ``claude`` CLI, which is already installed and authenticated
for anyone using Claude Code. ``claude -p`` is a non-interactive completion:
prompt on stdin, JSON on stdout, cost reported per call. It is a subprocess, not
an SDK, so it is slower and chattier than the real API -- but it is *real model
output*, which a stub is not, and the comparison is worthless without it.

Two design choices worth stating:

**Every response is cached to disk**, content-addressed on
``(model, system_prompt, prompt)``. A notebook re-run costs nothing and returns
byte-identical results, which is what makes an LLM-vs-encoder benchmark
reproducible rather than re-billed and slightly different every time. Delete the
cache directory to force a genuine re-run.

**Extraction returns the same** :class:`~kgx.extract.DocGraph` **as GLiNER.**
The whole point is a fair comparison, so the two extractors must be
interchangeable everywhere downstream -- resolution, graph building, scoring.
The LLM is prompted with the same :class:`~kgx.ontology.Ontology` object that
compiles to GLiNER's schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .extract import DocGraph, Edge, Mention, _context_window
from .ontology import Ontology

__all__ = [
    "LLMResult",
    "LLMUnavailable",
    "ClaudeCLI",
    "parse_json_response",
    "LLMExtractor",
    "EXTRACTION_SYSTEM_PROMPT",
]

EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise information extraction service. "
    "You respond with valid JSON and nothing else: no prose, no explanation, "
    "no markdown code fences."
)


class LLMUnavailable(RuntimeError):
    """The ``claude`` CLI is not installed, not authenticated, or failed."""


@dataclass
class LLMResult:
    text: str
    model: str
    cost_usd: float = 0.0
    duration_ms: int = 0
    cached: bool = False

    def json(self) -> Any:
        return parse_json_response(self.text)


def parse_json_response(text: str) -> Any:
    """Pull JSON out of a model response.

    Small models fence their output in ```json blocks even when told not to;
    larger ones usually do not, and either may add a sentence in front. Rather
    than trusting the instruction, strip fences, then fall back to the first
    balanced ``{...}`` or ``[...]`` in the string.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth, in_string, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"no JSON found in response: {text[:200]!r}")


class ClaudeCLI:
    """A cached, subprocess-backed LLM.

    Parameters
    ----------
    model:
        ``"haiku"``, ``"sonnet"`` or ``"opus"``. Haiku is the sensible default
        for bulk extraction -- roughly a cent a call here, and the comparison is
        about the *class* of model, not about squeezing the last point of F1.
    cache_dir:
        Where responses are stored. Delete it to force real calls.
    """

    def __init__(
        self,
        model: str = "haiku",
        *,
        system_prompt: str | None = EXTRACTION_SYSTEM_PROMPT,
        cache_dir: str | Path = "output/llm_cache",
        timeout: int = 300,
        binary: str = "claude",
        json_schema: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.json_schema = json_schema
        self.system_prompt = system_prompt
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.binary = binary
        self.total_cost_usd = 0.0
        self.cached_cost_usd = 0.0   # what the cached responses cost when they were first made
        self.n_calls = 0
        self.n_cached = 0
        self.total_ms = 0            # live calls only -- wall time actually spent
        self.observed_ms = 0         # live + cached -- what the latency WAS
        self.n_observed = 0

    # -- availability ----------------------------------------------------

    @property
    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def check(self) -> str:
        """Raise unless the CLI is usable; return its version."""
        if not self.available:
            raise LLMUnavailable(
                f"{self.binary!r} not found on PATH. Install Claude Code, or pass "
                f"an already-cached cache_dir to replay a previous run."
            )
        out = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip()

    # -- caching ---------------------------------------------------------

    def _cache_path(self, prompt: str) -> Path:
        key = hashlib.sha256(
            json.dumps([self.model, self.system_prompt, prompt], sort_keys=True).encode()
        ).hexdigest()[:32]
        return self.cache_dir / f"{key}.json"

    # -- completion ------------------------------------------------------

    def complete(self, prompt: str, *, refresh: bool = False) -> LLMResult:
        """One completion, cached on disk."""
        path = self._cache_path(prompt)
        if path.exists() and not refresh:
            blob = json.loads(path.read_text())
            self.n_cached += 1
            # A cached response still records what the call originally cost and
            # how long it took. Tracking that separately from live spend lets a
            # fully-cached notebook report real latency and real cost instead of
            # NaN and zero, while `total_cost_usd` stays an honest record of what
            # THIS run actually billed.
            self.cached_cost_usd += float(blob.get("cost_usd", 0.0))
            self.observed_ms += int(blob.get("duration_ms", 0))
            self.n_observed += 1
            return LLMResult(blob["text"], blob["model"], blob.get("cost_usd", 0.0),
                             blob.get("duration_ms", 0), cached=True)

        if not self.available:
            raise LLMUnavailable(
                f"{self.binary!r} not found and no cached response for this prompt "
                f"({path.name}). Nothing to replay."
            )

        # Three flags do almost all the work of making this affordable, because
        # by default `claude -p` ships the entire Claude Code agent preamble --
        # tool definitions, CLAUDE.md, skills, MCP config -- as prompt on every
        # single call. Measured on one trivial completion:
        #
        #     no flags                          ~18,000 prompt tokens   $0.0187
        #     --system-prompt + --safe-mode     ~18,500 tokens          $0.0382
        #     --tools "" + --safe-mode           ~3,600 tokens          $0.0048
        #     all three                             254 tokens          $0.0003
        #
        # --system-prompt REPLACES the preamble (--append-system-prompt does not),
        # --tools "" removes the tool definitions and the agent loop, and
        # --safe-mode drops project config. MAX_THINKING_TOKENS=0 stops Haiku
        # spending ~1,150 thinking tokens per document on an extraction task.
        argv = [self.binary, "-p", "--model", self.model, "--output-format", "json",
                "--tools", "", "--safe-mode"]
        if self.system_prompt:
            argv += ["--system-prompt", self.system_prompt]
        if self.json_schema is not None:
            argv += ["--json-schema", json.dumps(self.json_schema)]

        env = {**os.environ, "MAX_THINKING_TOKENS": "0"}
        started = time.time()
        try:
            proc = subprocess.run(argv, input=prompt, capture_output=True,
                                  text=True, timeout=self.timeout, env=env)
        except subprocess.TimeoutExpired as exc:
            raise LLMUnavailable(f"claude timed out after {self.timeout}s") from exc
        if proc.returncode != 0:
            raise LLMUnavailable(f"claude exited {proc.returncode}: {proc.stderr[:400]}")

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"unparseable CLI output: {proc.stdout[:300]!r}") from exc
        if payload.get("is_error"):
            raise LLMUnavailable(f"claude reported an error: {payload.get('result', '')[:300]}")

        if not isinstance(payload.get("result"), str):
            raise LLMUnavailable(
                f"claude returned no text (result={payload.get('result')!r}); "
                f"stop_reason={payload.get('stop_reason')!r}"
            )
        result = LLMResult(
            text=payload["result"],
            model=self.model,
            cost_usd=float(payload.get("total_cost_usd", 0.0)),
            duration_ms=int(payload.get("duration_ms", (time.time() - started) * 1000)),
        )
        # `model` is the alias asked for ("haiku"); `resolved_model` is what the
        # CLI actually served. The key deliberately does NOT include the resolved
        # id -- that would invalidate the cache every time the alias is repointed
        # -- but recording it gives a replayed response provenance. Without it a
        # regenerated cache silently changes results with nothing to compare.
        resolved = None
        usage = payload.get("modelUsage")
        if isinstance(usage, dict) and usage:
            resolved = next(iter(usage))
        path.write_text(json.dumps({
            "text": result.text, "model": result.model,
            "resolved_model": resolved,
            "cost_usd": result.cost_usd, "duration_ms": result.duration_ms,
        }, indent=1))
        self.n_calls += 1
        self.total_cost_usd += result.cost_usd
        self.total_ms += result.duration_ms
        self.observed_ms += result.duration_ms
        self.n_observed += 1
        return result

    def complete_json(self, prompt: str, *, refresh: bool = False, retries: int = 1) -> Any:
        """Complete and parse JSON, retrying once on a malformed response."""
        last: Exception | None = None
        for attempt in range(retries + 1):
            result = self.complete(prompt, refresh=refresh or attempt > 0)
            try:
                return result.json()
            except ValueError as exc:
                last = exc
        raise ValueError(f"model did not return usable JSON after {retries + 1} attempts") from last

    def batch(self, prompts: Sequence[str], *, workers: int = 4) -> list[LLMResult]:
        """Complete several prompts concurrently. Cached entries cost nothing."""
        from concurrent.futures import ThreadPoolExecutor

        if workers <= 1:
            return [self.complete(p) for p in prompts]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.complete, prompts))

    @property
    def mean_latency_ms(self) -> float:
        """Mean per-call latency across live *and* cached calls.

        Reported from the cache too, so the number survives a re-run.
        """
        return self.observed_ms / self.n_observed if self.n_observed else float("nan")

    @property
    def observed_cost_usd(self) -> float:
        """What every call cost when first made, live or replayed from cache.

        Distinct from :attr:`total_cost_usd`, which is what *this* run billed --
        zero on a warm cache. Use this one for "what does extraction cost per
        document"; use the other for "what did I just spend".
        """
        return self.total_cost_usd + self.cached_cost_usd

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "live_calls": self.n_calls,
            "cache_hits": self.n_cached,
            "cost_usd_this_run": round(self.total_cost_usd, 4),
            "cost_usd_observed": round(self.observed_cost_usd, 4),
            "live_seconds": round(self.total_ms / 1000, 1),
            "mean_latency_ms": round(self.mean_latency_ms) if self.n_observed else None,
        }


# ---------------------------------------------------------------------------
# ontology-driven extraction
# ---------------------------------------------------------------------------

def _ontology_prompt(text: str, ontology: Ontology) -> str:
    entity_lines = "\n".join(f"  - {e.name}: {e.description}" for e in ontology.entities)
    relation_lines = "\n".join(
        f"  - {r.name}({'|'.join(r.head)} -> {'|'.join(r.tail)}): {r.description}"
        for r in ontology.relations
    )
    return f"""Extract a knowledge graph from the text below, using ONLY the entity types and relation types defined in this ontology.

ENTITY TYPES:
{entity_lines}

RELATION TYPES (head_types -> tail_types):
{relation_lines}

RULES:
- Use only the listed types. Do not invent new ones.
- Every relation's head and tail must be entities you also list, and their types must satisfy the relation's declared head/tail types.
- Extract the exact surface string as it appears in the text for each entity.
- Extract only what the text states or directly implies. Do not add outside knowledge.
- If the text states a fact that requires one step of inference to see (e.g. "she was promoted to lead the team" implies she leads the team), include it.

Respond with JSON of exactly this shape and nothing else:
{{
  "entities": [{{"text": "<exact surface string>", "type": "<entity type>"}}],
  "relations": [{{"head": "<entity surface string>", "type": "<relation type>", "tail": "<entity surface string>"}}]
}}

TEXT:
{text}"""


class LLMExtractor:
    """An LLM extractor that returns the same :class:`DocGraph` as GLiNER.

    Interchangeable with :class:`~kgx.extract.GlinerExtractor` everywhere
    downstream, which is the only way the comparison means anything.
    """

    def __init__(self, llm: ClaudeCLI | None = None, *, context_width: int = 90) -> None:
        self.llm = llm or ClaudeCLI()
        self.context_width = context_width

    def _to_doc_graph(self, payload: Mapping[str, Any], text: str, doc_id: str,
                      ontology: Ontology, elapsed: float, meta: dict[str, Any]) -> DocGraph:
        valid_types = set(ontology.entity_names)
        mentions: list[Mention] = []
        by_surface: dict[str, str] = {}

        for i, item in enumerate(payload.get("entities") or [], start=1):
            surface = str(item.get("text", "")).strip()
            etype = str(item.get("type", "")).strip()
            if not surface or etype not in valid_types:
                continue
            start = text.find(surface)
            end = start + len(surface) if start >= 0 else -1
            mid = f"{doc_id}:e{i}"
            mentions.append(Mention(
                mention_id=mid, doc_id=doc_id, type=etype, text=surface,
                start=max(start, 0), end=max(end, 0), confidence=1.0,
                context=_context_window(text, max(start, 0), max(end, 0), self.context_width)
                if start >= 0 else "",
            ))
            by_surface.setdefault(surface.casefold(), mid)

        edges: list[Edge] = []
        seen: set[tuple[str, str, str]] = set()
        for item in payload.get("relations") or []:
            rel = str(item.get("type", "")).strip()
            head = by_surface.get(str(item.get("head", "")).strip().casefold())
            tail = by_surface.get(str(item.get("tail", "")).strip().casefold())
            if not head or not tail or head == tail:
                continue
            try:
                ontology.relation(rel)
            except KeyError:
                continue
            key = (head, rel, tail)
            if key in seen:
                continue
            seen.add(key)
            edges.append(Edge(doc_id=doc_id, type=rel, head=head, tail=tail, confidence=1.0))

        return DocGraph(doc_id=doc_id, text=text, mentions=mentions, edges=edges,
                        feasible=True, elapsed_s=elapsed, meta=meta)

    def extract(self, text: str, ontology: Ontology, doc_id: str = "doc", **meta: Any) -> DocGraph:
        started = time.time()
        payload = self.llm.complete_json(_ontology_prompt(text, ontology))
        return self._to_doc_graph(payload, text, doc_id, ontology,
                                  time.time() - started, dict(meta))

    def extract_batch(
        self,
        docs: Sequence[Mapping[str, Any]],
        ontology: Ontology,
        *,
        text_key: str = "text",
        id_key: str = "doc_id",
        workers: int = 4,
    ) -> list[DocGraph]:
        prompts = [_ontology_prompt(d[text_key], ontology) for d in docs]
        started = time.time()
        results = self.llm.batch(prompts, workers=workers)
        per_doc = (time.time() - started) / max(len(docs), 1)
        graphs = []
        for i, (doc, result) in enumerate(zip(docs, results)):
            try:
                payload = result.json()
            except ValueError:
                payload = {"entities": [], "relations": []}
            graphs.append(self._to_doc_graph(
                payload, doc[text_key], doc.get(id_key, f"doc{i}"), ontology, per_doc,
                {k: v for k, v in doc.items() if k not in {text_key, id_key}},
            ))
        return graphs
