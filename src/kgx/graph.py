"""Assembling one knowledge graph from many document-local graphs.

Rewrites every edge endpoint from a document-local mention id to a canonical
entity id, collapses the duplicates that fall out of that, and keeps the evidence
that justified each edge.

The evidence is the point. An edge that says *Northwind acquires Cascade* with
no pointer back to the sentence it came from cannot be audited, corrected, or
trusted, and a graph built from a 194M-parameter model will contain wrong edges.
Every :class:`GraphEdge` carries the document ids, character offsets and per-
occurrence confidences that produced it, so any claim in the graph can be traced
to text in one hop.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .extract import DocGraph
from .ontology import Ontology
from .resolve import CanonicalEntity, Resolution

__all__ = [
    "Evidence",
    "GraphEdge",
    "KnowledgeGraph",
    "build_graph",
    "to_cypher",
    "draw",
    "to_pyvis",
]

def _read_json_source(value: "str | Path") -> str:
    """Accept either a path or a JSON string.

    ``Path.exists()`` raises OSError on a long string, so sniff for JSON first
    rather than letting a serialised graph be mistaken for a filename.
    """
    text = str(value)
    if isinstance(value, Path) or not text.lstrip().startswith(("{", "[")):
        return Path(value).read_text()
    return text



@dataclass(frozen=True)
class Evidence:
    """Where in the corpus an edge came from."""

    doc_id: str
    head_span: tuple[int, int]
    tail_span: tuple[int, int]
    confidence: float
    snippet: str = ""

    def as_record(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "head_span": list(self.head_span),
            "tail_span": list(self.tail_span),
            "confidence": round(self.confidence, 4),
            "snippet": self.snippet,
        }


@dataclass
class GraphEdge:
    """A canonical edge: one relation between two resolved entities."""

    head: str  # canon_id
    type: str
    tail: str  # canon_id
    evidence: list[Evidence] = field(default_factory=list)
    derived: bool = False

    @property
    def confidence(self) -> float:
        """Best supporting occurrence.

        Max rather than mean: an edge stated clearly once and hedged twice is
        still stated clearly once. ``support`` carries the corroboration signal
        separately.
        """
        return max((e.confidence for e in self.evidence), default=0.0)

    @property
    def support(self) -> int:
        """How many distinct occurrences asserted this edge."""
        return len(self.evidence)

    @property
    def n_docs(self) -> int:
        return len({e.doc_id for e in self.evidence})

    def key(self) -> tuple[str, str, str]:
        return (self.head, self.type, self.tail)


@dataclass
class KnowledgeGraph:
    """Canonical entities + deduplicated, evidence-backed edges."""

    entities: dict[str, CanonicalEntity]
    edges: list[GraphEdge]
    ontology: Ontology | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def entity(self, canon_id: str) -> CanonicalEntity:
        return self.entities[canon_id]

    def name(self, canon_id: str) -> str:
        ent = self.entities.get(canon_id)
        return ent.canonical if ent else canon_id

    # -- querying --------------------------------------------------------

    def find(self, name: str, type_: str | None = None) -> list[CanonicalEntity]:
        """Look an entity up by canonical name or any alias, case-insensitively."""
        needle = name.casefold()
        return [
            e
            for e in self.entities.values()
            if (type_ is None or e.type == type_)
            and (
                needle == e.canonical.casefold()
                or needle in {a.casefold() for a in e.aliases}
            )
        ]

    def out_edges(self, canon_id: str, relation: str | None = None) -> list[GraphEdge]:
        return [
            e for e in self.edges
            if e.head == canon_id and (relation is None or e.type == relation)
        ]

    def in_edges(self, canon_id: str, relation: str | None = None) -> list[GraphEdge]:
        return [
            e for e in self.edges
            if e.tail == canon_id and (relation is None or e.type == relation)
        ]

    def neighbors(self, canon_id: str, relation: str | None = None) -> list[str]:
        seen: list[str] = []
        for e in self.edges:
            if relation is not None and e.type != relation:
                continue
            other = e.tail if e.head == canon_id else e.head if e.tail == canon_id else None
            if other and other not in seen:
                seen.append(other)
        return seen

    def triples(
        self, *, min_confidence: float = 0.0, min_support: int = 1, relation: str | None = None
    ) -> list[tuple[str, str, str]]:
        """Canonical-name triples, optionally filtered by strength."""
        return [
            (self.name(e.head), e.type, self.name(e.tail))
            for e in self.edges
            if e.confidence >= min_confidence
            and e.support >= min_support
            and (relation is None or e.type == relation)
        ]

    def frame(self, *, min_confidence: float = 0.0, min_support: int = 1):
        """Edges as a DataFrame, strongest first."""
        import pandas as pd

        rows = [
            {
                "head": self.name(e.head),
                "head_type": self.entities[e.head].type if e.head in self.entities else "?",
                "relation": e.type,
                "tail": self.name(e.tail),
                "tail_type": self.entities[e.tail].type if e.tail in self.entities else "?",
                "confidence": round(e.confidence, 3),
                "support": e.support,
                "docs": e.n_docs,
                "derived": e.derived,
            }
            for e in self.edges
            if e.confidence >= min_confidence and e.support >= min_support
        ]
        columns = ["head", "head_type", "relation", "tail", "tail_type",
                   "confidence", "support", "docs", "derived"]
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows).sort_values(
            ["support", "confidence"], ascending=False
        ).reset_index(drop=True)

    def evidence_for(self, head: str, relation: str, tail: str) -> list[Evidence]:
        """Every occurrence backing a triple, addressed by canonical names."""
        h = {e.canon_id for e in self.find(head)}
        t = {e.canon_id for e in self.find(tail)}
        out: list[Evidence] = []
        for e in self.edges:
            if e.type == relation and e.head in h and e.tail in t:
                out.extend(e.evidence)
        return sorted(out, key=lambda ev: -ev.confidence)

    def by_attr(self, attr: str, value: Any) -> list[CanonicalEntity]:
        """Entities whose qualifier attribute has this label -- e.g. modality."""
        out = []
        for e in self.entities.values():
            got = e.attrs.get(attr)
            label = got.get("label") if isinstance(got, dict) else got
            if label == value:
                out.append(e)
        return out

    # -- validation ------------------------------------------------------

    def violations(self) -> list[GraphEdge]:
        """Edges the ontology does not permit. Should be empty after JointIE."""
        if self.ontology is None:
            return []
        out = []
        for e in self.edges:
            h = self.entities.get(e.head)
            t = self.entities.get(e.tail)
            if h and t and not self.ontology.permits(h.type, e.type, t.type):
                out.append(e)
        return out

    # -- interop ---------------------------------------------------------

    def to_networkx(self):
        import networkx as nx

        g = nx.MultiDiGraph()
        for ent in self.entities.values():
            g.add_node(
                ent.canon_id,
                label=ent.canonical,
                type=ent.type,
                aliases=ent.aliases,
                n_mentions=len(ent.mentions),
                n_docs=len(set(ent.docs)),
                **{f"attr_{k}": (v.get("label") if isinstance(v, dict) else v)
                   for k, v in ent.attrs.items()},
            )
        for e in self.edges:
            g.add_edge(
                e.head, e.tail, key=e.type, label=e.type,
                confidence=round(e.confidence, 4), support=e.support,
                docs=sorted({ev.doc_id for ev in e.evidence}), derived=e.derived,
            )
        return g

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable form, evidence included."""
        return {
            "ontology": self.ontology.name if self.ontology else None,
            "stats": self.stats,
            "entities": [
                {
                    "canon_id": e.canon_id, "type": e.type, "canonical": e.canonical,
                    "aliases": e.aliases, "mentions": e.mentions, "docs": e.docs,
                    "attrs": e.attrs, "first_seen": e.first_seen, "last_seen": e.last_seen,
                }
                for e in self.entities.values()
            ],
            "edges": [
                {
                    "head": e.head, "type": e.type, "tail": e.tail, "derived": e.derived,
                    "evidence": [ev.as_record() for ev in e.evidence],
                }
                for e in self.edges
            ],
        }

    def to_json(self, path: str | Path | None = None, **kwargs: Any) -> str:
        """Serialise the graph so a later notebook can pick it up without re-extracting."""
        text = json.dumps(self.to_dict(), indent=1, **kwargs)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return text

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], ontology: Ontology | None = None) -> "KnowledgeGraph":
        entities = {
            e["canon_id"]: CanonicalEntity(
                canon_id=e["canon_id"], type=e["type"], canonical=e["canonical"],
                aliases=list(e.get("aliases", [])), mentions=list(e.get("mentions", [])),
                docs=list(e.get("docs", [])), attrs=dict(e.get("attrs", {})),
                first_seen=e.get("first_seen"), last_seen=e.get("last_seen"),
            )
            for e in data.get("entities", [])
        }
        edges = [
            GraphEdge(
                head=e["head"], type=e["type"], tail=e["tail"],
                derived=bool(e.get("derived", False)),
                evidence=[
                    Evidence(
                        doc_id=ev["doc_id"],
                        head_span=tuple(ev.get("head_span", (0, 0))),
                        tail_span=tuple(ev.get("tail_span", (0, 0))),
                        confidence=float(ev.get("confidence", 1.0)),
                        snippet=ev.get("snippet", ""),
                    )
                    for ev in e.get("evidence", [])
                ],
            )
            for e in data.get("edges", [])
        ]
        return cls(entities, edges, ontology, dict(data.get("stats", {})))

    @classmethod
    def from_json(cls, value: str | Path, ontology: Ontology | None = None) -> "KnowledgeGraph":
        return cls.from_dict(json.loads(_read_json_source(value)), ontology)

    def summary(self) -> str:
        types = Counter(e.type for e in self.entities.values())
        rels = Counter(e.type for e in self.edges)
        return (
            f"KnowledgeGraph: {len(self.entities)} entities, {len(self.edges)} edges\n"
            f"  node labels: {dict(types.most_common())}\n"
            f"  edge types:  {dict(rels.most_common())}"
        )


def build_graph(
    doc_graphs: Sequence[DocGraph],
    resolution: Resolution,
    ontology: Ontology | None = None,
    *,
    pin: Mapping[str, str] | None = None,
    snippet_width: int = 110,
    drop_self_loops: bool = True,
) -> KnowledgeGraph:
    """Rewrite endpoints to canonical ids and collapse duplicate edges.

    ``pin`` force-maps specific mention ids to a canonical id, overriding the
    resolver. That is how the assistant's own user gets to be one node
    (:data:`kgx.coref.USER_CANON_ID`) regardless of how the first-person
    substitution phrased it in any given turn.

    Self-loops created *by resolution* -- two mentions the extractor thought were
    different entities that turn out to be the same one -- are dropped by
    default. They are an artefact of merging, not a fact.
    """
    pin = dict(pin or {})
    entities = {cid: ent for cid, ent in resolution.entities.items()}

    # Apply pins by folding the affected clusters into one entity.
    if pin:
        for target_id in set(pin.values()):
            members = [mid for mid, cid in pin.items() if cid == target_id]
            if not members:
                continue
            # sorted, so the merged entity's name and type do not depend on set order
            sources = sorted({resolution.mention_to_canon.get(m) for m in members} - {None})
            base = next((entities[s] for s in sources if s in entities), None)
            existing = entities.get(target_id)
            anchor = existing or base
            merged = CanonicalEntity(
                canon_id=target_id,
                type=anchor.type if anchor else "user",
                canonical=anchor.canonical if anchor else target_id.split(":")[-1],
            )
            if existing is not None:
                # Pinning onto an id that already exists must fold into it rather
                # than replace it, or the pin silently deletes an entity.
                merged.aliases = list(existing.aliases)
                merged.mentions = list(existing.mentions)
                merged.docs = list(existing.docs)
                merged.attrs = dict(existing.attrs)
            for s in sources:
                src = entities.pop(s, None)
                if src is None:
                    continue
                merged.aliases = sorted(set(merged.aliases) | set(src.aliases) | {src.canonical})
                merged.mentions = sorted(set(merged.mentions) | set(src.mentions))
                merged.docs = sorted(set(merged.docs) | set(src.docs))
                merged.attrs.update(src.attrs)
            merged.aliases = [a for a in merged.aliases if a != merged.canonical]
            entities[target_id] = merged

    def canon_of(mention_id: str) -> str | None:
        if mention_id in pin:
            return pin[mention_id]
        cid = resolution.mention_to_canon.get(mention_id)
        if cid in entities:
            return cid
        # the cluster was folded into a pinned entity
        for eid, ent in entities.items():
            if mention_id in ent.mentions:
                return eid
        return None

    merged: dict[tuple[str, str, str], GraphEdge] = {}
    dropped_unresolved = 0
    dropped_self = 0

    for dg in doc_graphs:
        for edge in dg.edges:
            head_c, tail_c = canon_of(edge.head), canon_of(edge.tail)
            if head_c is None or tail_c is None:
                dropped_unresolved += 1
                continue
            if head_c == tail_c and drop_self_loops:
                dropped_self += 1
                continue
            hm, tm = dg.mention(edge.head), dg.mention(edge.tail)
            lo, hi = min(hm.start, tm.start), max(hm.end, tm.end)
            snippet = re.sub(
                r"\s+", " ",
                dg.text[max(0, lo - 20): min(len(dg.text), hi + 20)],
            ).strip()
            if len(snippet) > snippet_width:
                snippet = snippet[:snippet_width].rstrip() + "…"
            key = (head_c, edge.type, tail_c)
            if key not in merged:
                merged[key] = GraphEdge(head_c, edge.type, tail_c, derived=edge.derived)
            merged[key].evidence.append(
                Evidence(dg.doc_id, (hm.start, hm.end), (tm.start, tm.end),
                         edge.confidence, snippet)
            )
            merged[key].derived = merged[key].derived and edge.derived

    raw_edges = sum(len(dg.edges) for dg in doc_graphs)
    stats = {
        "documents": len(doc_graphs),
        "raw_mentions": sum(len(dg.mentions) for dg in doc_graphs),
        "canonical_entities": len(entities),
        "raw_edges": raw_edges,
        "canonical_edges": len(merged),
        "edge_reduction": round(1 - len(merged) / max(raw_edges, 1), 3),
        "dropped_self_loops": dropped_self,
        "dropped_unresolved": dropped_unresolved,
    }
    return KnowledgeGraph(entities, list(merged.values()), ontology, stats)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def _cypher_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_cypher_literal(v) for v in value) + "]"
    return json.dumps(str(value))


def _label(type_: str) -> str:
    """``business_segment`` -> ``BusinessSegment``.

    Split on non-alphanumerics, not ``\\W`` -- underscore is a word character, so
    ``\\W`` leaves it in and yields ``Business_segment``.
    """
    return "".join(p.capitalize() for p in re.split(r"[^0-9A-Za-z]+", type_) if p) or "Entity"


def to_cypher(
    graph: KnowledgeGraph,
    *,
    min_confidence: float = 0.0,
    min_support: int = 1,
    batch: int = 500,
    include_constraints: bool = True,
) -> str:
    """Idempotent Cypher for loading the graph into Neo4j.

    ``MERGE`` on ``canon_id`` throughout, so re-running after a re-extraction
    updates in place rather than duplicating. Evidence is written onto the
    relationship as parallel arrays -- enough to click from an edge back to the
    sentence.
    """
    lines: list[str] = ["// Generated by kgx.graph.to_cypher"]

    if include_constraints:
        for type_ in sorted({e.type for e in graph.entities.values()}):
            lines.append(
                f"CREATE CONSTRAINT {_label(type_).lower()}_id IF NOT EXISTS "
                f"FOR (n:{_label(type_)}) REQUIRE n.canon_id IS UNIQUE;"
            )
        lines.append("")

    ents = list(graph.entities.values())
    for i in range(0, len(ents), batch):
        for ent in ents[i : i + batch]:
            props = {
                "name": ent.canonical,
                "aliases": ent.aliases,
                "n_mentions": len(ent.mentions),
                "n_docs": len(set(ent.docs)),
            }
            for k, v in ent.attrs.items():
                props[k] = v.get("label") if isinstance(v, dict) else v
            body = ", ".join(f"n.{k} = {_cypher_literal(v)}" for k, v in props.items())
            lines.append(
                f"MERGE (n:{_label(ent.type)} {{canon_id: {_cypher_literal(ent.canon_id)}}}) "
                f"SET {body};"
            )
    lines.append("")

    for e in graph.edges:
        if e.confidence < min_confidence or e.support < min_support:
            continue
        if e.head not in graph.entities or e.tail not in graph.entities:
            continue
        rel = re.sub(r"[^\w]", "_", e.type).upper()
        docs = sorted({ev.doc_id for ev in e.evidence})
        snippets = [ev.snippet for ev in sorted(e.evidence, key=lambda x: -x.confidence)[:3]]
        lines.append(
            f"MATCH (a {{canon_id: {_cypher_literal(e.head)}}}), "
            f"(b {{canon_id: {_cypher_literal(e.tail)}}})\n"
            f"MERGE (a)-[r:{rel}]->(b)\n"
            f"SET r.confidence = {e.confidence:.4f}, r.support = {e.support}, "
            f"r.docs = {_cypher_literal(docs)}, r.evidence = {_cypher_literal(snippets)};"
        )
    return "\n".join(lines)


PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#7F3C8D", "#11A579",
    "#3969AC", "#F2B701", "#E73F74", "#80BA5A",
]


def type_colors(graph: KnowledgeGraph) -> dict[str, str]:
    types = sorted({e.type for e in graph.entities.values()})
    return {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}


def _edge_labels(g) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for u, v, d in g.edges(data=True):
        grouped.setdefault((u, v), []).append(d["label"])
    return {k: "\n".join(dict.fromkeys(v)) for k, v in grouped.items()}


def draw(
    graph: KnowledgeGraph,
    *,
    min_confidence: float = 0.0,
    min_support: int = 1,
    figsize: tuple[int, int] = (15, 11),
    seed: int = 7,
    max_nodes: int | None = None,
    title: str = "",
    ax=None,
):
    """Static matplotlib rendering. Good enough to read, cheap to regenerate."""
    import matplotlib.pyplot as plt
    import networkx as nx

    g = graph.to_networkx()
    keep = {
        (u, v, k)
        for u, v, k, d in g.edges(keys=True, data=True)
        if d["confidence"] >= min_confidence and d["support"] >= min_support
    }
    g.remove_edges_from([e for e in list(g.edges(keys=True)) if e not in keep])
    g.remove_nodes_from([n for n, d in list(g.degree()) if d == 0])

    if max_nodes and g.number_of_nodes() > max_nodes:
        top = sorted(g.degree(), key=lambda kv: -kv[1])[:max_nodes]
        g = g.subgraph([n for n, _ in top]).copy()

    if g.number_of_nodes() == 0:
        raise ValueError("nothing left to draw at these thresholds")

    colors = type_colors(graph)
    pos = nx.spring_layout(g, seed=seed, k=1.5 / max(g.number_of_nodes(), 1) ** 0.35, iterations=180)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        g, pos, ax=ax,
        node_color=[colors.get(g.nodes[n].get("type"), "#999") for n in g],
        node_size=[420 + 130 * g.nodes[n].get("n_mentions", 1) for n in g],
        alpha=0.92, linewidths=0.6, edgecolors="white",
    )
    nx.draw_networkx_edges(
        g, pos, ax=ax, edge_color="#8a8a8a", alpha=0.45,
        arrowsize=11, width=[0.6 + 0.45 * g.edges[e]["support"] for e in g.edges(keys=True)],
        connectionstyle="arc3,rad=0.09", node_size=700,
    )
    nx.draw_networkx_labels(
        g, pos, ax=ax,
        labels={n: g.nodes[n].get("label", n)[:26] for n in g},
        font_size=8,
    )
    nx.draw_networkx_edge_labels(
        g, pos, ax=ax, font_size=6, alpha=0.75, rotate=False,
        # join parallel edges rather than letting the last one win: two relation
        # types between the same pair are two facts, not one
        edge_labels=_edge_labels(g),
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.65},
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c, label=t)
        for t, c in colors.items()
        if any(g.nodes[n].get("type") == t for n in g)
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False, ncol=2)
    ax.set_title(title or f"{g.number_of_nodes()} entities, {g.number_of_edges()} edges", fontsize=11)
    ax.axis("off")
    return ax


def to_pyvis(
    graph: KnowledgeGraph,
    path: str | Path = "graph.html",
    *,
    min_confidence: float = 0.0,
    min_support: int = 1,
    height: str = "720px",
    notebook: bool = True,
):
    """Interactive HTML rendering. Hover a node for its aliases, an edge for evidence."""
    from pyvis.network import Network

    colors = type_colors(graph)
    net = Network(height=height, width="100%", directed=True, notebook=notebook,
                  cdn_resources="in_line", bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-14000, spring_length=190, spring_strength=0.02)

    used: set[str] = set()
    edges = [
        e for e in graph.edges
        if e.confidence >= min_confidence and e.support >= min_support
        and e.head in graph.entities and e.tail in graph.entities
    ]
    for e in edges:
        used.add(e.head)
        used.add(e.tail)

    for cid in used:
        ent = graph.entities[cid]
        tip = [f"<b>{html.escape(ent.canonical)}</b>", f"type: {ent.type}",
               f"mentions: {len(ent.mentions)} in {len(set(ent.docs))} doc(s)"]
        if ent.aliases:
            tip.append("aliases: " + html.escape(", ".join(ent.aliases[:8])))
        for k, v in ent.attrs.items():
            tip.append(f"{k}: {html.escape(str(v.get('label') if isinstance(v, dict) else v))}")
        net.add_node(
            cid, label=ent.canonical[:30], title="<br>".join(tip),
            color=colors.get(ent.type, "#999"),
            size=14 + 3.2 * len(ent.mentions), shape="dot",
        )

    for e in edges:
        tip = [f"<b>{e.type}</b>", f"confidence: {e.confidence:.2f}", f"support: {e.support}"]
        tip += [f"<i>{html.escape(ev.doc_id)}</i>: {html.escape(ev.snippet)}"
                for ev in sorted(e.evidence, key=lambda x: -x.confidence)[:3]]
        net.add_edge(
            e.head, e.tail, label=e.type, title="<br>".join(tip),
            width=0.8 + 0.5 * e.support, arrows="to",
            color={"color": "#9aa0a6", "opacity": 0.7}, font={"size": 9, "align": "middle"},
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(path), notebook=False, open_browser=False)
    return path
