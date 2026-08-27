"""Bi-temporal agent memory in Neo4j: schema, incremental write-back, retrieval.

:mod:`kgx.neo4j_io` loads a *finished* graph. That is the right shape for
document intelligence, where a corpus arrives once and never changes. Agent
memory is the other shape entirely -- episodes arrive one at a time, every
episode may contradict what is already stored, and the whole graph is read back
into a prompt on every turn. Three consequences drive everything in this module.

**Facts need validity, so validity has to live somewhere.** The choice is
between putting ``valid_from`` / ``valid_until`` / ``superseded_at`` on the
relationship, or reifying each fact into its own node. Both are implemented here
(:func:`load_facts` and :func:`reify_facts`) because the trade only becomes
concrete when you can measure it. Relationship properties keep every read at one
hop and cannot express a fact *about* a fact; reification is fully general and
doubles the hop count on every query.

**Resolution has to run against the database.** A new episode's mentions must be
linked to entities that already exist, and "already exist" means "are in Neo4j",
not "are in a Python dict". :class:`LiveResolver` does candidate generation with
a Cypher query -- exact normalised match plus a fulltext probe -- and scores the
handful of candidates that come back with the same
:meth:`~kgx.resolve.EntityResolver.score_pair` used in batch resolution. The
in-memory :class:`~kgx.resolve.CanonicalRegistry` scores every new mention
against *every* known entity, which is fine at 200 entities and not at 10^6.

**Supersession is a graph pattern.** The reason preferring pnpm invalidates
preferring npm is that the user said pnpm replaced npm, and that statement is an
edge in the same graph. :func:`assert_fact` expresses the rule as a Cypher match
over ``REPLACES``, so the store layer needs no alternatives table and no
embedding threshold.

Everything written here is namespaced -- ``Mem``-prefixed labels plus the shared
``__Memory__`` label -- so a memory graph can coexist with an unrelated graph in
the single database a Neo4j Community instance gives you. :func:`teardown`
removes exactly what this module wrote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .neo4j_io import label_for, rel_type_for
from .resolve import EntityResolver, normalize
from .temporal import SUPERSEDING_RELATIONS, Fact

__all__ = [
    "drop_reification",
    "await_index",
    "MEMORY_LABEL",
    "LABEL_PREFIX",
    "mem_label",
    "to_datetime",
    "create_schema",
    "load_episodes",
    "load_facts",
    "reify_facts",
    "current_facts",
    "facts_as_of",
    "memory_prompt_one_query",
    "memory_prompt_naive",
    "LiveResolver",
    "assert_fact",
    "ALREADY_ASSERTED",
    "teardown",
    "memory_counts",
]

MEMORY_LABEL = "__Memory__"
"""Shared secondary label on every node this module writes.

Serves the same purpose as ``__Entity__`` in :mod:`kgx.neo4j_io` -- one label an
index or a teardown can address -- but deliberately a *different* one, so a
memory graph loaded beside a document graph does not join its indexes or its
counts.
"""

LABEL_PREFIX = "Mem"
"""Prefix on every ontology label. ``person`` -> ``MemPerson``.

Neo4j Community gives you one database. Two graphs that both call a node
``Person`` are one graph with a confusing schema, and every aggregate query over
either of them is silently wrong. Prefixing is the cheapest thing that makes
coexistence safe for label-scoped queries -- and note the qualifier: it does
nothing for ``MATCH (n) RETURN count(n)``.
"""


def mem_label(entity_type: str) -> str:
    """``business_segment`` -> ``MemBusinessSegment``."""
    return LABEL_PREFIX + label_for(entity_type)


def to_datetime(value: Any) -> Any:
    """ISO-8601 string (``Z`` suffix tolerated) -> timezone-aware ``datetime``.

    Validity is stored as a native temporal type rather than a string. ISO-8601
    strings do happen to compare correctly with ``<`` as long as every timestamp
    is UTC with the same precision, which is exactly the kind of invariant that
    holds until the day one caller writes ``+02:00`` and every point-in-time
    query starts quietly returning the wrong answer.
    """
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def create_schema(driver, labels: Iterable[str], *, database: str = "neo4j") -> dict[str, list[str]]:
    """Constraints and search indexes for the memory namespace.

    The fulltext index is not decoration: it is the candidate generator for
    :class:`LiveResolver`. Without it, resolving a new episode against the live
    graph means pulling every entity back to the client.
    """
    constraints, indexes = [], []
    for label in sorted({*labels, "MemEpisode", "MemFact", MEMORY_LABEL}):
        name = f"{re.sub(r'[^a-z0-9]', '', label.lower())}_canon_id"
        key = "episode_id" if label == "MemEpisode" else (
            "fact_id" if label == "MemFact" else "canon_id")
        name = f"{re.sub(r'[^a-z0-9]', '', label.lower())}_{key}"
        driver.execute_query(
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE",
            database_=database,
        )
        constraints.append(name)

    driver.execute_query(
        f"CREATE INDEX mem_entity_norm IF NOT EXISTS FOR (n:{MEMORY_LABEL}) ON (n.norm)",
        database_=database,
    )
    indexes.append("mem_entity_norm")
    driver.execute_query(
        f"CREATE FULLTEXT INDEX mem_entity_ft IF NOT EXISTS "
        f"FOR (n:{MEMORY_LABEL}) ON EACH [n.name, n.aliases]",
        database_=database,
    )
    indexes.append("mem_entity_ft")
    return {"constraints": constraints, "indexes": indexes}


# ---------------------------------------------------------------------------
# write path -- bulk
# ---------------------------------------------------------------------------

def load_episodes(driver, sessions: Sequence[Mapping[str, Any]], *,
                  database: str = "neo4j") -> int:
    """Write ``(:MemEpisode)`` nodes -- the transaction-time anchor.

    Every fact points back to the episode that asserted it. Keeping episodes as
    nodes rather than as a string property is what makes "everything we learned
    on 4 March" a traversal instead of a scan.
    """
    rows = [
        {
            "episode_id": s["session_id"],
            "props": {
                "timestamp": to_datetime(s["timestamp"]),
                "title": s.get("title", ""),
                "n_turns": len(s.get("turns", [])),
            },
        }
        for s in sessions
    ]
    driver.execute_query(
        f"UNWIND $rows AS row MERGE (e:MemEpisode:{MEMORY_LABEL} "
        f"{{episode_id: row.episode_id}}) SET e += row.props",
        rows=rows, database_=database,
    )
    return len(rows)


def _entity_rows(graph, canon_ids: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for cid in sorted(set(canon_ids)):
        e = graph.entities.get(cid)
        if e is None:
            continue
        rows.append({
            "canon_id": e.canon_id,
            "label": mem_label(e.type),
            "props": {
                "name": e.canonical,
                "type": e.type,
                "aliases": list(e.aliases),
                "norm": normalize(e.canonical, e.type).key,
                "n_mentions": len(e.mentions),
                "episodes": sorted({d.split(":", 1)[0] for d in e.docs}),
            },
        })
    return rows


def _fact_rows(facts: Sequence[Fact]) -> list[dict[str, Any]]:
    return [{
        "head": f.head,
        "tail": f.tail,
        "rel_type": rel_type_for(f.relation),
        "props": {
            "fact_id": f.fact_id,
            "relation": f.relation,
            "episode": f.episode_id,
            "confidence": round(f.confidence, 4),
            "support": f.support,
            "valid_from": to_datetime(f.valid_from),
            "valid_until": to_datetime(f.valid_until),
            "recorded_at": to_datetime(f.recorded_at),
            "superseded_at": to_datetime(f.superseded_at),
            "superseded_by": f.superseded_by,
            "episodes": [f.episode_id],
            "evidence": list(f.evidence[:3]),
        },
    } for f in facts]


def load_facts(driver, graph, facts: Sequence[Fact], *, database: str = "neo4j",
               batch_size: int = 500) -> dict[str, Any]:
    """Encoding A: one relationship per fact, validity as relationship properties.

    The relationship is keyed on ``fact_id`` (``head|relation|tail|episode``)
    rather than on its endpoints, so a fact that is superseded and later asserted
    again gets a second parallel relationship instead of overwriting the first.
    Neo4j allows parallel relationships of the same type between the same pair;
    a schema that forbade them could not hold a history.
    """
    canon_ids = {f.head for f in facts} | {f.tail for f in facts}
    nodes = _entity_rows(graph, canon_ids)
    create_schema(driver, {r["label"] for r in nodes}, database=database)

    node_q = (f"UNWIND $rows AS row MERGE (n:$(row.label) {{canon_id: row.canon_id}}) "
              f"SET n:$($mem), n += row.props")
    for i in range(0, len(nodes), batch_size):
        driver.execute_query(node_q, rows=nodes[i:i + batch_size], mem=MEMORY_LABEL,
                             database_=database)

    edges = _fact_rows(facts)
    edge_q = f"""
    UNWIND $rows AS row
    MATCH (h:$($mem) {{canon_id: row.head}})
    MATCH (t:$($mem) {{canon_id: row.tail}})
    MERGE (h)-[r:$(row.rel_type) {{fact_id: row.props.fact_id}}]->(t)
    SET r += row.props
    """
    for i in range(0, len(edges), batch_size):
        driver.execute_query(edge_q, rows=edges[i:i + batch_size], mem=MEMORY_LABEL,
                             database_=database)
    return {"nodes": len(nodes), "facts": len(edges)}


def reify_facts(driver, *, database: str = "neo4j") -> dict[str, Any]:
    """Encoding B: rewrite the graph so every fact is a ``(:MemFact)`` node.

    Runs as one Cypher statement over whatever encoding A already wrote, so the
    two encodings hold exactly the same facts and can be timed against each
    other::

        (subject)-[:SUBJECT_OF]->(:MemFact)-[:HAS_OBJECT]->(object)
        (:MemFact)-[:ASSERTED_IN]->(:MemEpisode)

    One node and three relationships where encoding A used one relationship, and
    every read costs two hops instead of one.

    What that buys is generality. A ``MemFact`` node can be an endpoint, so
    per-source confidence, adjudication decisions, and facts *about* facts are
    all expressible; on encoding A they are not, because Neo4j has no
    relationship-to-relationship edge and a list property is not a traversal.
    The ``ASSERTED_IN`` edge is the smallest example: on encoding A the episode
    is a string property, and "everything we learned on 20 May" is a scan.
    """
    query = f"""
    MATCH (h:{MEMORY_LABEL})-[r]->(t:{MEMORY_LABEL})
    WHERE r.fact_id IS NOT NULL
    MERGE (f:MemFact:{MEMORY_LABEL} {{fact_id: r.fact_id}})
    SET f.relation = r.relation, f.episode = r.episode, f.confidence = r.confidence,
        f.support = r.support, f.valid_from = r.valid_from, f.valid_until = r.valid_until,
        f.recorded_at = r.recorded_at, f.superseded_at = r.superseded_at
    MERGE (h)-[:SUBJECT_OF]->(f)
    MERGE (f)-[:HAS_OBJECT]->(t)
    WITH f, r
    OPTIONAL MATCH (e:MemEpisode {{episode_id: r.episode}})
    FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
        MERGE (f)-[:ASSERTED_IN]->(e))
    RETURN count(*) AS n
    """
    create_schema(driver, ["MemFact"], database=database)
    records, _, _ = driver.execute_query(query, database_=database)
    return {"fact_nodes": records[0]["n"] if records else 0}


# ---------------------------------------------------------------------------
# read path
# ---------------------------------------------------------------------------

CURRENT_QUERY = f"""
MATCH (h:{MEMORY_LABEL})-[r]->(t:{MEMORY_LABEL})
WHERE r.relation IS NOT NULL AND r.superseded_at IS NULL
  AND ($subject IS NULL OR h.name = $subject)
  AND ($relation IS NULL OR r.relation = $relation)
RETURN h.name AS head, r.relation AS relation, t.name AS tail, t.type AS tail_type,
       r.valid_from AS valid_from, r.episode AS episode, r.confidence AS confidence
ORDER BY relation, confidence DESC
"""

AS_OF_VALID = f"""
MATCH (h:{MEMORY_LABEL})-[r]->(t:{MEMORY_LABEL})
WHERE r.relation IS NOT NULL
  AND r.valid_from <= $when AND (r.valid_until IS NULL OR r.valid_until > $when)
  AND ($subject IS NULL OR h.name = $subject)
  AND ($relation IS NULL OR r.relation = $relation)
RETURN h.name AS head, r.relation AS relation, t.name AS tail, r.episode AS episode
ORDER BY relation, tail
"""

AS_OF_TRANSACTION = f"""
MATCH (h:{MEMORY_LABEL})-[r]->(t:{MEMORY_LABEL})
WHERE r.relation IS NOT NULL
  AND r.recorded_at <= $when AND (r.superseded_at IS NULL OR r.superseded_at > $when)
  AND ($subject IS NULL OR h.name = $subject)
  AND ($relation IS NULL OR r.relation = $relation)
RETURN h.name AS head, r.relation AS relation, t.name AS tail, r.episode AS episode
ORDER BY relation, tail
"""


def _frame(records):
    import pandas as pd

    return pd.DataFrame([dict(r) for r in records])


def current_facts(driver, *, subject: str | None = None, relation: str | None = None,
                  database: str = "neo4j"):
    """What the assistant believes now: ``superseded_at IS NULL``."""
    records, _, _ = driver.execute_query(
        CURRENT_QUERY, subject=subject, relation=relation, database_=database)
    return _frame(records)


def facts_as_of(driver, when: Any, *, subject: str | None = None,
                relation: str | None = None, timeline: str = "valid",
                database: str = "neo4j"):
    """State at a past instant, on the world timeline or the system's.

    ``timeline="valid"`` answers *what was true then*; ``"transaction"`` answers
    *what did we believe then*. They differ whenever an episode reported
    something that had already been true for a while, which is most of the time.
    """
    query = AS_OF_VALID if timeline == "valid" else AS_OF_TRANSACTION
    records, _, _ = driver.execute_query(
        query, when=to_datetime(when), subject=subject, relation=relation,
        database_=database)
    return _frame(records)


PROMPT_ONE_QUERY = f"""
MATCH (u:{MEMORY_LABEL} {{canon_id: $canon_id}})-[r]->(t:{MEMORY_LABEL})
WHERE r.relation IS NOT NULL AND r.superseded_at IS NULL
WITH r.relation AS relation, t.name AS value, r.confidence AS confidence
ORDER BY confidence DESC
WITH relation, collect(DISTINCT value)[..$per_relation] AS values
RETURN relation, values ORDER BY relation
"""


def memory_prompt_one_query(driver, canon_id: str, *, name: str | None = None,
                            per_relation: int = 8, database: str = "neo4j") -> str:
    """Assemble ``TemporalGraph.memory_prompt``-style context in one round trip.

    One traversal from the subject, grouped server-side. This is what runs on
    every turn, so the shape of the query matters more than it usually does.
    """
    records, _, _ = driver.execute_query(
        PROMPT_ONE_QUERY, canon_id=canon_id, per_relation=per_relation, database_=database)
    subject = name or canon_id
    if not records:
        return f"No stored facts about {subject}."
    lines = [f"Known facts about {subject}:"]
    for r in records:
        lines.append(f"- {r['relation'].replace('_', ' ')}: {', '.join(r['values'])}")
    return "\n".join(lines)


def memory_prompt_naive(driver, canon_id: str, *, name: str | None = None,
                        relations: Sequence[str] | None = None,
                        per_relation: int = 8, database: str = "neo4j") -> str:
    """The same context, one query per relation type. Here to be timed against.

    This is what a memory layer with a per-relation accessor API produces, and it
    is the obvious thing to write. It costs one network round trip per relation
    in the ontology whether or not the subject has any facts of that type.
    """
    relations = list(relations or SUPERSEDING_RELATIONS)
    subject = name or canon_id
    lines = []
    for relation in sorted(relations):
        # Typed expansion, not `WHERE r.relation = $relation`: the relationship
        # type is itself an index and the property is not. This keeps the
        # comparison about round trips rather than about a bad predicate.
        records, _, _ = driver.execute_query(
            f"MATCH (u:{MEMORY_LABEL} {{canon_id: $canon_id}})-[r:$($rel_type)]->(t:{MEMORY_LABEL}) "
            f"WHERE r.superseded_at IS NULL "
            f"RETURN t.name AS value ORDER BY r.confidence DESC LIMIT $limit",
            canon_id=canon_id, rel_type=rel_type_for(relation), limit=per_relation,
            database_=database)
        if records:
            values = list(dict.fromkeys(r["value"] for r in records))
            lines.append(f"- {relation.replace('_', ' ')}: {', '.join(values)}")
    if not lines:
        return f"No stored facts about {subject}."
    return "\n".join([f"Known facts about {subject}:", *lines])


# ---------------------------------------------------------------------------
# incremental write-back
# ---------------------------------------------------------------------------

_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


def _lucene_escape(text: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", text)


CANDIDATE_QUERY = f"""
CALL () {{
    MATCH (c:{MEMORY_LABEL}) WHERE c.norm = $norm AND c.type = $type
    RETURN c, 1.0 AS score
  UNION
    CALL db.index.fulltext.queryNodes('mem_entity_ft', $q, {{limit: $k}})
    YIELD node AS c, score
    WHERE c.type = $type
    RETURN c, score
}}
RETURN c.canon_id AS canon_id, c.name AS name, c.type AS type,
       c.aliases AS aliases, max(score) AS score
ORDER BY score DESC LIMIT $k
"""


@dataclass
class LiveResolver:
    """Resolve a new episode's mentions against the graph already in Neo4j.

    The two-pass structure is the same as
    :class:`~kgx.resolve.CanonicalRegistry` -- match against what is known, then
    resolve the unmatched remainder among itself -- but the first pass generates
    candidates with a Cypher query instead of scoring against every known
    entity. That is the difference between O(known) work per mention and O(k).

    ``pins`` maps a surface form to a fixed canonical id. The assistant's own
    user must be exactly one node; that is not a decision a similarity threshold
    should get to make.
    """

    driver: Any
    resolver: EntityResolver = field(default_factory=lambda: EntityResolver(threshold=0.90))
    database: str = "neo4j"
    top_k: int = 8
    pins: dict[str, str] = field(default_factory=dict)
    candidates_seen: int = 0
    queries_run: int = 0

    def candidates(self, mention) -> list[dict[str, Any]]:
        """Blocking, done in the database: exact normalised key OR fulltext."""
        norm = normalize(mention.text, mention.type)
        q = _lucene_escape(norm.display) or _lucene_escape(mention.text)
        try:
            records, _, _ = self.driver.execute_query(
                CANDIDATE_QUERY, norm=norm.key, type=mention.type, q=q, k=self.top_k,
                database_=self.database)
        except Exception:
            # A malformed Lucene query should degrade to exact matching, not
            # abort the episode.
            records, _, _ = self.driver.execute_query(
                f"MATCH (c:{MEMORY_LABEL}) WHERE c.norm = $norm AND c.type = $type "
                f"RETURN c.canon_id AS canon_id, c.name AS name, c.type AS type, "
                f"c.aliases AS aliases, 1.0 AS score",
                norm=norm.key, type=mention.type, database_=self.database)
        self.queries_run += 1
        rows = [dict(r) for r in records]
        self.candidates_seen += len(rows)
        return rows

    def _anchor(self, row: Mapping[str, Any]):
        from .extract import Mention

        return Mention(
            mention_id=f"__db__:{row['canon_id']}",
            doc_id="__db__",
            type=row["type"],
            text=row["name"],
            start=-1, end=-1, confidence=1.0,
            context=" ".join((row.get("aliases") or [])[:6]),
        )

    def match(self, mention) -> tuple[str | None, float, str]:
        """Best existing entity for one mention, or ``(None, ...)`` if it is new."""
        pin = self.pins.get(mention.text.strip().casefold())
        if pin:
            return pin, 1.0, "pinned"
        best: tuple[str | None, float, str] = (None, 0.0, "no candidate above threshold")
        na = normalize(mention.text, mention.type)
        for row in self.candidates(mention):
            anchor = self._anchor(row)
            nb = normalize(anchor.text, anchor.type)
            pair = self.resolver.score_pair(mention, anchor, na, nb, 0.0, "db")
            if pair.score > best[1]:
                best = (row["canon_id"], pair.score, pair.reason or "string similarity")
        if best[1] < self.resolver.threshold:
            return (None, best[1], best[2])
        return best

    def resolve_episode(self, graphs: Sequence[Any]) -> dict[str, Any]:
        """Two passes over one episode's document graphs. Returns the id map.

        Pass 1 links what it can to the database. Pass 2 resolves the leftovers
        against each other, because an episode that mentions "Acme" and "Acme
        Corp" for the first time must still produce one node, and neither had a
        database entry to anchor to.
        """
        mentions = [m for g in graphs for m in g.mentions]
        assigned: dict[str, str] = {}
        details: list[dict[str, Any]] = []
        unmatched = []
        for m in mentions:
            canon_id, score, reason = self.match(m)
            details.append({"mention": m.mention_id, "text": m.text, "type": m.type,
                            "canon_id": canon_id, "score": round(score, 3),
                            "reason": reason, "pass": 1 if canon_id else 2})
            if canon_id:
                assigned[m.mention_id] = canon_id
            else:
                unmatched.append(m)

        new_entities: dict[str, Any] = {}
        if unmatched:
            res = self.resolver.resolve(unmatched)
            for mid, cid in res.mention_to_canon.items():
                assigned[mid] = cid
            new_entities = dict(res.entities)
            for row in details:
                if row["pass"] == 2:
                    row["canon_id"] = assigned.get(row["mention"])
        return {
            "assigned": assigned,
            "new_entities": new_entities,
            "details": details,
            "linked": sum(1 for d in details if d["pass"] == 1),
            "new": len(new_entities),
            "candidate_queries": self.queries_run,
            "candidates_scored": self.candidates_seen,
        }


MERGE_ENTITY = f"""
UNWIND $rows AS row
MERGE (n:$(row.label) {{canon_id: row.canon_id}})
SET n:$($mem), n += row.props
"""

SUPERSEDE_SINGLE = f"""
MATCH (h:{MEMORY_LABEL} {{canon_id: $head}})-[old]->(prev:{MEMORY_LABEL})
WHERE old.relation = $relation AND old.superseded_at IS NULL AND prev.canon_id <> $tail
MATCH (new:{MEMORY_LABEL} {{canon_id: $tail}})
SET old.valid_until = $valid_from, old.superseded_at = $recorded_at,
    old.superseded_by = $fact_id
RETURN h.name AS head, old.relation AS relation, prev.name AS was, new.name AS now,
       old.episode AS was_episode
"""

# The supersession rule for "alternatives" relations, expressed as the graph
# pattern it actually is: the old value is invalidated only when the new value
# REPLACES it, and REPLACES is an edge the extractor produced from a sentence the
# user said. No alternatives table, no embedding threshold, and the justification
# is one hop away in the same graph.
SUPERSEDE_ALTERNATIVES = f"""
MATCH (h:{MEMORY_LABEL} {{canon_id: $head}})-[old]->(prev:{MEMORY_LABEL})
WHERE old.relation = $relation AND old.superseded_at IS NULL AND prev.canon_id <> $tail
MATCH (new:{MEMORY_LABEL} {{canon_id: $tail}})
WHERE (new)-[:REPLACES]->(prev) OR (prev)<-[:REPLACES]-(new)
SET old.valid_until = $valid_from, old.superseded_at = $recorded_at,
    old.superseded_by = $fact_id
RETURN h.name AS head, old.relation AS relation, prev.name AS was, new.name AS now,
       old.episode AS was_episode
"""

# Replay detection, and it has to come first and has to ignore whether the fact
# is still current. The obvious version -- fold replay into the corroboration
# query, which only looks at current facts -- gets it wrong in one specific case:
# an episode whose own later facts superseded one of its earlier ones. On the
# second ingest that fact is no longer current, corroboration misses it, and the
# episode inserts a duplicate edge. Matching on fact_id OR membership in the
# episode list catches both shapes.
ALREADY_ASSERTED = f"""
MATCH (h:{MEMORY_LABEL} {{canon_id: $head}})-[r]->(t:{MEMORY_LABEL} {{canon_id: $tail}})
WHERE r.relation = $relation
  AND (r.fact_id = $fact_id OR $episode IN coalesce(r.episodes, [r.episode]))
RETURN r.fact_id AS fact_id LIMIT 1
"""

# Corroboration: the same triple, asserted again by a *different* episode. Bump
# support, do not create a second edge, and above all do not touch valid_until --
# closing the window on a fact the graph still believes deletes it from every
# later point-in-time query.
CORROBORATE = f"""
MATCH (h:{MEMORY_LABEL} {{canon_id: $head}})-[r]->(t:{MEMORY_LABEL} {{canon_id: $tail}})
WHERE r.relation = $relation AND r.superseded_at IS NULL
SET r.support = r.support + $support,
    r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
    r.episodes = coalesce(r.episodes, [r.episode]) + $episode,
    r.last_seen = $valid_from
RETURN r.fact_id AS fact_id
"""

INSERT_FACT = f"""
MATCH (h:{MEMORY_LABEL} {{canon_id: $head}})
MATCH (t:{MEMORY_LABEL} {{canon_id: $tail}})
MERGE (h)-[r:$($rel_type) {{fact_id: $fact_id}}]->(t)
SET r += $props
RETURN r.fact_id AS fact_id
"""


def assert_fact(driver, fact: Fact, *, policies: Mapping[str, str] | None = None,
                database: str = "neo4j") -> dict[str, Any]:
    """Write one fact against the live graph, superseding what it contradicts.

    Mirrors :meth:`kgx.temporal.TemporalGraph.assert_fact` with the state in the
    database instead of a Python list. Three outcomes, in order:

    corroboration
        The same ``(head, relation, tail)`` is already current. Bump support and
        confidence, record the extra episode, do not create a second edge --
        and crucially do not touch ``valid_until``, because closing the window on
        a fact you still believe deletes it from every later point-in-time query.
    supersession
        Under ``single`` every other current value for the relation is closed;
        under ``alternatives`` only the values the new value ``REPLACES``.
    insertion
        Whatever survives is written as a new relationship keyed on ``fact_id``.
    """
    policies = policies or SUPERSEDING_RELATIONS
    policy = policies.get(fact.relation, "accumulate")
    params = {
        "head": fact.head, "tail": fact.tail, "relation": fact.relation,
        "fact_id": fact.fact_id, "episode": fact.episode_id,
        "valid_from": to_datetime(fact.valid_from),
        "recorded_at": to_datetime(fact.recorded_at),
        "confidence": round(fact.confidence, 4), "support": fact.support,
    }

    records, _, _ = driver.execute_query(ALREADY_ASSERTED, **params, database_=database)
    if records:
        return {"action": "replayed", "fact_id": records[0]["fact_id"], "superseded": []}

    records, _, _ = driver.execute_query(CORROBORATE, **params, database_=database)
    if records:
        return {"action": "corroborated", "fact_id": records[0]["fact_id"], "superseded": []}

    superseded = []
    if policy in ("single", "alternatives"):
        query = SUPERSEDE_SINGLE if policy == "single" else SUPERSEDE_ALTERNATIVES
        records, _, _ = driver.execute_query(query, **params, database_=database)
        superseded = [dict(r) for r in records]

    props = {
        "fact_id": fact.fact_id, "relation": fact.relation, "episode": fact.episode_id,
        "confidence": round(fact.confidence, 4), "support": fact.support,
        "valid_from": to_datetime(fact.valid_from),
        "valid_until": to_datetime(fact.valid_until),
        "recorded_at": to_datetime(fact.recorded_at),
        "superseded_at": to_datetime(fact.superseded_at),
        "episodes": [fact.episode_id],
        "evidence": list(fact.evidence[:3]),
    }
    driver.execute_query(INSERT_FACT, **params, rel_type=rel_type_for(fact.relation),
                         props=props, database_=database)
    return {"action": "inserted", "fact_id": fact.fact_id, "superseded": superseded}


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------

def memory_counts(driver, *, database: str = "neo4j") -> dict[str, Any]:
    """Node/relationship counts for the memory namespace only."""
    records, _, _ = driver.execute_query(
        f"MATCH (n:{MEMORY_LABEL}) RETURN count(n) AS nodes", database_=database)
    nodes = records[0]["nodes"]
    records, _, _ = driver.execute_query(
        f"MATCH (:{MEMORY_LABEL})-[r]->(:{MEMORY_LABEL}) RETURN count(r) AS rels",
        database_=database)
    rels = records[0]["rels"]
    records, _, _ = driver.execute_query(
        f"MATCH (n:{MEMORY_LABEL}) UNWIND labels(n) AS l "
        f"WITH l WHERE l <> '{MEMORY_LABEL}' RETURN l, count(*) AS c ORDER BY c DESC",
        database_=database)
    return {"nodes": nodes, "relationships": rels,
            "by_label": {r["l"]: r["c"] for r in records}}


def teardown(driver, *, database: str = "neo4j") -> dict[str, Any]:
    """Remove everything this module wrote: nodes, indexes, constraints.

    Namespacing makes coexistence safe for label-scoped queries. It does not make
    it free -- a database-wide ``MATCH (n) RETURN count(n)`` still sees both
    graphs -- so a notebook that borrows someone else's database should be able
    to give it back exactly as found.
    """
    deleted = 0
    while True:
        records, _, _ = driver.execute_query(
            f"MATCH (n:{MEMORY_LABEL}) WITH n LIMIT 10000 DETACH DELETE n "
            f"RETURN count(*) AS deleted", database_=database)
        n = records[0]["deleted"] if records else 0
        deleted += n
        if not n:
            break

    dropped = []
    records, _, _ = driver.execute_query(
        "SHOW INDEXES YIELD name, labelsOrTypes, owningConstraint "
        "WHERE name STARTS WITH 'mem' AND owningConstraint IS NULL RETURN name",
        database_=database)
    for r in records:
        driver.execute_query(f"DROP INDEX {r['name']} IF EXISTS", database_=database)
        dropped.append(r["name"])

    records, _, _ = driver.execute_query(
        "SHOW CONSTRAINTS YIELD name, labelsOrTypes RETURN name, labelsOrTypes",
        database_=database)
    constraints = []
    for r in records:
        labels = r["labelsOrTypes"] or []
        if any(str(l).startswith(LABEL_PREFIX) or str(l) == MEMORY_LABEL for l in labels):
            driver.execute_query(f"DROP CONSTRAINT {r['name']} IF EXISTS", database_=database)
            constraints.append(r["name"])

    return {"nodes_deleted": deleted, "indexes_dropped": dropped,
            "constraints_dropped": constraints}


# ---------------------------------------------------------------------------
# episode -> facts -> database
# ---------------------------------------------------------------------------

def merge_entities(driver, entities: Iterable[Any], *, database: str = "neo4j") -> int:
    """MERGE canonical entities discovered in a new episode.

    Takes :class:`~kgx.resolve.CanonicalEntity` objects, which is what both the
    batch resolver and :meth:`LiveResolver.resolve_episode` produce.
    """
    rows = []
    for e in entities:
        rows.append({
            "canon_id": e.canon_id,
            "label": mem_label(e.type),
            "props": {
                "name": e.canonical,
                "type": e.type,
                "aliases": list(e.aliases),
                "norm": normalize(e.canonical, e.type).key,
                "n_mentions": len(e.mentions),
                "episodes": sorted({d.split(":", 1)[0] for d in e.docs}),
            },
        })
    if not rows:
        return 0
    create_schema(driver, {r["label"] for r in rows}, database=database)
    driver.execute_query(MERGE_ENTITY, rows=rows, mem=MEMORY_LABEL, database_=database)
    return len(rows)


def facts_from_graphs(graphs: Sequence[Any], assignment: Mapping[str, str],
                      names: Mapping[str, tuple[str, str]], *, episode_id: str,
                      valid_from: str, min_confidence: float = 0.6) -> list[Fact]:
    """Collapse an episode's document-local edges into canonical :class:`Fact` rows.

    Windows overlap, so the same statement is seen more than once; occurrences of
    the same canonical triple are folded into one fact whose ``support`` counts
    them. Support here counts *occurrences*, not independent witnesses -- with a
    stride of 3 over windows of 4, one turn is read by two windows.
    """
    from collections import defaultdict

    buckets: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    snippets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for g in graphs:
        for edge in g.edges:
            if edge.confidence < min_confidence:
                continue
            h, t = assignment.get(edge.head), assignment.get(edge.tail)
            if not h or not t or h == t:
                continue
            key = (h, edge.type, t)
            buckets[key].append(edge)
            snippets[key].append(g.text[max(0, g.mention(edge.head).start - 40):
                                        g.mention(edge.head).start + 80].replace("\n", " "))

    facts = []
    for (h, relation, t), edges in buckets.items():
        head_name, _ = names.get(h, (h, ""))
        tail_name, tail_type = names.get(t, (t, ""))
        facts.append(Fact(
            head=h, relation=relation, tail=t,
            head_name=head_name, tail_name=tail_name, tail_type=tail_type,
            confidence=max(e.confidence for e in edges), support=len(edges),
            episode_id=episode_id, valid_from=valid_from, recorded_at=valid_from,
            evidence=list(dict.fromkeys(snippets[(h, relation, t)]))[:2],
        ))
    # `replaces` first: supersession for "alternatives" relations is a match over
    # REPLACES edges, so the evidence has to be in the graph before the facts it
    # invalidates are evaluated. Otherwise the episode that announces the switch
    # is exactly the episode that fails to apply it.
    facts.sort(key=lambda f: (f.relation != "replaces", -f.confidence))
    return facts


def write_episode(driver, facts: Sequence[Fact], *, policies: Mapping[str, str] | None = None,
                  database: str = "neo4j") -> dict[str, Any]:
    """Assert a whole episode against the live graph. Returns a per-fact report."""
    report = {"inserted": 0, "corroborated": 0, "replayed": 0, "superseded": []}
    for fact in facts:
        out = assert_fact(driver, fact, policies=policies, database=database)
        report[out["action"]] += 1
        for s in out["superseded"]:
            report["superseded"].append({**s, "by_episode": fact.episode_id,
                                         "fact_id": fact.fact_id})
    return report


def entity_names(driver, canon_ids: Iterable[str], *, database: str = "neo4j"):
    """``canon_id -> (name, type)`` for entities already in the graph."""
    records, _, _ = driver.execute_query(
        f"MATCH (n:{MEMORY_LABEL}) WHERE n.canon_id IN $ids "
        f"RETURN n.canon_id AS canon_id, n.name AS name, n.type AS type",
        ids=list(canon_ids), database_=database)
    return {r["canon_id"]: (r["name"], r["type"]) for r in records}


def drop_reification(driver, *, database: str = "neo4j") -> int:
    """Remove encoding B, leaving encoding A intact.

    Worth having as a one-liner because the two encodings do not compose
    silently: an untyped ``MATCH (u)-[r]->(t)`` written for encoding A also
    traverses ``SUBJECT_OF``, and those relationships have no ``relation`` and no
    ``superseded_at``, so "everything the assistant currently believes" quietly
    grows a row per fact. Every read query here guards with
    ``r.relation IS NOT NULL`` for that reason.
    """
    records, _, _ = driver.execute_query(
        "MATCH (f:MemFact) DETACH DELETE f RETURN count(*) AS n", database_=database)
    return records[0]["n"] if records else 0


def await_index(driver, name: str, *, timeout: float = 60.0, database: str = "neo4j") -> str:
    """Block until an index reports ONLINE.

    Index creation returns before the index is queryable. At this scale
    population takes milliseconds, but profiling a query against a half-built
    index reports the plan you were trying to avoid, which is a confusing way to
    lose ten minutes.
    """
    import time as _time

    deadline = _time.time() + timeout
    while True:
        records, _, _ = driver.execute_query(
            "SHOW INDEXES YIELD name, state WHERE name = $name RETURN state",
            name=name, database_=database)
        state = records[0]["state"] if records else "MISSING"
        if state == "ONLINE":
            return state
        if _time.time() > deadline:
            raise TimeoutError(f"index {name} is {state} after {timeout}s")
        _time.sleep(0.1)
