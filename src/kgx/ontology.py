"""Domain ontologies: the declarative graph model that drives extraction.

An :class:`Ontology` is a portable, JSON-serialisable description of a property
graph -- which node labels exist, which edge types exist, and which
``(head_type, RELATION, tail_type)`` triples are legal. It compiles down to a
GLiNER2.5 :class:`~gliner2.joint_ie.JointSchema`, which the model uses to decode
entities *and* relations as one globally consistent graph rather than as two
independent tasks that have to be stitched together afterwards.

The point of keeping this layer separate from GLiNER is that the same ontology
object also drives validation, Cypher/DDL export, and documentation. Swap the
extractor and the graph model survives.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "EntityType",
    "RelationType",
    "Ontology",
    "AGENT_MEMORY",
    "BUSINESS_NEWS",
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



def _as_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


@dataclass(frozen=True)
class EntityType:
    """A node label.

    ``description`` is not documentation -- GLiNER serialises it into the encoder
    input, so it behaves like a zero-shot annotation guideline. Write it the way
    you would write instructions for a human annotator, including the negative
    case ("Not a one-off task"). Label wording is a hyperparameter; see the
    sensitivity sweep in the notebook.
    """

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("entity type name must be non-empty")


@dataclass(frozen=True)
class RelationType:
    """An edge type with typed endpoints and optional structural constraints.

    Constraints are enforced *during* decoding, not filtered afterwards, so they
    change which graph the model chooses rather than just pruning a bad one.

    ``unique_head``
        At most one edge of this type per head node -- "a task has one assignee".
    ``unique_tail``
        At most one edge of this type per tail node.
    ``acyclic``
        The subgraph induced by this edge type must be a DAG -- "``part_of``
        cannot loop".
    ``inverse``
        Name of another relation whose endpoint types are the exact mirror of
        this one. Decoding emits the mirrored edge with ``derived=True``. This is
        how you express symmetry -- see the note on ``symmetric`` below.

    .. warning::
       ``JointSchema`` also accepts ``symmetric=True``, but in ``gliner2`` 2.0.0
       that flag compiles to a constraint set that rejects every candidate edge:
       the relation silently yields **zero** results with ``feasible=True``. Use
       a directed relation plus ``inverse=`` (or symmetrise in post-processing)
       instead. ``Ontology.compile`` never sets ``symmetric``.
    """

    name: str
    head: tuple[str, ...]
    tail: tuple[str, ...]
    description: str = ""
    unique_head: bool = False
    unique_tail: bool = False
    acyclic: bool = False
    inverse: str | None = None
    threshold: float | None = None
    allow_self: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "head", _as_tuple(self.head))
        object.__setattr__(self, "tail", _as_tuple(self.tail))
        if not self.head or not self.tail:
            raise ValueError(f"relation {self.name!r} needs at least one head and tail type")

    def patterns(self) -> set[tuple[str, str, str]]:
        """Every legal ``(head_type, relation, tail_type)`` triple."""
        return {(h, self.name, t) for h in self.head for t in self.tail}


@dataclass(frozen=True)
class Ontology:
    """A named graph model: node labels, edge types, and legal triples."""

    name: str
    entities: tuple[EntityType, ...]
    relations: tuple[RelationType, ...]
    description: str = ""
    no_self_loops: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "relations", tuple(self.relations))
        known = set(self.entity_names)
        if len(known) != len(self.entities):
            raise ValueError("duplicate entity type names")
        seen: set[str] = set()
        for rel in self.relations:
            if rel.name in seen:
                raise ValueError(f"duplicate relation name {rel.name!r}")
            seen.add(rel.name)
            unknown = (set(rel.head) | set(rel.tail)) - known
            if unknown:
                raise ValueError(
                    f"relation {rel.name!r} references undeclared entity types: {sorted(unknown)}"
                )
        for rel in self.relations:
            if rel.inverse is None:
                continue
            other = self.relation(rel.inverse)
            if set(rel.head) != set(other.tail) or set(rel.tail) != set(other.head):
                raise ValueError(
                    f"inverse endpoints for {rel.name!r} and {rel.inverse!r} are not mirrored; "
                    f"GLiNER requires head(a) == tail(b) and tail(a) == head(b)"
                )

    # -- lookups ---------------------------------------------------------

    @property
    def entity_names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.entities)

    @property
    def relation_names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.relations)

    def entity(self, name: str) -> EntityType:
        for e in self.entities:
            if e.name == name:
                return e
        raise KeyError(f"unknown entity type {name!r}")

    def relation(self, name: str) -> RelationType:
        for r in self.relations:
            if r.name == name:
                return r
        raise KeyError(f"unknown relation {name!r}")

    def patterns(self) -> set[tuple[str, str, str]]:
        """Every legal triple pattern in the ontology."""
        out: set[tuple[str, str, str]] = set()
        for rel in self.relations:
            out |= rel.patterns()
        return out

    def permits(self, head_type: str, relation: str, tail_type: str) -> bool:
        """Is this triple legal under the ontology?"""
        try:
            rel = self.relation(relation)
        except KeyError:
            return False
        return head_type in rel.head and tail_type in rel.tail

    def subset(
        self,
        entities: Iterable[str] | None = None,
        relations: Iterable[str] | None = None,
        *,
        name: str | None = None,
    ) -> "Ontology":
        """A narrower ontology over the same model.

        Relation endpoint types are pruned to the retained entity types, and any
        relation left with no legal head or tail is dropped. Useful for drawing a
        readable diagram of a 15-relation ontology, and for the ablation in the
        notebook that measures what each additional relation type costs.
        """
        keep_e = set(entities) if entities is not None else set(self.entity_names)
        unknown = keep_e - set(self.entity_names)
        if unknown:
            raise KeyError(f"unknown entity types: {sorted(unknown)}")
        keep_r = set(relations) if relations is not None else set(self.relation_names)

        new_entities = tuple(e for e in self.entities if e.name in keep_e)
        new_relations: list[RelationType] = []
        for rel in self.relations:
            if rel.name not in keep_r:
                continue
            head = tuple(t for t in rel.head if t in keep_e)
            tail = tuple(t for t in rel.tail if t in keep_e)
            if not head or not tail:
                continue
            inverse = rel.inverse if rel.inverse in keep_r else None
            new_relations.append(
                RelationType(rel.name, head, tail, rel.description, rel.unique_head,
                             rel.unique_tail, rel.acyclic, inverse, rel.threshold, rel.allow_self)
            )
        return Ontology(
            name=name or f"{self.name}_subset",
            entities=new_entities,
            relations=tuple(new_relations),
            description=self.description,
            no_self_loops=self.no_self_loops,
        )

    # -- compilation -----------------------------------------------------

    def compile(self, joint: Any = None):
        """Build the ``JointSchema`` GLiNER2.5 decodes against.

        ``joint`` may be a ``JointIE`` engine (its ``create_schema()`` is used) or
        ``None``, in which case ``JointSchema`` is constructed directly. The two
        are equivalent; passing the engine just mirrors the upstream examples.
        """
        if joint is not None and hasattr(joint, "create_schema"):
            schema = joint.create_schema()
        else:
            from gliner2.joint_ie import JointSchema

            schema = JointSchema()

        for ent in self.entities:
            schema.entity(ent.name, ent.description or None)

        for rel in self.relations:
            schema.relation(
                rel.name,
                list(rel.head),
                list(rel.tail),
                rel.description or None,
                threshold=rel.threshold,
                allow_self=rel.allow_self,
                inverse=rel.inverse,
                unique_head=rel.unique_head,
                unique_tail=rel.unique_tail,
                acyclic=rel.acyclic,
            )

        if self.no_self_loops:
            # A global no-self-loops constraint would override any relation that
            # explicitly permits them, so apply it per relation instead.
            for rel in self.relations:
                if not rel.allow_self:
                    schema.no_self_loops(rel.name)
        return schema

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "no_self_loops": self.no_self_loops,
            "entities": [asdict(e) for e in self.entities],
            "relations": [asdict(r) for r in self.relations],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Ontology":
        """Load an ontology from a plain dict (so it can live in YAML or JSON).

        ``relations`` entries accept ``head``/``tail`` as a string or a list.
        """
        entities = tuple(
            EntityType(**e) if isinstance(e, Mapping) else EntityType(e)
            for e in data.get("entities", ())
        )
        relations = tuple(RelationType(**r) for r in data.get("relations", ()))
        return cls(
            name=data.get("name", "ontology"),
            description=data.get("description", ""),
            entities=entities,
            relations=relations,
            no_self_loops=data.get("no_self_loops", True),
        )

    def to_json(self, path: str | Path | None = None, **kwargs: Any) -> str:
        text = json.dumps(self.to_dict(), indent=2, **kwargs)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, value: str | Path) -> "Ontology":
        return cls.from_dict(json.loads(_read_json_source(value)))

    # -- presentation ----------------------------------------------------

    def to_mermaid(self) -> str:
        """A Mermaid graph of the *model* (types and legal edges), not the data."""
        lines = ["graph LR"]
        for e in self.entities:
            lines.append(f"  {e.name}([{e.name}])")
        seen: set[tuple[str, str, str]] = set()
        for rel in self.relations:
            for h in rel.head:
                for t in rel.tail:
                    key = (h, rel.name, t)
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(f"  {h} -->|{rel.name}| {t}")
        return "\n".join(lines)

    def summary(self) -> str:
        rels = "\n".join(
            f"  {r.name:<18} {'|'.join(r.head):<28} -> {'|'.join(r.tail)}"
            + "".join(
                f"  [{flag}]"
                for flag in (
                    "unique_head" if r.unique_head else "",
                    "unique_tail" if r.unique_tail else "",
                    "acyclic" if r.acyclic else "",
                    f"inverse={r.inverse}" if r.inverse else "",
                )
                if flag
            )
            for r in self.relations
        )
        return (
            f"Ontology {self.name!r}: {len(self.entities)} entity types, "
            f"{len(self.relations)} relations, {len(self.patterns())} legal triple patterns\n"
            f"  entities: {', '.join(self.entity_names)}\n{rels}"
        )


# ---------------------------------------------------------------------------
# Ontology A -- agent memory
# ---------------------------------------------------------------------------
# Adapted from the node/edge vocabulary that temporal agent-memory systems
# (Graphiti/Zep, mem0, Cognee) converge on, with two adaptations forced by
# GLiNER2.5's JointSchema:
#
#   1. `prefers` and `avoids` are separate relations rather than one edge with a
#      polarity property, because JointSchema edges carry no attributes.
#   2. `constraint` is a node, not an edge, so every hard rule for the user can
#      be enumerated with one label lookup instead of a traversal. That is what
#      prompt assembly actually needs on each turn.
#
# GLiNER2.5 has no temporal model at all. Validity intervals are a store-layer
# concern -- see kgx.temporal.

AGENT_MEMORY = Ontology(
    name="agent_memory",
    description=(
        "Personal work/coding-assistant memory graph: who the user is, what they "
        "are working on, who with, what they prefer, and what constrains them."
    ),
    entities=(
        EntityType("user", "The single human this assistant serves; first-person mentions."),
        EntityType("person", "A named human other than the user: colleague, manager, client."),
        EntityType("organization", "A company, employer, client, vendor, team, or foundation."),
        EntityType("project", "A named body of work: product, initiative, repository. Not a one-off task."),
        EntityType("task", "A discrete actionable item with a doer and a completion state."),
        EntityType("goal", "An outcome the user is pursuing; an intention, not an action."),
        EntityType("constraint", "A hard rule or limit the assistant must respect: policy, deadline window, compliance rule."),
        EntityType("tool", "A software product, service, library, language, framework, or device."),
        EntityType("skill", "A competency a person has or is building."),
        EntityType("event", "A time-bound meeting, deadline, trip, incident, or release."),
        EntityType("place", "An office, city, venue, or virtual room."),
        EntityType("artifact", "A produced object: document, repository, pull request, design, dataset."),
        EntityType("topic", "A subject of interest or knowledge domain. Last resort - check every other type first."),
    ),
    relations=(
        RelationType("works_at", ("user", "person"), ("organization",),
                     "The person is employed by or contracts for the organization."),
        RelationType("works_on", ("user", "person"), ("project",),
                     "The person actively contributes to the project."),
        RelationType("assigned_to", ("task",), ("user", "person"),
                     "The task is owned by this person.", unique_head=True),
        RelationType("part_of", ("task", "project"), ("project", "organization"),
                     "Containment: the head belongs to the tail.", unique_head=True, acyclic=True),
        RelationType("collaborates_with", ("user", "person"), ("user", "person"),
                     "The two people work together directly."),
        RelationType("prefers", ("user",), ("tool", "topic", "organization", "place"),
                     "The user has stated a positive preference for this."),
        RelationType("avoids", ("user",), ("tool", "topic", "organization", "place"),
                     "The user has stated they dislike, refuse, or want to stop using this."),
        RelationType("has_constraint", ("user", "person", "project", "organization"), ("constraint",),
                     "A hard rule that binds the head."),
        RelationType("pursues", ("user", "person", "project"), ("goal",),
                     "The head is trying to achieve this outcome."),
        RelationType("uses_tool", ("user", "person", "project"), ("tool",),
                     "The head currently uses this tool. Neutral, unlike prefers/avoids."),
        RelationType("has_skill", ("user", "person"), ("skill",),
                     "The person has or is building this competency."),
        RelationType("attended", ("user", "person"), ("event",),
                     "The person took part in the event."),
        RelationType("located_in", ("user", "person", "organization", "event"), ("place",),
                     "Where the head is based or takes place.", unique_head=True),
        RelationType("produced", ("user", "person", "project"), ("artifact",),
                     "The head authored or created this artifact."),
        RelationType("blocked_by", ("task", "project"), ("task", "project", "person", "tool", "constraint"),
                     "Progress on the head is prevented by the tail.", acyclic=True),
        RelationType("replaces", ("tool", "topic", "place", "organization"),
                     ("tool", "topic", "place", "organization"),
                     "The head has superseded or been adopted instead of the tail: a switch, "
                     "a migration away from, a standardisation on something else.",
                     acyclic=True),
    ),
)


# ---------------------------------------------------------------------------
# Ontology B -- document intelligence (business & financial news)
# ---------------------------------------------------------------------------
# Chosen over contracts because every relation fires in a five-sentence press
# release, so a small corpus still yields a connected, legible graph.
#
# `impacts` is deliberately NOT split into positively_impacts / negatively_impacts
# / affects_guidance / ... . Predicate explosion costs encoder input budget and
# hurts recall. Direction and modality are recovered by a second span-attribute
# pass and stored as node attributes -- see kgx.extract.Qualifier.

BUSINESS_NEWS = Ontology(
    name="business_news",
    description=(
        "Business and financial news / filing narratives: who owns, supplies, "
        "competes with, and regulates whom, and what events move the numbers."
    ),
    entities=(
        EntityType("company", "A commercial issuer or private firm, referred to by name."),
        EntityType("person", "An executive, director, founder, analyst, or named official."),
        EntityType("regulator", "A government body, agency, central bank, or court."),
        EntityType("product", "A named product, service, or platform."),
        EntityType("business_segment", "A reportable segment or division of a company."),
        EntityType("geography", "A country, region, state, city, or named market."),
        EntityType("sector", "An industry classification, e.g. semiconductors, freight, retail banking."),
        EntityType("security", "An issued instrument: shares, notes, bonds, or a ticker symbol."),
        EntityType("financial_metric", "A named measure: revenue, operating margin, EPS, free cash flow."),
        EntityType("risk_factor", "A disclosed risk or exposure, e.g. tariff exposure, supply disruption."),
        EntityType("litigation", "A named case, investigation, or enforcement action."),
        EntityType("business_event", "The newsworthy corporate event itself: acquisition, layoff, recall, guidance cut."),
        EntityType("commodity", "A raw material or physical input: diesel, silicon, copper."),
    ),
    relations=(
        RelationType("subsidiary_of", ("company",), ("company",),
                     "The head company is owned or controlled by the tail company.",
                     unique_head=True, acyclic=True),
        RelationType("officer_of", ("person",), ("company", "regulator"),
                     "The person holds an executive or board position at the organization.",
                     unique_head=True),
        RelationType("acquires", ("company",), ("company", "business_segment", "product"),
                     "The head company is buying or has bought the tail.", acyclic=True),
        RelationType("has_stake_in", ("company", "person"), ("company", "security"),
                     "The head holds an ownership interest in the tail."),
        RelationType("supplies", ("company", "commodity"), ("company",),
                     "The head provides goods or inputs to the tail company."),
        RelationType("partners_with", ("company",), ("company", "regulator"),
                     "A stated partnership, alliance, or joint venture."),
        RelationType("competes_with", ("company",), ("company",),
                     "The two companies are stated rivals in a market."),
        RelationType("operates_in", ("company", "business_segment"), ("geography", "sector"),
                     "Where or in which industry the head does business."),
        RelationType("produces", ("company", "business_segment"), ("product", "commodity"),
                     "The head makes or sells the tail."),
        RelationType("reports_metric", ("company", "business_segment"), ("financial_metric",),
                     "The head disclosed a value for this measure."),
        RelationType("faces_risk", ("company", "sector"), ("risk_factor",),
                     "The head is exposed to this risk."),
        RelationType("party_to", ("company", "person"), ("litigation",),
                     "The head is a named party in the case or investigation."),
        RelationType("subject_to", ("company", "product"), ("regulator",),
                     "The head falls under this body's oversight or review."),
        RelationType("participant_in", ("company", "person", "regulator"), ("business_event",),
                     "The head takes part in the event."),
        RelationType("impacts", ("business_event", "commodity", "risk_factor"),
                     ("company", "business_segment", "sector", "financial_metric", "security", "geography"),
                     "The head materially affects the tail. Direction and certainty are "
                     "recovered separately as span attributes.", acyclic=True),
    ),
)
