"""kgx - a sandbox for entity extraction and resolution for knowledge graphs.

Pipeline, end to end::

    ontology  ->  extract  ->  [coref]  ->  resolve  ->  graph  ->  [temporal]

:mod:`kgx.ontology`
    The graph model as data: node labels, edge types, legal triples. Compiles to
    a GLiNER2.5 ``JointSchema`` and also drives validation and Cypher export.
:mod:`kgx.extract`
    GLiNER2.5 joint entity+relation decoding, plus a span-attribute pass for the
    qualifiers (modality, direction) that typed edges cannot carry.
:mod:`kgx.coref`
    Deterministic conversation preprocessing, plus an optional trained
    coreference layer. Encoder models have no coreference; this is what stops
    that from looking like an extraction failure.
:mod:`kgx.resolve`
    normalize -> block -> score -> cluster -> canonicalize, plus an incremental
    registry for episode-at-a-time agent memory and B-cubed scoring.
:mod:`kgx.graph`
    Canonical, evidence-backed graph assembly; matplotlib, pyvis and Cypher output.
:mod:`kgx.temporal`
    Bi-temporal fact store: supersede contradictions instead of accumulating them.
"""

from .ontology import AGENT_MEMORY, BUSINESS_NEWS, EntityType, Ontology, RelationType
from .extract import (
    DEFAULT_MODEL,
    FAST_MODEL,
    MULTILINGUAL_MODEL,
    DocGraph,
    Edge,
    GlinerExtractor,
    Mention,
    edges_frame,
    mentions_frame,
)
from .coref import (
    USER_CANON_ID,
    ConversationPreprocessor,
    NeuralCorefPreprocessor,
    load_coref_engine,
)
from .resolve import CanonicalRegistry, EntityResolver, Resolution, bcubed, normalize
from .graph import KnowledgeGraph, build_graph, draw, to_cypher, to_pyvis
from .temporal import (
    Fact,
    TemporalGraph,
    explicit_alternatives,
    graph_alternatives,
)

__version__ = "0.1.0"

__all__ = [
    "AGENT_MEMORY", "BUSINESS_NEWS", "Ontology", "EntityType", "RelationType",
    "GlinerExtractor", "DocGraph", "Mention", "Edge",
    "DEFAULT_MODEL", "FAST_MODEL", "MULTILINGUAL_MODEL",
    "mentions_frame", "edges_frame",
    "ConversationPreprocessor", "USER_CANON_ID",
    "NeuralCorefPreprocessor", "load_coref_engine",
    "EntityResolver", "CanonicalRegistry", "Resolution", "bcubed", "normalize",
    "KnowledgeGraph", "build_graph", "to_cypher", "draw", "to_pyvis",
    "TemporalGraph", "Fact", "graph_alternatives", "explicit_alternatives",
]
