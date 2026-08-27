"""Closed-vocabulary NER as the honest floor in an extractor comparison.

spaCy's `en_core_web_trf` scores 0.899 F1 on OntoNotes NER -- far above anything
zero-shot manages on its own labels. Any comparison that omits it is flattering
the zero-shot models, because for a great many real problems the answer is
"your types are PERSON/ORG/GPE, use a supervised tagger and stop reading".

The catch is in the mapping below. OntoNotes has 18 labels chosen for a 2006
newswire annotation project; `BUSINESS_NEWS` has 13 chosen for this corpus.
Five map cleanly. The rest are either absent from OntoNotes entirely
(`risk_factor`, `financial_metric`, `business_segment`) or present as labels
nobody asked for (`PERCENT`, `ORDINAL`, `WORK_OF_ART`). That gap is not a
deficiency in spaCy -- it is what "closed vocabulary" means, and it is the whole
reason the zero-shot models exist.

spaCy also has no relation extraction without training a custom component, so it
contributes entities only. A graph needs edges; this baseline structurally
cannot produce any.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from .extract import DocGraph, Mention, _context_window
from .ontology import Ontology

__all__ = ["ONTONOTES_TO_BUSINESS_NEWS", "SpacyExtractor", "SPACY_MODELS"]

SPACY_MODELS = ("en_core_web_trf", "en_core_web_lg", "en_core_web_sm")

# OntoNotes label -> BUSINESS_NEWS type. Deliberately conservative: ORG covers
# both companies and regulators and spaCy cannot tell them apart, so it maps to
# `company` and the confusion is left visible in the scores rather than papered
# over with a keyword list.
ONTONOTES_TO_BUSINESS_NEWS: dict[str, str] = {
    "ORG": "company",
    "PERSON": "person",
    "GPE": "geography",
    "LOC": "geography",
    "PRODUCT": "product",
    "EVENT": "business_event",
    # Deliberately unmapped, and worth naming: MONEY, DATE, PERCENT, CARDINAL,
    # ORDINAL, QUANTITY, TIME, NORP, FAC, LAW, LANGUAGE, WORK_OF_ART.
}

# The ontology types no OntoNotes label can reach at all.
UNREACHABLE = (
    "regulator", "business_segment", "sector", "security",
    "financial_metric", "risk_factor", "litigation", "commodity",
)


class SpacyExtractor:
    """Entity-only extraction with a supervised, closed-vocabulary tagger."""

    def __init__(
        self,
        model: str = "en_core_web_trf",
        mapping: Mapping[str, str] | None = None,
        *,
        context_width: int = 90,
    ) -> None:
        import spacy

        last: Exception | None = None
        for candidate in (model, *[m for m in SPACY_MODELS if m != model]):
            try:
                self.nlp = spacy.load(candidate)
                self.model = candidate
                break
            except OSError as exc:  # pragma: no cover - depends on what is installed
                last = exc
        else:
            raise RuntimeError(
                f"no spaCy model available; install one with "
                f"`uv run python -m spacy download {model}`"
            ) from last

        self.mapping = dict(mapping or ONTONOTES_TO_BUSINESS_NEWS)
        self.context_width = context_width
        self.labels: tuple[str, ...] = tuple(self.nlp.get_pipe("ner").labels)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        f1 = self.nlp.meta.get("performance", {}).get("ents_f")
        return (f"<SpacyExtractor {self.model} {len(self.labels)} labels"
                + (f" ents_f={f1:.3f}>" if f1 else ">"))

    def coverage(self, ontology: Ontology) -> dict[str, Any]:
        """How much of an ontology this tagger can even express."""
        reachable = {v for v in self.mapping.values() if v in set(ontology.entity_names)}
        missing = [t for t in ontology.entity_names if t not in reachable]
        return {
            "ontology_types": len(ontology.entity_names),
            "reachable": len(reachable),
            "unreachable": missing,
            "spacy_labels_unused": [l for l in self.labels if l not in self.mapping],
            "relations_supported": 0,
            "ontology_relations": len(ontology.relations),
        }

    def extract(self, text: str, ontology: Ontology, doc_id: str = "doc", **meta: Any) -> DocGraph:
        started = time.time()
        doc = self.nlp(text)
        valid = set(ontology.entity_names)
        mentions = []
        for i, ent in enumerate(doc.ents, start=1):
            mapped = self.mapping.get(ent.label_)
            if mapped is None or mapped not in valid:
                continue
            mentions.append(Mention(
                mention_id=f"{doc_id}:e{i}", doc_id=doc_id, type=mapped, text=ent.text,
                start=ent.start_char, end=ent.end_char, confidence=1.0,
                context=_context_window(text, ent.start_char, ent.end_char, self.context_width),
                attrs={"spacy_label": ent.label_},
            ))
        return DocGraph(doc_id=doc_id, text=text, mentions=mentions, edges=[],
                        feasible=True, elapsed_s=time.time() - started, meta=dict(meta))

    def extract_batch(
        self, docs: Sequence[Mapping[str, Any]], ontology: Ontology,
        *, text_key: str = "text", id_key: str = "doc_id", batch_size: int = 8,
    ) -> list[DocGraph]:
        texts = [d[text_key] for d in docs]
        started = time.time()
        processed = list(self.nlp.pipe(texts, batch_size=batch_size))
        per_doc = (time.time() - started) / max(len(texts), 1)
        valid = set(ontology.entity_names)

        graphs = []
        for i, (raw, doc) in enumerate(zip(docs, processed)):
            doc_id = raw.get(id_key, f"doc{i}")
            mentions = []
            for j, ent in enumerate(doc.ents, start=1):
                mapped = self.mapping.get(ent.label_)
                if mapped is None or mapped not in valid:
                    continue
                mentions.append(Mention(
                    mention_id=f"{doc_id}:e{j}", doc_id=doc_id, type=mapped, text=ent.text,
                    start=ent.start_char, end=ent.end_char, confidence=1.0,
                    context=_context_window(raw[text_key], ent.start_char, ent.end_char,
                                            self.context_width),
                    attrs={"spacy_label": ent.label_},
                ))
            graphs.append(DocGraph(
                doc_id=doc_id, text=raw[text_key], mentions=mentions, edges=[],
                feasible=True, elapsed_s=per_doc,
                meta={k: v for k, v in raw.items() if k not in {text_key, id_key}},
            ))
        return graphs
