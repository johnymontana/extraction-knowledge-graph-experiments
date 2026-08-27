"""Reload notebook 02's Neo4j state: the graph, the documents, and the four indexes.

Run after `scripts/neo4j_up.sh` on a fresh instance, or any time a notebook has
left the database somewhere unexpected. Idempotent.
"""
from __future__ import annotations

import sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import kgx
from kgx.graph import KnowledgeGraph
from kgx.data.documents import DOCUMENTS
from kgx.neo4j_io import (Neo4jConfig, connect, clear_database, drop_search_indexes,
                          load_graph, load_documents, counts, ENTITY_LABEL)


def main() -> int:
    cfg = Neo4jConfig.from_env(uri="bolt://localhost:7690", user="neo4j", password="sandbox-kg")
    driver = connect(cfg, notifications="OFF")
    try:
        print(f"connected to {cfg.uri}")
        clear_database(driver)
        drop_search_indexes(driver)

        kg = KnowledgeGraph.from_json(ROOT / "output" / "business_news_kg.json", kgx.BUSINESS_NEWS)
        print(" ", load_graph(driver, kg))
        print(" ", load_documents(driver, DOCUMENTS, kg))

        from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
        from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index, upsert_vectors
        from neo4j_graphrag.types import EntityType

        embedder = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        dim = len(embedder.embed_query("probe"))
        create_vector_index(driver, "document_vec", label="Document",
                            embedding_property="embedding", dimensions=dim, similarity_fn="cosine")
        create_fulltext_index(driver, "document_ft", label="Document", node_properties=["title", "text"])
        create_vector_index(driver, "entity_vec", label=ENTITY_LABEL,
                            embedding_property="embedding", dimensions=dim, similarity_fn="cosine")
        create_fulltext_index(driver, "entity_ft", label=ENTITY_LABEL, node_properties=["name", "aliases"])

        for query, describe in (
            ("MATCH (n:Document) RETURN elementId(n) AS eid, n.title + '\\n' + n.text AS t", "documents"),
            ("MATCH (n:__Entity__) RETURN elementId(n) AS eid, "
             "coalesce(n.type,'') + ': ' + coalesce(n.name,'') AS t", "entities"),
        ):
            records, _, _ = driver.execute_query(query)
            upsert_vectors(driver, ids=[r["eid"] for r in records], embedding_property="embedding",
                           embeddings=[embedder.embed_query(r["t"]) for r in records],
                           entity_type=EntityType.NODE)
            print(f"  embedded {len(records)} {describe}")

        deadline = time.time() + 60
        names = ["document_vec", "document_ft", "entity_vec", "entity_ft"]
        while time.time() < deadline:
            recs, _, _ = driver.execute_query(
                "SHOW INDEXES YIELD name, state WHERE name IN $n RETURN collect(state) AS s", n=names)
            if recs[0]["s"].count("ONLINE") == len(names):
                break
            time.sleep(0.3)

        print("\nrestored:", counts(driver))
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
