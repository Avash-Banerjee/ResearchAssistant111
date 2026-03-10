"""
ResearchIQ - Vector Store Manager
===================================
ChromaDB integration for paper embeddings and semantic search.
"""

import logging
from typing import List, Dict, Optional, Any
import chromadb
from sentence_transformers import SentenceTransformer
from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB collections for research paper embeddings."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Fix: use EphemeralClient for simplicity, or properly init PersistentClient
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        except Exception:
            # Fallback to in-memory client if persistent fails
            logger.warning("PersistentClient failed, falling back to EphemeralClient")
            self.client = chromadb.EphemeralClient()

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._initialized = True
        logger.info(f"VectorStore ready. Papers indexed: {self.collection.count()}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        return self.embedding_model.encode(texts, show_progress_bar=False).tolist()

    def add_papers(self, papers: List[Dict]) -> int:
        """Add papers to the vector store. Returns count of newly added papers."""
        new_count = 0
        for paper in papers:
            paper_id = paper.get("id", paper.get("arxiv_id", ""))
            if not paper_id:
                continue

            # Skip if already exists
            try:
                existing = self.collection.get(ids=[paper_id])
                if existing["ids"]:
                    continue
            except Exception:
                pass

            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            embedding = self.embed([text])[0]

            self.collection.add(
                ids=[paper_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "title": paper.get("title", "")[:500],
                    "authors": ", ".join(paper.get("authors", []))[:300],
                    "year": str(paper.get("year", "")),
                    "venue": paper.get("venue", "")[:200],
                    "url": paper.get("url", "")[:300],
                    "citations": str(paper.get("citations", 0)),
                    "categories": ", ".join(paper.get("categories", []))[:200],
                }],
            )
            new_count += 1

        logger.info(f"Added {new_count} new papers. Total: {self.collection.count()}")
        return new_count

    def semantic_search(
        self, query: str, n_results: int = 10, filter_year: Optional[int] = None
    ) -> List[Dict]:
        """Search for semantically similar papers."""
        total = self.collection.count()
        if total == 0:
            return []

        query_embedding = self.embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, total),
            include=["documents", "metadatas", "distances"],
        )

        papers = []
        if results["ids"] and results["ids"][0]:
            for i, pid in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                papers.append({
                    "id": pid,
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year", ""),
                    "venue": meta.get("venue", ""),
                    "url": meta.get("url", ""),
                    "citations": int(meta.get("citations", 0)),
                    "similarity": round(1 - results["distances"][0][i], 4),
                    "abstract": results["documents"][0][i],
                })
        return papers

    def get_all_papers(self, limit: int = 500) -> List[Dict]:
        """Retrieve all indexed papers."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )
        papers = []
        for i, pid in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            papers.append({
                "id": pid,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "year": meta.get("year", ""),
                "venue": meta.get("venue", ""),
                "citations": int(meta.get("citations", 0)),
                "categories": meta.get("categories", ""),
                "abstract": results["documents"][i],
            })
        return papers

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        import numpy as np
        emb1, emb2 = self.embed([text1, text2])
        emb1, emb2 = np.array(emb1), np.array(emb2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def get_stats(self) -> Dict:
        return {"total_papers": self.collection.count()}


def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()
