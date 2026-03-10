"""
ResearchIQ - Vector Store Manager (v2 - True RAG)
===================================================
ChromaDB with two collections:
  1. papers     — whole-paper embeddings (title + abstract) for similarity search
  2. rag_chunks — chunked text segments for true RAG retrieval

RAG pipeline:
  add_papers()      → indexes whole paper + splits into overlapping chunks
  rag_retrieve()    → embeds query → retrieves top-K chunks → returns formatted
                       context string ready to inject directly into LLM prompts
"""

import logging
from typing import List, Dict, Optional
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

CHUNK_COLLECTION = "rag_chunks"
CHUNK_SIZE       = 400   # words per chunk
CHUNK_OVERLAP    = 80    # word overlap between adjacent chunks


class VectorStoreManager:
    """
    Two ChromaDB collections:
      papers     : one vector per paper  (title + abstract)
      rag_chunks : overlapping word-window chunks for RAG retrieval
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        try:
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        except Exception:
            logger.warning("PersistentClient failed - falling back to EphemeralClient")
            self.client = chromadb.EphemeralClient()

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.chunk_collection = self.client.get_or_create_collection(
            name=CHUNK_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        self._initialized = True
        logger.info(
            f"VectorStore ready - papers: {self.collection.count()}, "
            f"chunks: {self.chunk_collection.count()}"
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(texts, show_progress_bar=False).tolist()

    def add_papers(self, papers: List[Dict]) -> int:
        new_count = 0
        for paper in papers:
            paper_id = paper.get("id", paper.get("arxiv_id", ""))
            if not paper_id:
                continue
            try:
                if self.collection.get(ids=[paper_id])["ids"]:
                    continue
            except Exception:
                pass

            title    = paper.get("title", "")
            abstract = paper.get("abstract", "")
            text     = f"{title} {abstract}"
            embedding = self.embed([text])[0]

            self.collection.add(
                ids=[paper_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "title":      title[:500],
                    "authors":    ", ".join(paper.get("authors", []))[:300],
                    "year":       str(paper.get("year", "")),
                    "venue":      paper.get("venue", "")[:200],
                    "url":        paper.get("url", "")[:300],
                    "citations":  str(paper.get("citations", 0)),
                    "categories": ", ".join(paper.get("categories", []))[:200],
                }],
            )
            self._index_chunks(paper, paper_id)
            new_count += 1

        logger.info(
            f"Added {new_count} papers. "
            f"Total - papers: {self.collection.count()}, chunks: {self.chunk_collection.count()}"
        )
        return new_count

    def _index_chunks(self, paper: Dict, paper_id: str):
        title     = paper.get("title", "")
        abstract  = paper.get("abstract", "")
        full_text = paper.get("full_text", "")

        if full_text:
            source_text = f"Title: {title}\n\nAbstract: {abstract}\n\n{full_text}"
        else:
            source_text = (
                f"Title: {title}\n\n"
                f"Abstract: {abstract}\n\n"
                f"Summary: {abstract}\n\n"
                f"Research focus: {title}. {abstract}"
            )

        for i, chunk in enumerate(self._chunk_text(source_text)):
            chunk_id = f"{paper_id}__chunk_{i}"
            try:
                if self.chunk_collection.get(ids=[chunk_id])["ids"]:
                    continue
            except Exception:
                pass
            emb = self.embed([chunk])[0]
            self.chunk_collection.add(
                ids=[chunk_id],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{
                    "paper_id":    paper_id,
                    "paper_title": title[:300],
                    "year":        str(paper.get("year", "")),
                    "chunk_index": str(i),
                    "url":         paper.get("url", "")[:300],
                }],
            )

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks, start = [], 0
        while start < len(words):
            end   = min(start + CHUNK_SIZE, len(words))
            chunk = " ".join(words[start:end]).strip()
            if len(chunk) > 80:
                chunks.append(chunk)
            if end == len(words):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def rag_retrieve(self, query: str, n_chunks: int = 10, max_per_paper: int = 2) -> str:
        """
        True RAG retrieval. Returns a formatted context string for LLM injection.

        Usage:
            context = vs.rag_retrieve("attention mechanisms efficiency")
            prompt  = f"CONTEXT FROM LITERATURE:\\n{context}\\n\\nQUESTION: {q}"
            answer  = llm.generate(prompt)
        """
        total = self.chunk_collection.count()
        if total == 0:
            return ""

        query_emb = self.embed([query])[0]
        results   = self.chunk_collection.query(
            query_embeddings=[query_emb],
            n_results=min(n_chunks * 2, total),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return ""

        parts: List[str] = []
        paper_counts: Dict[str, int] = {}

        for i, chunk_id in enumerate(results["ids"][0]):
            meta      = results["metadatas"][0][i]
            text      = results["documents"][0][i]
            paper_id  = meta.get("paper_id", chunk_id)
            title     = meta.get("paper_title", "Unknown")
            year      = meta.get("year", "")
            relevance = round(1 - results["distances"][0][i], 3)

            count = paper_counts.get(paper_id, 0)
            if count >= max_per_paper:
                continue
            paper_counts[paper_id] = count + 1

            parts.append(
                f'[Source: "{title}" ({year}) | relevance: {relevance:.2f}]\n{text}'
            )
            if len(parts) >= n_chunks:
                break

        return "\n\n---\n\n".join(parts)

    def rag_retrieve_structured(self, query: str, n_chunks: int = 8) -> List[Dict]:
        total = self.chunk_collection.count()
        if total == 0:
            return []
        query_emb = self.embed([query])[0]
        results   = self.chunk_collection.query(
            query_embeddings=[query_emb],
            n_results=min(n_chunks, total),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                chunks.append({
                    "chunk_id":    cid,
                    "text":        results["documents"][0][i],
                    "paper_title": meta.get("paper_title", ""),
                    "year":        meta.get("year", ""),
                    "url":         meta.get("url", ""),
                    "relevance":   round(1 - results["distances"][0][i], 3),
                })
        return chunks

    def semantic_search(self, query: str, n_results: int = 10, filter_year: Optional[int] = None) -> List[Dict]:
        total = self.collection.count()
        if total == 0:
            return []
        query_emb = self.embed([query])[0]
        results   = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(n_results, total),
            include=["documents", "metadatas", "distances"],
        )
        papers = []
        if results["ids"] and results["ids"][0]:
            for i, pid in enumerate(results["ids"][0]):
                meta     = results["metadatas"][0][i]
                year_val = meta.get("year", "")
                if filter_year and year_val:
                    try:
                        if int(year_val) < filter_year:
                            continue
                    except ValueError:
                        pass
                papers.append({
                    "id":         pid,
                    "title":      meta.get("title", ""),
                    "authors":    meta.get("authors", ""),
                    "year":       year_val,
                    "venue":      meta.get("venue", ""),
                    "url":        meta.get("url", ""),
                    "citations":  int(meta.get("citations", 0) or 0),
                    "similarity": round(1 - results["distances"][0][i], 4),
                    "abstract":   results["documents"][0][i],
                })
        return papers

    def get_all_papers(self, limit: int = 500) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        results = self.collection.get(limit=limit, include=["documents", "metadatas"])
        papers  = []
        for i, pid in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            papers.append({
                "id":         pid,
                "title":      meta.get("title", ""),
                "authors":    meta.get("authors", ""),
                "year":       meta.get("year", ""),
                "venue":      meta.get("venue", ""),
                "citations":  int(meta.get("citations", 0) or 0),
                "categories": meta.get("categories", ""),
                "abstract":   results["documents"][i],
            })
        return papers

    def compute_similarity(self, text1: str, text2: str) -> float:
        a, b = np.array(self.embed([text1, text2]))
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def get_stats(self) -> Dict:
        return {
            "total_papers": self.collection.count(),
            "total_chunks": self.chunk_collection.count(),
        }


def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()
