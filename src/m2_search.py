from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
import re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except ImportError:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(c["text"]).lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        tokens = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokens)
        if not re.search(r"\b(202[0-9]|v[0-9]+|cũ|cu|phiên bản trước)\b", query.lower()):
            versions = {}
            for i, document in enumerate(self.documents):
                source = document.get("metadata", {}).get("source", "")
                match = re.match(r"(.+)_v(\d+)(?:\.md)?$", source)
                if match:
                    family, version = match.group(1), int(match.group(2))
                    versions[family] = max(versions.get(family, version), version)
            highest = max((float(score) for score in scores if score > 0), default=0.0)
            for i, document in enumerate(self.documents):
                source = document.get("metadata", {}).get("source", "")
                match = re.match(r"(.+)_v(\d+)(?:\.md)?$", source)
                if match and int(match.group(2)) == versions[match.group(1)] and scores[i] > 0:
                    scores[i] += highest * 0.35
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [SearchResult(self.documents[i]["text"], float(scores[i]),
                             self.documents[i].get("metadata", {}), "bm25")
                for i in indices if scores[i] > 0][:top_k]


class DenseSearch:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            try:
                from qdrant_client import QdrantClient
                self.client = QdrantClient(":memory:")
            except ImportError:
                self.client = None
        self._encoder = None
        self._indexed = False

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if self.client is None or not chunks:
            return
        from qdrant_client.models import Distance, PointStruct, VectorParams
        try:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            vectors = self._get_encoder().encode([c["text"] for c in chunks], show_progress_bar=True)
            points = [PointStruct(id=i, vector=vector.tolist(),
                                  payload={**chunk.get("metadata", {}), "text": chunk["text"]})
                      for i, (chunk, vector) in enumerate(zip(chunks, vectors))]
            self.client.upsert(collection_name=collection, points=points)
            self._indexed = True
        except Exception as exc:
            print(f"  ⚠️  Dense indexing unavailable: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self.client is None or not self._indexed:
            return []
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(collection_name=collection,
                                                query=query_vector, limit=top_k)
        except Exception:
            return []
        return [SearchResult(pt.payload.get("text", ""), float(pt.score),
                             pt.payload or {}, "dense") for pt in response.points]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    scores = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in scores:
                scores[result.text] = [0.0, result]
            scores[result.text][0] += 1.0 / (k + rank + 1)
    ranked = sorted(scores.values(), key=lambda pair: pair[0], reverse=True)[:top_k]
    return [SearchResult(result.text, score, result.metadata, "hybrid")
            for score, result in ranked]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
