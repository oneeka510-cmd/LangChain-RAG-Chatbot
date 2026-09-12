from dataclasses import dataclass
import math
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


@dataclass
class ScoredDocument:
    document: Document
    vector_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.hybrid_score


class HybridRetriever:
    """Weighted BM25 + semantic search followed by optional cross-encoder reranking."""

    def __init__(
        self,
        vectorstore: Chroma,
        chunks: list[Document],
        candidate_count: int = 10,
        final_count: int = 4,
        vector_weight: float = 0.65,
        reranker=None,
    ) -> None:
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.candidate_count = candidate_count
        self.final_count = final_count
        self.vector_weight = vector_weight
        self.reranker = reranker
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in chunks])

    @staticmethod
    def _key(document: Document) -> str:
        return str(document.metadata.get("chunk_id", document.page_content[:100]))

    def retrieve(self, query: str) -> list[ScoredDocument]:
        semantic = self.vectorstore.similarity_search_with_score(
            query, k=self.candidate_count
        )
        raw_bm25 = list(self.bm25.get_scores(tokenize(query)))
        normalized_bm25 = minmax(raw_bm25)
        top_keyword = sorted(
            range(len(self.chunks)), key=lambda i: raw_bm25[i], reverse=True
        )[: self.candidate_count]

        combined: dict[str, ScoredDocument] = {}
        for document, distance in semantic:
            key = self._key(document)
            # Chroma returns a distance: smaller is better. This bounded transform
            # avoids embedding-specific negative "relevance" conversions.
            score = 1.0 / (1.0 + max(0.0, float(distance)))
            combined[key] = ScoredDocument(document=document, vector_score=score)
        for index in top_keyword:
            document = self.chunks[index]
            key = self._key(document)
            item = combined.setdefault(key, ScoredDocument(document=document))
            item.keyword_score = float(normalized_bm25[index])

        for item in combined.values():
            item.hybrid_score = (
                self.vector_weight * item.vector_score
                + (1 - self.vector_weight) * item.keyword_score
            )
        candidates = sorted(combined.values(), key=lambda item: item.hybrid_score, reverse=True)

        if self.reranker and candidates:
            pairs = [(query, item.document.page_content) for item in candidates]
            scores = [float(score) for score in self.reranker.predict(pairs)]
            # The default MS MARCO cross-encoder returns logits. Sigmoid preserves
            # absolute relevance, unlike per-query min-max scaling which would make
            # even an irrelevant best candidate appear perfectly confident.
            probabilities = [1 / (1 + math.exp(-max(-30, min(30, score)))) for score in scores]
            for item, score in zip(candidates, probabilities):
                item.rerank_score = 0.75 * score + 0.25 * item.hybrid_score
            candidates.sort(key=lambda item: item.final_score, reverse=True)

        return candidates[: self.final_count]


def confidence(results: list[ScoredDocument]) -> float:
    """Conservative confidence based on the best and supporting retrieval results."""
    if not results:
        return 0.0
    scores = [max(0.0, min(1.0, item.final_score)) for item in results]
    support = sum(scores[:3]) / min(3, len(scores))
    return float(round(0.7 * scores[0] + 0.3 * support, 4))
