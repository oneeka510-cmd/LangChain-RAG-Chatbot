from langchain_core.documents import Document

from rag.retrieval import ScoredDocument, confidence, minmax, tokenize


def test_tokenize_normalizes_words_and_numbers():
    assert tokenize("Hybrid Search, BM25!") == ["hybrid", "search", "bm25"]


def test_minmax_handles_equal_and_varied_scores():
    assert minmax([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert minmax([0.0, 0.0]) == [0.0, 0.0]


def test_confidence_uses_top_score_and_support():
    results = [
        ScoredDocument(Document(page_content="a"), hybrid_score=0.8),
        ScoredDocument(Document(page_content="b"), hybrid_score=0.5),
        ScoredDocument(Document(page_content="c"), hybrid_score=0.2),
    ]
    assert confidence(results) == 0.71
    assert confidence([]) == 0.0

