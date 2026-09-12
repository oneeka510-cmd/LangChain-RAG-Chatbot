from langchain_core.documents import Document

from rag.retrieval import ScoredDocument, confidence, minmax, tokenize
from rag.service import cited_sources, clean_document_text, ensure_citations, greeting_response


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


def test_greetings_are_handled_without_retrieval():
    assert greeting_response("hi") == "Hi! Ask me anything about your documents."
    assert greeting_response("Hello there!") == "Hi! Ask me anything about your documents."
    assert greeting_response("What is vector data?") is None


def test_document_formatting_is_removed_from_previews():
    assert clean_document_text("# Remote Sensing\nUseful content") == "Remote Sensing\nUseful content"


def test_only_cited_sources_are_returned():
    sources = [{"citation": 1}, {"citation": 2}, {"citation": 3}]
    assert cited_sources("Supported by [1] and [3].", sources) == [sources[0], sources[2]]


def test_missing_model_citation_falls_back_to_top_source():
    sources = [{"citation": 1}, {"citation": 2}]
    answer, matched = ensure_citations("Grounded answer.", sources)
    assert answer == "Grounded answer. [1]"
    assert matched == [sources[0]]
