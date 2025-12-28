import math
from typing import Any

from constants import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT
from lib.inverted_index import InvertedIndex
from preprocessing import preprocess_text


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(q: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[Any, Any]]:
    inverted_idx = InvertedIndex()
    inverted_idx.load()
    seen, res = set(), []
    query_tokens = preprocess_text(q)
    for token in query_tokens:
        doc_ids = inverted_idx.get_documents(token)
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = inverted_idx.docmap[doc_id]
            res.append(doc)
            if len(res) >= limit:
                return res
    return res


def tf_command(doc_id: int, term: str) -> int:
    inverted_idx = InvertedIndex()
    inverted_idx.load()
    return inverted_idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    inverted_idx = InvertedIndex()
    inverted_idx.load()
    idf = inverted_idx.get_idf(term)
    return idf


def tfidf_command(doc_id: int, term: str) -> float:
    inverted_idx = InvertedIndex()
    inverted_idx.load()
    tfidf = inverted_idx.get_tfidf(doc_id, term)
    return tfidf


def bm25_idf_command(term: str) -> float:
    inverted_idx = InvertedIndex()
    inverted_idx.load()
    bm25_idf = inverted_idx.get_bm25_idf(term)
    return bm25_idf


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    inverted_idx = InvertedIndex()
    inverted_idx.load()
    bm25_tf = inverted_idx.get_bm25_tf(doc_id, term, k1)
    return bm25_tf


def bm25_search_command(query: str):
    inverted_idx = InvertedIndex()
    inverted_idx.load()
    bm25 = inverted_idx.bm25_search(query)
    for i, doc_id in enumerate(bm25):
        title = inverted_idx.docmap[doc_id]["title"]
        print(f"{i + 1}. ({doc_id}) {title} - Score: {bm25[doc_id]:.2f}")
