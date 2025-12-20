from typing import Any

from inverted_index import InvertedIndex
from preprocessing import preprocess_text
from search_utils import DEFAULT_SEARCH_LIMIT, load_movies


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
