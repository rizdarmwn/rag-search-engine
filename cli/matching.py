from typing import Any

from inverted_index import InvertedIndex
from preprocessing import preprocess_text
from search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def has_matching_token(query_tokens, title_tokens):
    for qt in query_tokens:
        for tt in title_tokens:
            if qt in tt:
                return True
    return False


def matching_title(s: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[Any, Any]]:
    inverted_idx = InvertedIndex()
    inverted_idx.load()
    seen, res = set(), []
    query_tokens = preprocess_text(s)
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
