from typing import Optional
import heapq
import os

from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

from constants import INDEX_CACHE_PATH, SEARCH_MULTIPLIER
from search_enhancement import enhance_query
from reranking import rerank
from search_utils import load_movies, format_search_result
from custom_types import RRFSearchResult, SearchResult


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search: ChunkedSemanticSearch = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_CACHE_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5) -> list[SearchResult]:
        kw_search = self._bm25_search(query, limit * 500)
        sm_search = self.semantic_search.search_chunks(query, limit * 500)

        kw_scores = list(map(lambda x: x["score"], kw_search))
        sm_scores = list(map(lambda x: x["score"], sm_search))

        kw_norm = normalize_command(kw_scores)
        sm_norm = normalize_command(sm_scores)

        document_map = {}
        for i, res in enumerate(kw_search):
            id = res["id"]
            title = res["title"]
            description = res["document"]
            if id not in document_map:
                document_map[id] = {
                    "id": id,
                    "title": title,
                    "document": description,
                    "kw_score": 0.0,
                    "sm_score": 0.0
                }
            if kw_norm[i] > document_map[id]["kw_score"]:
                document_map[id]["kw_score"] = kw_norm[i]
        for i, res in enumerate(sm_search):
            id = res["id"]
            title = res["title"]
            description = res["document"]
            if id not in document_map:
                document_map[id] = {
                    "id": id,
                    "title": title,
                    "document": description,
                    "kw_score": 0.0,
                    "sm_score": 0.0
                }
            if sm_norm[i] > document_map[id]["sm_score"]:
                document_map[id]["sm_score"] = sm_norm[i]

        for k in document_map:
            kw_score = document_map[k]["kw_score"]
            sm_score = document_map[k]["sm_score"]
            document_map[k]["hybrid_score"] = hybrid_score(kw_score, sm_score, alpha)

        sorted_scores = heapq.nlargest(limit, document_map.items(), key=lambda x: x[1]["hybrid_score"])

        return list(map(lambda x: format_search_result(x[0], x[1]["title"], x[1]["document"], x[1]["hybrid_score"], kw_score=x[1]["kw_score"], sm_score=x[1]["sm_score"], hybrid_score=x[1]["hybrid_score"]), sorted_scores))


    def rrf_search(self, query, k, limit=10) -> list[SearchResult]:
        kw_search = self._bm25_search(query, limit * 500)
        sm_search = self.semantic_search.search_chunks(query, limit * 500)

        document_map = {}
        for i, res in enumerate(kw_search, 1):
            id = res["id"]
            title = res["title"]
            description = res["document"]
            if id not in document_map:
                document_map[id] = {
                    "id": id,
                    "title": title,
                    "document": description,
                    "rrf_score": 0.0,
                    "kw_rank": None,
                    "sm_rank": None,
                }
            if document_map[id]["kw_rank"] is None:
                document_map[id]["kw_rank"] = i
                document_map[id]["rrf_score"] += rrf_score(i, k)
        for i, res in enumerate(sm_search, 1):
            id = res["id"]
            title = res["title"]
            description = res["document"]
            if id not in document_map:
                document_map[id] = {
                    "id": id,
                    "title": title,
                    "document": description,
                    "rrf_score": 0.0,
                    "kw_rank": None,
                    "sm_rank": None,
                }
            if document_map[id]["sm_rank"] is None:
                document_map[id]["sm_rank"] = i
                document_map[id]["rrf_score"] += rrf_score(i, k)

        sorted_scores = heapq.nlargest(limit, document_map.items(), key=lambda x: x[1]["rrf_score"])

        results: list[SearchResult] = list(map(lambda x: format_search_result(x[0], x[1]["title"], x[1]["document"], x[1]["rrf_score"], rrf_score=x[1]["rrf_score"], kw_rank=x[1]["kw_rank"], sm_rank=x[1]["sm_rank"]), sorted_scores))

        return results

def normalize_command(scores: list[float | int] = []):
    if len(scores) == 0:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)

    normalized_scores = list(map(lambda x: (x - min_score) / (max_score - min_score), scores))
    return normalized_scores

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query, alpha: float=0.5, limit: int=5):
    documents = load_movies()
    hs = HybridSearch(documents)
    return hs.weighted_search(query, alpha, limit)

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def rrf_search_command(query:str, enhance: Optional[str] = None, rerank_method: Optional[str] = None, k:int = 60, limit:int=5) -> RRFSearchResult:
    documents = load_movies()
    hs = HybridSearch(documents)

    org_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query

    search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
    results = hs.rrf_search(query, k, search_limit)

    reranked = False
    if rerank_method:
        results = rerank(query, results, method=rerank_method, limit=limit)
        reranked = True

    res: RRFSearchResult = {
        "original_query": org_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "reranked": reranked,
        "results": results,
    }

    return res
