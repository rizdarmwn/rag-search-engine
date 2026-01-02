from typing import TypedDict, Any, Optional

class SearchResult(TypedDict):
    id: str
    title: str
    document: str
    score: float
    metadata: dict[str, Any]


class RRFSearchResult(TypedDict):
    original_query: str
    enhanced_query: Optional[str]
    enhance_method: Optional[str]
    query: str
    k: int
    rerank_method: Optional[str]
    reranked: bool
    results: list[dict] | list[SearchResult]
