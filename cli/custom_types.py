from typing import TypedDict, Any

class SearchResult(TypedDict):
    id: str
    title: str
    document: str
    score: float
    metadata: dict[str, Any]
