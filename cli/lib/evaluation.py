from search_utils import load_golden_dataset
from hybrid_search import rrf_search_command

def precision_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k

def precision_command(limit: int = 5):
    golden_data = load_golden_dataset()
    results_by_query = {}
    for tc in golden_data["test_cases"]:
        query = tc["query"]
        results = rrf_search_command(query, k=60, limit=limit)
        relevant_docs = set(tc["relevant_docs"])
        retrieved_docs = list(map(lambda x: x["title"], results["results"]))
        precision = precision_at_k(retrieved_docs, relevant_docs, limit)

        results_by_query[query] = {
            "precision": precision,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs)
        }

    return {
        "test_cases_count": len(golden_data["test_cases"]),
        "limit": limit,
        "results": results_by_query
    }
