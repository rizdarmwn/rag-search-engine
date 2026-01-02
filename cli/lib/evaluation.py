from search_utils import load_golden_dataset
from lib.hybrid_search import rrf_search_command
from dotenv import load_dotenv
from google import genai
import json
import os
from custom_types import SearchResult


def precision_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k

def recall_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / len(relevant_docs)

def f1_score(precision: float, recall:float):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def evaluation_command(limit: int = 5):
    golden_data = load_golden_dataset()
    results_by_query = {}
    for tc in golden_data["test_cases"]:
        query = tc["query"]
        results = rrf_search_command(query, k=60, limit=limit)
        relevant_docs = set(tc["relevant_docs"])
        retrieved_docs = list(map(lambda x: x["title"], results["results"]))
        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)
        f1 = f1_score(precision, recall)

        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs)
        }

    return {
        "test_cases_count": len(golden_data["test_cases"]),
        "limit": limit,
        "results": results_by_query
    }

def llm_judge_evaluate(query: str, results: list[SearchResult] | list[dict]):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"

    formatted_results = list(map(lambda x: f"{x["title"]}", results))
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers out than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""
    response = client.models.generate_content(model=model, contents=prompt)
    scores_text = (response.text or "").strip()

    parsed_scores = json.loads(scores_text)

    scores = []
    for i, score in enumerate(parsed_scores, 0):
        scores.append({
            "title": results[i]["title"],
            "score": score
        })

    return scores
