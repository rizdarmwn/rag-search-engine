import heapq
import json
import os
from typing import Any
from time import sleep

from dotenv import load_dotenv
from google import genai
from constants import DEFAULT_SEARCH_LIMIT

from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

def individual_rerank(query: str, docs: list[dict], limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    scored_docs = []
    for doc in docs:
        individual_reranker = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.

        Score:"""
        response = client.models.generate_content(model=model, contents=individual_reranker)
        score_text = (response.text or "").strip().strip('"')
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    return heapq.nlargest(limit, scored_docs, key=lambda x: x["individual_score"])

def batch_rerank(query: str, docs: list[dict[str, Any]], limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    if not docs:
        return []

    doc_map = {}
    doc_list = []
    for doc in docs:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)

    batch_reranker = f"""Rank these movies by relevance to the search query.

    Query: "{query}"

    Movies:
    {docs}

    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else.
    Remove any text formatting like ```json or anything else. Just the list. For example:

    [75, 12, 34, 2, 1]
    """
    response = client.models.generate_content(model=model, contents=batch_reranker)
    ranking_text = (response.text or "").strip()

    parsed_ids = json.loads(ranking_text)

    reranked = []
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            reranked.append({**doc_map[doc_id], "batch_rank": i + 1})

    return reranked[:limit]

def cross_encoder_rerank(query: str, docs: list[dict], limit: int=5) -> list[dict]:
    pairs = []
    doc_list = []
    for doc in docs:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
    ce = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = ce.predict(pairs)
    for i, doc in enumerate(docs):
        doc_list.append({**doc, "cross_encoder_score": scores[i]})
    return heapq.nlargest(limit, doc_list, key=lambda x: x["cross_encoder_score"])

def rerank(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    if method == "individual":
        return individual_rerank(query, documents, limit)
    if method == "batch":
        return batch_rerank(query, documents, limit)
    if method == "cross_encoder":
        return cross_encoder_rerank(query, documents, limit)
    else:
        return documents[:limit]
