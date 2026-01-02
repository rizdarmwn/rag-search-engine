from search_utils import load_movies
from lib.hybrid_search import HybridSearch
from constants import DEFAULT_K, DEFAULT_SEARCH_LIMIT, SEARCH_MULTIPLIER
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

def rag_command(query:str, limit:int =  DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    hs = HybridSearch(documents)
    results = hs.rrf_search(query, DEFAULT_K, limit * SEARCH_MULTIPLIER)
    if not results:
        return {
            "query": query,
            "results": [],
            "error": "No results found.",
            "answer": None
        }
    docs = []
    for res in results[:limit]:
        docs.append(f"{res["title"]}: {res["document"]}")

    prompt = f"""Hoopla is a streaming service for movies. You are a RAG agent that provides a human answer
    to the user's query based on the documents that were retrieved during search. Provide a comprehensive
    answer that addresses the user's query.

    Query: {query}

    Documents:
    {"\n\n".join(docs)}
    """

    response = client.models.generate_content(model=model, contents=prompt)
    text = (response.text or "").strip()

    return {
        "query": query,
        "results": results[:limit],
        "error": None,
        "answer": text
    }
