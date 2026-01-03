import os

from constants import DEFAULT_K, DEFAULT_SEARCH_LIMIT, SEARCH_MULTIPLIER
from dotenv import load_dotenv
from google import genai
from lib.hybrid_search import HybridSearch
from search_utils import load_movies

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def rag_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    hs = HybridSearch(documents)
    results = hs.rrf_search(query, DEFAULT_K, limit * SEARCH_MULTIPLIER)
    if not results:
        return {
            "query": query,
            "results": [],
            "error": "No results found.",
            "answer": None,
        }
    docs = []
    for res in results[:limit]:
        docs.append(f"{res['title']}: {res['document']}")

    prompt = f"""Hoopla is a streaming service for movies. You are a RAG agent that provides a human answer
    to the user's query based on the documents that were retrieved during search. Provide a comprehensive
    answer that addresses the user's query.

    Query: {query}

    Documents:
    {"\n\n".join(docs)}
    """

    response = client.models.generate_content(model=model, contents=prompt)
    text = (response.text or "").strip()

    return {"query": query, "results": results[:limit], "error": None, "answer": text}


def summarize_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    hs = HybridSearch(documents)
    results = hs.rrf_search(query, DEFAULT_K, limit * SEARCH_MULTIPLIER)
    if not results:
        return {
            "query": query,
            "results": [],
            "error": "No results found.",
            "answer": None,
        }
    docs = []
    for res in results[:limit]:
        docs.append(f"{res['title']}: {res['document']}")

    prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}
    Search Results:
    {"\n\n".join(docs)}
    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """
    response = client.models.generate_content(model=model, contents=prompt)
    text = (response.text or "").strip()

    return {"query": query, "results": results[:limit], "error": None, "answer": text}

def citations_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    hs = HybridSearch(documents)
    results = hs.rrf_search(query, DEFAULT_K, limit * SEARCH_MULTIPLIER)
    if not results:
        return {
            "query": query,
            "results": [],
            "error": "No results found.",
            "answer": None,
        }
    docs = []
    for res in results[:limit]:
        docs.append(f"{res['title']}: {res['document']}")

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{"\n\n".join(docs)}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    text = (response.text or "").strip()

    return {"query": query, "results": results[:limit], "error": None, "answer": text}


def question_command(question: str, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    hs = HybridSearch(documents)
    results = hs.rrf_search(question, DEFAULT_K, limit * SEARCH_MULTIPLIER)
    if not results:
        return {
            "query": question,
            "results": [],
            "error": "No results found.",
            "answer": None,
        }
    docs = []
    for res in results[:limit]:
        docs.append(f"{res['title']}: {res['document']}")

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{"\n\n".join(docs)}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    text = (response.text or "").strip()

    return {"query": question, "results": results[:limit], "error": None, "answer": text}
