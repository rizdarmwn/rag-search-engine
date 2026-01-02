import heapq
from typing import Any
import json
from sympy.combinatorics.fp_groups import rewrite
from google import genai
import os
from dotenv import load_dotenv

def setup_llm():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"
    return client, model


def enhance_query(query:str, method: str) -> str:
    match method:
        case "spell":
            return spell_enhance(query)
        case "rewrite":
            return rewrite_enhance(query)
        case "expand":
            return expand_enhance(query)
        case _:
            return query



def expand_enhance(query: str) -> str:
    client, model = setup_llm()
    expander_query = f"""Expand this movie search query with related terms.
    Strip leading and before whitespaces. Remove any formatting, spit only the text.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    This will be appended to the original query.

    Examples:

    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    Query: "{query}"
    """
    response = client.models.generate_content(model=model, contents=expander_query)
    query = (response.text or "").strip().strip('"')
    return query

def rewrite_enhance(query: str) -> str:
    client, model = setup_llm()
    rewriter_query = f"""Rewrite this movie search query to be more specific and searchable.
    Strip leading and before whitespaces. Remove any formatting, spit only the text.

    Original: "{query}"

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep it concise (under 10 words)
    - It should be a google style search query that's very specific
    - Don't use boolean logic

    Examples:

    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    Rewritten query:"""
    response = client.models.generate_content(model=model, contents=rewriter_query)
    query = (response.text or "").strip().strip('"')
    return query


def spell_enhance(query:str) -> str:
    client, model = setup_llm()
    fix_spelling_query = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""

    response = client.models.generate_content(model=model, contents=fix_spelling_query)
    query = (response.text or "").strip().strip('"')
    return query
