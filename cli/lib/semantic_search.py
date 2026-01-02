import json
import os
import re
import heapq
from typing import Any

import numpy as np
from constants import MOVIE_EMBEDDINGS_CACHE_PATH, CHUNK_EMBEDDINGS_CACHE_PATH, CHUNK_METADATA_CACHE_PATH
from search_utils import load_movies, format_search_result
from sentence_transformers import SentenceTransformer
from custom_types import SearchResult


class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("text contains only empty string or whitespace")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[dict[Any, Any]]):
        self.documents = documents
        repr = []
        for document in documents:
            self.document_map[document["id"]] = document
            repr.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(repr, show_progress_bar=True)
        with open(MOVIE_EMBEDDINGS_CACHE_PATH, "wb") as m_embeddings:
            np.save(m_embeddings, self.embeddings)

        m_embeddings.close()
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict[Any, Any]]):
        if not bool(self.documents) :
            self.documents = documents
        if not bool(self.document_map):
            for document in documents:
                self.document_map[document["id"]] = document

        if os.path.exists(MOVIE_EMBEDDINGS_CACHE_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_CACHE_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit) -> list[SearchResult]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first")
        embedding = self.generate_embedding(query)
        sim_scores = []
        for i, document in enumerate(self.documents):
            sim_scores.append((cosine_similarity(embedding, self.embeddings[i]), document))

        sorted_scores = heapq.nlargest(limit, sim_scores, lambda x: x[0])

        return list(map(lambda x: format_search_result(x[1]["id"], x[1]["title"], x[1]["description"], x[0]), sorted_scores))

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        repr = []
        for document in documents:
            self.document_map[document["id"]] = document
        chunks = []
        meta_chunks = []
        for i, document in enumerate(self.documents):
            if not document["description"]:
                continue
            sem_chunks = semantic_chunk_command(document["description"], 4, 1)
            chunks.extend(sem_chunks)
            for j, chunk in enumerate(sem_chunks):
                meta_chunks.append({"movie_idx": i, "chunk_idx": j, "total_chunks": len(sem_chunks)})

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = meta_chunks

        with open(CHUNK_EMBEDDINGS_CACHE_PATH, "wb") as c_embeddings:
            np.save(c_embeddings, self.chunk_embeddings)

        with open(CHUNK_METADATA_CACHE_PATH, "w") as c_metadata:
            json.dump({"chunks": meta_chunks, "total_chunks": len(chunks)}, c_metadata, indent=2)

        c_embeddings.close()
        c_metadata.close()

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if not bool(self.documents) :
            self.documents = documents
        if not bool(self.document_map):
            for document in documents:
                self.document_map[document["id"]] = document

        if os.path.exists(CHUNK_EMBEDDINGS_CACHE_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_CACHE_PATH)

        if os.path.exists(CHUNK_METADATA_CACHE_PATH):
            with open(CHUNK_METADATA_CACHE_PATH, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]

        return self.chunk_embeddings if self.chunk_embeddings is not None else self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[SearchResult]:
        embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk_embed in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(embedding, chunk_embed)
            metadata = self.chunk_metadata[i]
            movie_idx = metadata["movie_idx"]
            chunk_idx = metadata["chunk_idx"]
            chunk_scores.append({"chunk_idx": chunk_idx, "movie_idx": movie_idx, "score": similarity})
        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in movie_scores or movie_scores[movie_idx]< score:
                movie_scores[movie_idx] = score

        sorted_scores = heapq.nlargest(limit, movie_scores.items(), key=lambda x: x[1])

        return list(map(lambda x: format_search_result(self.documents[x[0]]["id"], self.documents[x[0]]["title"], self.documents[x[0]]["description"], x[1]), sorted_scores))


def search_chunked_command(query: str, limit: int=5):
    ss = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_chunk_embeddings(documents)
    results = ss.search_chunks(query, limit)
    for i, res in enumerate(results, 1):
        title = res["title"]
        score = res["score"]
        description = res["document"]
        print(f"\n{i}. {title} (score: {score:.4f})")
        print(f"   {description}...")

def embed_chunks_command():
    ss = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def verify_model():
    ss = SemanticSearch()
    model = ss.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def search_command(query: str, limit: int=5):
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    resp = ss.search(query, limit)
    for i, r in enumerate(resp, 1):
        print(f"{i}. {r['title']} (score: {r['score']:.4f})")
        print(f"{r['description'][:100]}...\n")

def chunk_command(text: str, chunk_size: int=200, overlap: int=20):
    text_lst = text.split()
    chunks = []
    i = 0
    while len(text_lst) > i:
        chunk_sentence = text_lst[i : i + chunk_size]
        if chunks and len(chunk_sentence) <= overlap:
            break
        chunks.append(" ".join(chunk_sentence))
        i += chunk_size - overlap

    print(f"Chunking {len(text)} characters")
    for i in range(len(chunks)):
        print(f"{i+1}. {chunks[i]}")

def semantic_chunk_command(text: str, max_chunk_size: int=4, overlap: int = 0):
    text = text.strip()
    if not text:
        return []
    text_lst = re.split(r"(?<=[.!?])\s+", text)
    if len(text_lst) == 1 and text_lst[0].endswith((".", "!", "?")):
        return [text_lst[0].strip()]

    chunks = []
    i = 0
    while len(text_lst) > i:
        chunk_sentence = text_lst[i : i + max_chunk_size]
        if chunks and len(chunk_sentence) <= overlap:
            break
        chunk_sentence = list(filter(lambda x: x.strip() != "", chunk_sentence))
        chunks.append(" ".join(chunk_sentence))
        i += max_chunk_size - overlap
    return chunks


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec2)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
