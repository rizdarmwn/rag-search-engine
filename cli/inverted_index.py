import os
import pickle
from collections import defaultdict
from typing import Any

from preprocessing import preprocess_text
from search_utils import PROJECT_ROOT, load_movies

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_CACHE_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_CACHE_PATH = os.path.join(CACHE_PATH, "docmap.pkl")


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[Any, Any]] = {}

    def __add_document(self, doc_id, text):
        tokens = preprocess_text(text)
        for token in set(tokens):
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = set((doc_id,))

    def get_documents(self, term: str) -> list[int]:
        token = term.lower()
        doc_ids = self.index.get(token, set())
        return sorted(list(doc_ids))

    def build(self):
        movies = load_movies()
        for movie in movies:
            input_text = ""
            if "title" in movie:
                input_text = f"{input_text} {movie['title']}"
            if "description" in movie:
                input_text = f"{input_text} {movie['description']}"
            input_text = input_text.strip()
            self.__add_document(movie["id"], input_text)
            self.docmap[movie["id"]] = movie

    def save(self):
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
        with open(INDEX_CACHE_PATH, "wb") as idx_cache:
            pickle.dump(self.index, idx_cache)
        with open(DOCMAP_CACHE_PATH, "wb") as docmap_cache:
            pickle.dump(self.docmap, docmap_cache)

        idx_cache.close()
        docmap_cache.close()

    def load(self):
        if not os.path.exists(INDEX_CACHE_PATH) or not os.path.exists(
            DOCMAP_CACHE_PATH
        ):
            raise FileNotFoundError("File not found on cache")

        with open(INDEX_CACHE_PATH, "rb") as idx_cache:
            self.index = pickle.load(idx_cache)

        with open(DOCMAP_CACHE_PATH, "rb") as docmap_cache:
            self.docmap = pickle.load(docmap_cache)

        idx_cache.close()
        docmap_cache.close()
