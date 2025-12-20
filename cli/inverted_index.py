import os
import pickle
from collections import Counter, defaultdict
from typing import Any

from preprocessing import preprocess_text
from search_utils import PROJECT_ROOT, load_movies

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_CACHE_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_CACHE_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
TFREQ_CACHE_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[Any, Any]] = {}
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str):
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        tokens = preprocess_text(text)
        for token in set(tokens):
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = set((doc_id,))
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        token = term.lower()
        doc_ids = self.index.get(token, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str):
        token = preprocess_text(term)
        if len(token) > 1:
            raise Exception("token is more than 1")

        return self.term_frequencies[doc_id][token[0]]

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
        with open(TFREQ_CACHE_PATH, "wb") as tfreq_cache:
            pickle.dump(self.term_frequencies, tfreq_cache)

        idx_cache.close()
        docmap_cache.close()
        tfreq_cache.close()

    def load(self):
        if not os.path.exists(INDEX_CACHE_PATH) or not os.path.exists(
            DOCMAP_CACHE_PATH
        ):
            raise FileNotFoundError("File not found on cache")

        with open(INDEX_CACHE_PATH, "rb") as idx_cache:
            self.index = pickle.load(idx_cache)

        with open(DOCMAP_CACHE_PATH, "rb") as docmap_cache:
            self.docmap = pickle.load(docmap_cache)

        with open(TFREQ_CACHE_PATH, "rb") as tfreq_cache:
            self.term_frequencies = pickle.load(tfreq_cache)

        idx_cache.close()
        docmap_cache.close()
        tfreq_cache.close()
