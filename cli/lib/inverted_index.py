import heapq
import math
import os
import pickle
from collections import Counter, defaultdict
from typing import Any

from constants import (
    BM25_B,
    BM25_K1,
    CACHE_PATH,
    DEFAULT_SEARCH_LIMIT,
    DOC_LENGTHS_CACHE_PATH,
    DOCMAP_CACHE_PATH,
    INDEX_CACHE_PATH,
    TFREQ_CACHE_PATH,
)
from preprocessing import preprocess_text
from search_utils import load_movies


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[Any, Any]] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str):
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        tokens = preprocess_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = set((doc_id,))
        self.term_frequencies[doc_id].update(tokens)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        total_doc_length = 0
        for doc_id in self.doc_lengths:
            total_doc_length += self.doc_lengths[doc_id]

        return total_doc_length / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        token = term.lower()
        doc_ids = self.index.get(token, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str):
        token = preprocess_text(term)
        if len(token) > 1:
            raise Exception("token is more than 1")

        return self.term_frequencies[doc_id][token[0]]

    def get_idf(self, term: str) -> float:
        tokens = preprocess_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tokens = preprocess_text(term)
        tfidfs = []
        for token in tokens:
            tf = self.get_tf(doc_id, token)
            idf = self.get_idf(token)
            tfidfs.append(tf * idf)

        return sum(tfidfs)

    def get_bm25_idf(self, term: str) -> float:
        tokens = preprocess_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + (b * (doc_length / avg_doc_length))
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        tokens = preprocess_text(query)
        scores = {}
        for doc_id in self.docmap:
            for token in tokens:
                bm25 = self.bm25(doc_id, token)
                if doc_id in scores:
                    scores[doc_id] += bm25
                else:
                    scores[doc_id] = bm25

        return dict(heapq.nlargest(limit, scores.items(), key=lambda x: x[1]))

    def build(self):
        movies = load_movies()
        for movie in movies:
            input_text = f"{movie['title']} {movie['description']}"
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], input_text)

    def save(self):
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
        with open(INDEX_CACHE_PATH, "wb") as idx_cache:
            pickle.dump(self.index, idx_cache)
        with open(DOCMAP_CACHE_PATH, "wb") as docmap_cache:
            pickle.dump(self.docmap, docmap_cache)
        with open(TFREQ_CACHE_PATH, "wb") as tfreq_cache:
            pickle.dump(self.term_frequencies, tfreq_cache)
        with open(DOC_LENGTHS_CACHE_PATH, "wb") as doc_len_cache:
            pickle.dump(self.doc_lengths, doc_len_cache)

        idx_cache.close()
        docmap_cache.close()
        tfreq_cache.close()
        doc_len_cache.close()

    def load(self):
        if (
            not os.path.exists(INDEX_CACHE_PATH)
            or not os.path.exists(DOCMAP_CACHE_PATH)
            or not os.path.exists(TFREQ_CACHE_PATH)
            or not os.path.exists(DOC_LENGTHS_CACHE_PATH)
        ):
            raise FileNotFoundError("File not found on cache")

        with open(INDEX_CACHE_PATH, "rb") as idx_cache:
            self.index = pickle.load(idx_cache)
        with open(DOCMAP_CACHE_PATH, "rb") as docmap_cache:
            self.docmap = pickle.load(docmap_cache)
        with open(TFREQ_CACHE_PATH, "rb") as tfreq_cache:
            self.term_frequencies = pickle.load(tfreq_cache)
        with open(DOC_LENGTHS_CACHE_PATH, "rb") as doc_len_cache:
            self.doc_lengths = pickle.load(doc_len_cache)

        idx_cache.close()
        docmap_cache.close()
        tfreq_cache.close()
        doc_len_cache.close()
