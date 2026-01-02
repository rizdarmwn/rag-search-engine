import os

DEFAULT_ALPHA = 0.5
SCORE_PRECISION = 3
SEARCH_MULTIPLIER = 5

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_CACHE_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_CACHE_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
TFREQ_CACHE_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")
DOC_LENGTHS_CACHE_PATH = os.path.join(CACHE_PATH, "doc_lengths.pkl")
MOVIE_EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_PATH, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
CHUNK_METADATA_CACHE_PATH = os.path.join(CACHE_PATH, "chunk_metadata.json")

DEFAULT_SEARCH_LIMIT = 5
