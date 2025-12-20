import string

from nltk.stem import PorterStemmer
from search_utils import load_stopwords


def preprocess_text(s: str) -> list[str]:
    text = s.lower()
    translator = str.maketrans("", "", string.punctuation)

    translated = text.translate(translator)
    raw_tokens = tokenization(translated)
    rsw_tokens = remove_stopwords(raw_tokens)
    stemmed_tokens = stemming(rsw_tokens)
    return stemmed_tokens


def tokenization(s: str) -> list[str]:
    tokens = list(filter(lambda x: x != "", s.split()))
    return tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    stopwords = load_stopwords()
    return list(filter(lambda x: x not in stopwords, tokens))


def stemming(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens
