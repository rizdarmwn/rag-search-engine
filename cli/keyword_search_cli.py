#!/usr/bin/env python3

import argparse
import string

from nltk.stem import PorterStemmer
from search_utils import load_movies, load_stopwords


def preprocess_text(s):
    text = s.lower()
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_stopwords(tokens):
    stopwords = load_stopwords()
    return list(filter(lambda x: x not in stopwords, tokens))


def stemming(tokens: list[str]):
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens


def tokenization(s):
    text = preprocess_text(s)
    tokens = list(filter(lambda x: x != "", text.split(" ")))
    removed_sw = remove_stopwords(tokens)
    stemmed_tokens = stemming(removed_sw)
    return stemmed_tokens


def has_matching_token(query_tokens, title_tokens):
    for qt in query_tokens:
        for tt in title_tokens:
            if qt in tt:
                return True
    return False


def matching(s):
    movies = load_movies()
    res = []
    query_tokens = tokenization(s)
    for movie in movies:
        title_tokens = tokenization(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            res.append(movie)
            if len(res) >= 5:
                break

    res = sorted(res, key=lambda x: x["id"])

    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            res = matching(args.query)
            for m in range(len(res)):
                print(f"{m + 1}. {res[m]['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
