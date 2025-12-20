#!/usr/bin/env python3

import argparse

from commands import (
    bm25_idf_command,
    bm25_tf_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)
from constants import BM25_B, BM25_K1
from inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build the movies data")
    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of a string")
    tf_parser.add_argument(
        "doc_id", type=int, help="Document ID to search for term frequency"
    )
    tf_parser.add_argument(
        "term", type=str, help="The term to get the term frequency of"
    )

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate a term inverse document frequency"
    )
    idf_parser.add_argument(
        "term",
        type=str,
        help="The term to calculate it's IDF",
    )

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get the term TF-IDF from a document"
    )
    tfidf_parser.add_argument(
        "doc_id", type=int, help="Document ID to search for the TF-IDF"
    )
    tfidf_parser.add_argument("term", type=str, help="The term to get the TF-IDF of")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    args = parser.parse_args()

    match args.command:
        case "search":
            try:
                print(f"Searching for: {args.query}")
                res = search_command(args.query)
                for i, m in enumerate(res, 1):
                    print(f"{i}. {m['id']} {m['title']}")
            except Exception as e:
                print({e})
        case "build":
            build_command()
        case "tf":
            try:
                tf = tf_command(args.doc_id, args.term)
                print(f"Term frequency of {args.term} on {args.doc_id}: {tf}")
            except Exception as e:
                print({e})
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
