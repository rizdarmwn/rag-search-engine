#!/usr/bin/env python3

import argparse

from inverted_index import InvertedIndex
from matching import build_command, search_command, tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build the movies data")
    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of a string")
    tf_parser.add_argument(
        "document_id", type=int, help="Document ID to search for term frequency"
    )
    tf_parser.add_argument(
        "term", type=str, help="The term to get the term frequency of"
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
                tf = tf_command(args.document_id, args.term)
                print(f"Term frequency of {args.term} on {args.document_id}: {tf}")
            except Exception as e:
                print({e})

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
