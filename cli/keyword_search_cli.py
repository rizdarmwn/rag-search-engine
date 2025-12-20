#!/usr/bin/env python3

import argparse

from inverted_index import InvertedIndex
from matching import matching_title


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build the movies data")

    args = parser.parse_args()

    match args.command:
        case "search":
            # try:
            print(f"Searching for: {args.query}")
            res = matching_title(args.query)
            for i, m in enumerate(res, 1):
                print(f"{i}. {m['id']} {m['title']}")
        # except Exception as e:
        #     print({e})

        case "build":
            inverted_idx = InvertedIndex()
            inverted_idx.build()
            inverted_idx.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
