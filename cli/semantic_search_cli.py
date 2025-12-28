#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
    search_command,
    chunk_command,
    semantic_chunk_command,
    embed_chunks_command,
    search_chunked_command
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the model used")
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text input"
    )
    embed_text_parser.add_argument(
        "text", type=str, help="The text to generate embedding for"
    )

    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings of the document"
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Get the shape and the first 5 dimension of a query"
    )
    embed_query_parser.add_argument(
        "query", type=str, help="The query to generate the embedding for"
    )

    search_parser = subparsers.add_parser(
        "search", help="Search the data for the query"
    )
    search_parser.add_argument(
        "query", type=str, help="The query to search the data for"
    )
    search_parser.add_argument(
        "--limit", required=False, default=5, type=int, help="Limit for the results. default: 5"
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk the long text to smaller pieces for embedding"
    )

    chunk_parser.add_argument(
        "text", type=str, help="The text to chunk"
    )

    chunk_parser.add_argument(
        "--chunk-size", required=False, default=200, type=int, help="Size of the chunk, default: 200"
    )

    chunk_parser.add_argument(
        "--overlap", required=False, default=20, type=int, help="Overlap of the chunk, default: 20"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk the long text to smaller pieces for embedding, semantically"
    )

    semantic_chunk_parser.add_argument(
        "text", type=str, help="The text to chunk"
    )

    semantic_chunk_parser.add_argument(
        "--max-chunk-size", required=False, default=4, type=int, help="Max size of the chunk, default: 4"
    )

    semantic_chunk_parser.add_argument(
        "--overlap", required=False, default=0, type=int, help="Overlap of the chunk, default: 0"
    )

    embed_chunks_parser = subparsers.add_parser(
        "embed_chunks", help="Generate chunk embeddings"
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search the data for the query"
    )

    search_chunked_parser.add_argument(
        "query", type=str, help="The query to search on the data"
    )

    search_chunked_parser.add_argument(
        "--limit", required=False, default=5, type=int, help="The limit of the results to return"
    )

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            search_command(args.query, args.limit)

        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            chunks = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i in range(len(chunks)):
                print(f"{i+1}. {chunks[i]}")

        case "embed_chunks":
            embed_chunks_command()

        case "search_chunked":
            search_chunked_command(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
