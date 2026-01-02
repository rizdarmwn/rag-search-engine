import heapq
import argparse

import os
import time
from dotenv import load_dotenv
from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_command
from google import genai
from search_enhancement import enhance_query
from reranking import rerank
from lib.evaluation import llm_judge_evaluate

def main() -> None:

    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Get the normalized scores"
    )
    normalize_parser.add_argument(
        "scores", nargs='+', type=float, help="The scores to normalized. type=list"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Get the search results with weighted search"
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="The query to search"
    )
    weighted_search_parser.add_argument(
        "--alpha", type=float, required=False, default=0.5, help="The alpha of the weighted search"
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, required=False, default=5, help="The limit to the results"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Get the search results with RRF ranking search"
    )
    rrf_search_parser.add_argument(
        "query", type=str, help="The query to search"
    )
    rrf_search_parser.add_argument(
        "--k", type=int, required=False, default=60, help="The k constant to control how much more weight we give to higher-ranked results vs lower-ranked ones"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, required=False, default=5, help="The limit to the results"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method"
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        help="Rerank the query",
        required=False,
        choices=["individual", "batch", "cross_encoder"]
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        help="Evaluate the search results",
        required=False,
        action="store_true"
    )


    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_command(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            ws = weighted_search_command(args.query, args.alpha, args.limit)
            for i, res in enumerate(ws, 1):
                print(f"{i}. {res['title']}")
                print(f"\tHybrid Score: {res['metadata']['hybrid_score']:.4f}")
                print(f"\tBM25: {res['metadata']['kw_score']:.4f}, Semantic: {res['metadata']['sm_score']:.4f}")
                print(f"\t{res['document'][:100]}...")
        case "rrf-search":
            result = rrf_search_command(args.query, args.enhance, args.rerank_method, args.k, args.limit)
            if args.enhance:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{result["enhanced_query"]}'\n")

            if result["reranked"]:
                print(
                    f"Reranking top {len(result['results'])} results using {result['rerank_method']} method...\n"
                    )

            print(f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):")
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Rerank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Rerank Rank: {res.get('batch_rank', 0)}")
                if "cross_encoder_score" in res:
                    print(f"   Cross Encoder Score: {res.get('cross_encoder_score', 0):.4f}")
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("kw_rank"):
                    ranks.append(f"BM25 Rank: {metadata['kw_rank']}")
                if metadata.get("sm_rank"):
                    ranks.append(f"Semantic Rank: {metadata['sm_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()

            if args.evaluate:
                scores = llm_judge_evaluate(result["query"], result["results"])
                for i, score in enumerate(scores, 1):
                    print(f"{i}. {score["title"]}: {score["score"]}/3")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
