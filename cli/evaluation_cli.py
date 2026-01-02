import argparse
import json

from constants import GOLDEN_DATASET_PATH
from lib.hybrid_search import rrf_search_command
from lib.evaluation import precision_command

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    print(f"k={limit}")

    result = precision_command(limit)

    for q, res in result["results"].items():
        precision = res["precision"]
        retrieved_docs = res["retrieved"]
        relevant_docs = res["relevant"]
        print(f"- Query: {q}")
        print(f"\t- Precision@{limit}: {precision:.4f}")
        print(f"\t- Retrieved: {", ".join(retrieved_docs)}")
        print(f"\t- Relevant: {", ".join(relevant_docs)}")
        print()




if __name__ == "__main__":
    main()
