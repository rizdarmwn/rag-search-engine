import argparse
import json

from constants import GOLDEN_DATASET_PATH
from lib.hybrid_search import rrf_search_command

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

    with open(GOLDEN_DATASET_PATH, "r") as data:
        json_data = json.load(data)

    for tc in json_data["test_cases"]:
        query = tc["query"]
        results = rrf_search_command(query, k=60, limit=limit)
        total_retrieved = len(results["results"])
        relevant_retrieved = 0
        relevant_docs = tc["relevant_docs"]
        total_titles = list(map(lambda x: x["title"], results["results"]))
        for r_title in relevant_docs:
            for title in total_titles:
                if title == r_title:
                    relevant_retrieved += 1
                    break
        precision = min(1.0, relevant_retrieved / total_retrieved)
        print(f"- Query: {results["query"]}")
        print(f"\t- Precision@{limit}: {precision:.4f}")
        print(f"\t- Retrieved: {", ".join(total_titles)}")
        print(f"\t- Relevant: {", ".join(relevant_docs)}")
        print()




if __name__ == "__main__":
    main()
