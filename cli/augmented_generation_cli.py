import argparse

from constants import DEFAULT_SEARCH_LIMIT
from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize the search results"
    )
    summarize_parser.add_argument("query", type=str, help="The query to search for")
    summarize_parser.add_argument(
        "--limit",
        default=DEFAULT_SEARCH_LIMIT,
        help="The limit for the results searched",
        type=int,
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Summarize the search results"
    )
    citations_parser.add_argument("query", type=str, help="The query to search for")
    citations_parser.add_argument(
        "--limit",
        default=DEFAULT_SEARCH_LIMIT,
        help="The limit for the results searched",
        type=int,
    )

    question_parser = subparsers.add_parser(
        "question", help="Summarize the search results"
    )
    question_parser.add_argument("question", type=str, help="The question to ask the AI")
    question_parser.add_argument(
        "--limit",
        default=DEFAULT_SEARCH_LIMIT,
        help="The limit for the results searched",
        type=int,
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            response = rag_command(query)
            if response["error"]:
                print(response["error"])
            else:
                print("Search Results:")
                for res in response["results"]:
                    print(f"\t- {res['title']}")

                print("RAG Response:")
                print(f"{response['answer']}")
        case "summarize":
            query = args.query
            limit = args.limit
            response = summarize_command(query, limit)
            if response["error"]:
                print(response["error"])
            else:
                print("Search Results:")
                for res in response["results"]:
                    print(f"\t- {res['title']}")

                print("LLM Response:")
                print(f"{response['answer']}")
        case "citations":
            query = args.query
            limit = args.limit
            response = citations_command(query, limit)
            if response["error"]:
                print(response["error"])
            else:
                print("Search Results:")
                for res in response["results"]:
                    print(f"\t- {res['title']}")

                print("LLM Answer:")
                print(f"{response['answer']}")

        case "question":
            question = args.question
            limit = args.limit
            response = question_command(question, limit)
            if response["error"]:
                print(response["error"])
            else:
                print("Search Results:")
                for res in response["results"]:
                    print(f"\t- {res['title']}")

                print("Answer:")
                print(f"{response['answer']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
