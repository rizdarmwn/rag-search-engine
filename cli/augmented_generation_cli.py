import argparse
from lib.augmented_generation import rag_command

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

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
                    print(f"- {res["title"]}")

                print("RAG Response:")
                print(f"{response["answer"]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
