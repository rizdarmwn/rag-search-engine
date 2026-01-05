import argparse
from mimetypes import guess_type
from lib.describe_image import describe_image_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    parser.add_argument(
        "--image",
        type=str,
        help="The path to image file"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Text query to rewrite based on the image"
    )

    args = parser.parse_args()

    image_path = args.image
    query = args.query

    response = describe_image_command(image_path, query)
    print(f"Rewritten query: {response["answer"]}")
    if response["usage_metadata"] is not None:
        print(f"Total tokens:    {response["usage_metadata"].total_token_count}")




if __name__ == '__main__':
    main()
