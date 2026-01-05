from mimetypes import guess_type
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from constants import PROJECT_ROOT

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

def describe_image_command(image_path: str, query: str):
    image_path = os.path.join(PROJECT_ROOT, image_path)
    mime, _ = guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        image = f.read()

    system_prompt = """
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """

    parts = [
        system_prompt,
        types.Part.from_bytes(data=image, mime_type=mime),
        query.strip()
    ]

    response = client.models.generate_content(model=model, contents=parts)
    text = (response.text or "").strip()

    f.close()

    return {"query": query, "usage_metadata": response.usage_metadata,"error": None, "answer": text}
