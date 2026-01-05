import heapq
from PIL import Image
from sentence_transformers import SentenceTransformer
from vector_utils import cosine_similarity
from constants import DEFAULT_SEARCH_LIMIT
from search_utils import load_movies, format_search_result


class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = list(map(lambda doc: f"{doc['title']}: {doc['description']}", documents))
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path:str):
        image = Image.open(image_path)

        image_embedding = self.model.encode([image])[0]

        return image_embedding

    def search_with_image(self, image_path: str):
        image_embedding = self.embed_image(image_path)

        results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            results.append({
                "score": cosine_similarity(image_embedding, text_embedding),
                "doc_idx": i
            })

        sorted_results = heapq.nlargest(DEFAULT_SEARCH_LIMIT, results, lambda x: x["score"])

        dcts = []
        for res in sorted_results:
            doc_idx = res["doc_idx"]
            doc = self.documents[doc_idx]
            id = doc["id"]
            title = doc["title"]
            description = doc["description"]
            dcts.append(format_search_result(id, title, description[:100], res["score"]))

        return dcts


def verify_image_embedding(image_path: str):
    documents = load_movies()
    ms = MultimodalSearch(documents)
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str):
    documents = load_movies()
    ms = MultimodalSearch(documents)
    results = ms.search_with_image(image_path)
    return results
