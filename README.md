# RAG Search Engine

A sophisticated movie search and recommendation system leveraging Retrieval-Augmented Generation (RAG). This project demonstrates the integration of classical information retrieval (BM25) with modern vector-based semantic search, further enhanced by LLM-powered reranking, query expansion, and automated evaluation.

## 🚀 Key Features

### Advanced Search Algorithms
- **Hybrid Search**: Combines keyword-based retrieval with semantic vector search for high-precision results.
- **Reciprocal Rank Fusion (RRF)**: Merges multiple search result sets using industry-standard ranking aggregation.
- **Semantic Search**: Utilizes `sentence-transformers` (`all-MiniLM-L6-v2`) to understand the contextual meaning of queries.
- **Chunked Retrieval**: Implements semantic chunking to handle long documents and improve retrieval granularity.

### Intelligence & Generation
- **RAG (Retrieval-Augmented Generation)**: Generates contextually accurate answers and summaries using Google Gemini, backed by retrieved movie metadata.
- **Query Enhancement**: Automated spell-correction, query rewriting, and expansion to bridge the gap between user intent and indexed content.
- **Multimodal Capabilities**: Support for image description and cross-modal search.
- **Multi-Stage Reranking**: Refines results using Cross-Encoders and LLM-based batch reranking for maximum relevance.

### Evaluation & Tooling
- **LLM-Judge**: An automated evaluation framework that uses an LLM to score the relevance of search results against a query.
- **Preprocessing Pipeline**: Robust text processing including tokenization, Porter stemming, and stopword removal.
- **Embedding Cache**: Efficiently manages and caches vector embeddings to minimize computation time and API costs.

## 🛠️ Technical Stack

- **Language**: Python 3.13+
- **LLM Integration**: `google-genai` (Gemini API)
- **Machine Learning**: `sentence-transformers`, `torch`, `numpy`
- **Natural Language Processing**: `nltk`
- **Package Management**: `uv`

## 📦 Installation

1.  **Clone the repository**:
    ```/dev/null/shell
    git clone https://github.com/yourusername/rag-search-engine.git
    cd rag-search-engine
    ```

2.  **Install dependencies**:
    This project uses `uv` for fast, reliable dependency management.
    ```/dev/null/shell
    uv sync
    ```

3.  **Configure Environment**:
    Create a `.env` file in the root directory and add your Google API key:
    ```/dev/null/.env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```
    
4.  **Download the Dataset**:
    Download the dataset here: [movies.json](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json)
    
    Make a `data` directory on the project root, and put `movies.json` there.

## 📖 Usage Examples

The system is accessible via several specialized CLI tools located in the `cli/` directory.

### Hybrid Ranked Search
Perform a search using Reciprocal Rank Fusion and Cross-Encoder reranking:
```/dev/null/shell
python cli/hybrid_search_cli.py rrf-search "intense space exploration movies" --rerank-method cross_encoder
```

### Retrieval-Augmented Generation (RAG)
Generate an answer to a specific question based on the movie database:
```/dev/null/shell
python cli/augmented_generation_cli.py question "Which movies involve time travel paradoxes?"
```

### Search Evaluation
Run a search and immediately evaluate the relevance of the results using the LLM-Judge:
```/dev/null/shell
python cli/hybrid_search_cli.py rrf-search "classic film noir" --evaluate
```

## 📂 Project Structure

- `cli/`: Entry points for search, generation, and evaluation workflows.
- `cli/lib/`: Core logic for search engines (Inverted Index, Semantic, Hybrid).
- `data/`: Source datasets and utility files (e.g., `movies.json`).
- `cache/`: Local storage for pre-computed embeddings and metadata.
- `cli/preprocessing.py`: Text normalization and cleaning utilities.

---
*This project was developed to demonstrate best practices in modern search architecture and the practical application of LLMs in information retrieval.*
