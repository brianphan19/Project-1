
# RAG Assistant

A **Retrieval-Augmented Generation (RAG)** assistant that answers user queries using a collection of local documents. It retrieves relevant text chunks from your document database and generates responses using an LLM.

---

## Features

- Retrieves relevant chunks from local text files using embeddings.
- Supports multiple LLM backends:
  - OpenAI (`gpt-3.5-turbo`, `gpt-4o-mini`)  
  - Groq 
  - Google Gemini
- Handles document metadata and context tracking.
- Works offline with local models to avoid API quotas.

---

## Project Structure

```
.
├─ data/                  # Folder containing text documents
├─ .env                   # Environment variables with API keys
├─ rag_assistant.py       # Main RAG assistant code
├─ requirements.txt       # Python dependencies
└─ README.md
```

---

## Setup

1. **Clone the repo**:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Create a `.env` file** in the root folder:

```dotenv
# Content directory
CONTENT_DIR=./data

# OpenAI API Configuration
OPENAI_API_KEY=
OPENAI_MODEL=gpt-3.5-turbo


# Groq API Configuration (alternative to OpenAI)
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant

# Google Gemini API Configuration (alternative to OpenAI/Groq)
GOOGLE_API_KEY=
GOOGLE_MODEL=gemini-pro

# Embedding Configuration
# Optional: HuggingFace model for embeddings (default: sentence-transformers/all-MiniLM-L6-v2)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Database Configuration
# Optional: ChromaDB collection name (default: rag_documents)
CHROMA_COLLECTION_NAME=rag_documents
```

---

## Usage

1. **Load documents**  
   Place `.txt` files in the `data/` folder. The assistant will read them, split into chunks, and store embeddings for retrieval.

2. **Run the RAG assistant**:

```python
from rag_assistant import RAGAssistant

rag = RAGAssistant()
while True:
    query = input("Enter a question or 'quit' to exit: ")
    if query.lower() == "quit":
        break
    result = rag.query(query)
    print("Answer:", result["answer"])
```

3. **Example query**:

```
Enter a question or 'quit' to exit: What is an asteroid?
Answer: ...
```

---

## Supported LLM Backends

| Backend         | Free Tier Options                   | Notes                                        |
|-----------------|------------------------------------|---------------------------------------------|
| OpenAI GPT      | Free trial credits                  | 429 errors if quota exceeded                |
| Groq LLaMA 3.1  | Free trial / local testing          | Supports instant and batch inference        |
| Google Gemini   | Free-tier API access                | May require account setup and quota limits  |

> Recommended for avoiding quota limits: use local models via Groq or manage quotas on OpenAI/Google Gemini.


---

## Notes / Tips

- **Document loading fix**: `_CONTENT_DIR` is joined with filenames to prevent FileNotFound errors.
- **Avoid irrelevant retrievals**:
  - Reduce `n_results` for generic queries.
  - Use similarity thresholds.
- **Embedding considerations**: better embeddings reduce irrelevant chunks and improve answer accuracy.
- **Short test queries**: If a query is too vague (like `"test"`), consider skipping retrieval or returning a generic response.
- **LLM7 usage**: For free-tier usage, set `LLM7_API_KEY=unused`. Use the default or "fast"/"pro" models in the client API.
- **OpenAI quotas**: Free-tier OpenAI accounts may hit 429 errors when credits are exhausted.

---

## Dependencies

- Python 3.10+  
- `chromadb` or other vector database backend  
- `sentence-transformers` (for embeddings)  
- `python-dotenv` (for `.env` loading)  
- `langchain` (optional, for LLM chaining)

Install all dependencies:

```bash
pip install openai chromadb sentence-transformers python-dotenv langchain
```

---

## License

This project is released under the MIT License.
