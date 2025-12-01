import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        # Initialize the text splitter for chunking text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """
        Split a text string into smaller chunks using the configured text splitter.

        This method uses LangChain's RecursiveCharacterTextSplitter to handle
        sentence boundaries and preserve context better than simple splitting.

        Args:
            text (str): The input text to be chunked.
            chunk_size (int, optional): Approximate number of characters per chunk.
                Defaults to 500.

        Returns:
            list[str]: A list of text chunks.
        """
        # Update the text splitter's chunk size if different from requested
        if chunk_size != self.text_splitter._chunk_size:
            self.text_splitter._chunk_size = chunk_size
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text=text)
        return chunks

    def add_documents(self, documents: list[dict]):
        """
        Process a list of documents and add them to the vector database.

        Each document is split into text chunks, metadata is attached to
        each chunk, embeddings are created, and the chunks are stored in
        the vector database.

        Args:
            documents (list[dict]): List of documents, where each document
                is a dictionary containing:
                - 'content' (str): The text of the document.
                - 'metadata' (dict): Any metadata associated with the document.

        Notes:
            - Each chunk is assigned a unique ID for tracking.
            - Metadata is copied for each chunk and includes a 'chunk_index'.
            - Uses the class's `chunk_text` and `embedding_model`.
            - Assumes `self.collection` is a ChromaDB collection ready to store data.
        """
        print(f"Processing and add {len(documents)} documents...")

        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            chunks = self.chunk_text(content)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_idx

                all_metadatas.append(chunk_metadata)
                all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")

        if not all_chunks:
            print("No text content found in documents to add.")
            return

        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)

        self.collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )

        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> dict[str, any]:
        """
        Search for similar documents in the vector database using a query string.

        The query is converted into an embedding using the embedding model,
        then the vector database is queried to find the closest matches.

        Args:
            query (str): The search query.
            n_results (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            dict: A dictionary containing search results with keys:
                - 'documents' (list[str]): The text content of matched chunks.
                - 'metadatas' (list[dict]): Metadata associated with each chunk.
                - 'distances' (list[float]): Similarity distances between query and chunk.
                - 'ids' (list[str]): Unique IDs of the matched chunks.
        """
        # Step 1: Create query embedding
        query_embedding = self.embedding_model.encode([query])

        # Step 2: Search the vector database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances", "ids"]
        )

        # Step 3: Handle case when no results are returned
        if not results or not results["documents"]:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }

        # Step 4: Return the structured results
        return {
            "documents": results.get("documents", []),
            "metadatas": results.get("metadatas", []),
            "distances": results.get("distances", []),
            "ids": results.get("ids", []),
        }