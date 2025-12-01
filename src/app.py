import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI



# Load environment variables
load_dotenv()

# Document directory
_CONTENT_DIR = os.getenv("CONTENT_DIR")

def load_documents() -> List[str]:
    """
    Load all text (.txt) documents from the `_CONTENT_DIR` directory.

    Each document is read as a string, and metadata about the file
    (filename and full path) is attached.

    Returns:
        List[dict]: A list of dictionaries, each containing:
            - 'content' (str): The text content of the file.
            - 'metadata' (dict): Metadata about the file, including:
                - 'filename' (str): The name of the file.
                - 'full_path' (str): The relative path to the file.
    
    Notes:
        - Files that cannot be read will be skipped, and an error will
          be printed to the console.
        - Only files ending with '.txt' are considered.
    """
    results = []
    txt_files = [f for f in os.listdir(_CONTENT_DIR) if f.endswith(".txt")]

    for fname in txt_files:
        try:
            full_path = os.path.join(_CONTENT_DIR, fname)

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = {
                "filename": fname,
                "full_path": full_path,
            }

            results.append({
                "content": content,
                "metadata": metadata
            })

        except Exception as e:
            print(f"Error loading document {fname}: {e}")

    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
                """
            You are a helpful AI assistant. Use the retrieved context below to answer the user's question.

            --- Retrieved Context ---
            {context}
            --------------------------
            
            Communication style:
            - Use clear, concise language with bullet points when approriate

            Instructions:
            - Base your answer only on the provided context.
            - If the context does not contain enough information, say so clearly and give the best general answer you can.
            - If a question goes beyond scope, politely refuse: "I'm sorry, that information is not in the documents."
            - Do NOT make up facts that are not supported by the context.
            - Be concise and direct.

            User Question:
            {question}

            Answer:
            """
        )

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def query(self, input: str, n_results: int = 2) -> Dict[str, Any]:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        # 1. Retrieve relevant documents
        search_results = self.vector_db.search(query=input, n_results=n_results)

        # search_results expected shape:
        # {
        #   "ids": [...],
        #   "documents": [...],
        #   "metadatas": [...],
        #   "distances": [...]
        # }

        retrieved_docs = search_results.get("documents", [])
        if retrieved_docs and isinstance(retrieved_docs[0], list):
            retrieved_docs = retrieved_docs[0]
        combined_context = "\n\n".join(retrieved_docs)

        # 2. Invoke LLM chain
        llm_response = self.chain.invoke({
            "context": combined_context,
            "question": input
        })

        # Some LangChain LLMs return .content, others return raw str
        answer = llm_response.get("answer") if isinstance(llm_response, dict) else llm_response

        # 3. Return structured response
        return {
            "question": input,
            "answer": answer,
            "context_used": retrieved_docs,
            "raw_retrieval": search_results
        }


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.query(question)
                print(result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
