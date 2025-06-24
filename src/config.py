import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Configuration class for the InstructRAG application.
    """

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME", "cybersecurity_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents_vector_store")
    VECTOR_SEARCH_INDEX_NAME = "vector_index"

    # LLM and Embedding/Reranker Model configuration
    LLM_MODEL = "llama3-70b-8192"
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    RERANKER_MODEL = "BAAI/bge-reranker-base"

    # Data paths
    PDF_DIRECTORY = "data/pdfs"
    EVAL_DATA_PATH = "data/pentesting-eval.csv"

    # RAG Pipeline Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    RETRIEVAL_K = 20
    RERANK_K = 5

    # Batch processing configuration
    BATCH_DELAY_SECONDS = 5
