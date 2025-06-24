from typing import List, TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict):
    """
    Represents the state of our RAG pipeline.
    """

    query: str
    rewritten_query: str
    conversation_history: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    answer: str
    context: str
