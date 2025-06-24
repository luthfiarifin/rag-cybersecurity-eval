from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever

from src.config import Config


def get_llm():
    """Initializes and returns the main LLM."""
    if not Config.GROQ_API_KEY:
        raise ValueError("No Groq API key found in configuration.")
    return ChatGroq(
        temperature=0, groq_api_key=Config.GROQ_API_KEY, model_name=Config.LLM_MODEL
    )


def create_reranking_retriever(
    base_retriever: BaseRetriever,
) -> ContextualCompressionRetriever:
    """
    Creates a retriever that reranks documents using a BGE cross-encoder.
    """
    model = HuggingFaceCrossEncoder(model_name=Config.RERANKER_MODEL)
    # The new compressor will return the top 'k' documents after reranking
    compressor = CrossEncoderReranker(model=model, top_n=Config.RERANK_K)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return compression_retriever


def create_query_rewriter_chain():
    """Creates a chain to rewrite a query based on conversation history."""
    # (Your existing code is perfect)
    prompt = PromptTemplate(
        template="""
        Based on the conversation history, rewrite the following user query into a concise, standalone question
        that captures the full intent of the user.
        
        <Conversation History>
        {conversation_history}
        </Conversation History>

        User Query: {query}
        Standalone Question:
        """,
        input_variables=["conversation_history", "query"],
    )
    return prompt | get_llm() | StrOutputParser()


def create_answer_generation_chain():
    """
    Creates the final answer generation chain with citation instructions.
    """
    # (Your existing code is perfect)
    prompt = PromptTemplate(
        template="""
        **Task:** You are a helpful cybersecurity assistant. Your goal is to provide a clear and accurate answer to the user's question based *only* on the provided context blocks. After your answer, you **must** cite the specific sources you used.

        **Context:**
        {context}

        **Question:**
        {query}

        **Instructions for your response:**
        1. Formulate a comprehensive answer to the question using only the information from the context provided.
        2. At the end of your answer, create a "Sources" section.
        3. List each unique source document you used to formulate your answer in the "Sources" section. Do not make up sources.

        **Answer:**
        """,
        input_variables=["context", "query"],
    )
    return prompt | get_llm() | StrOutputParser()
