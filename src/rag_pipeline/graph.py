import os
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langgraph.graph import StateGraph, END
from src.config import Config
from src.rag_pipeline.state import RAGState
from src.rag_pipeline.chains import (
    create_query_rewriter_chain,
    create_answer_generation_chain,
    create_reranking_retriever,
)
from src.vector_store.builder import get_embeddings_model


def rewrite_query(state: RAGState) -> RAGState:
    """Rewrites the user's query to be standalone only if conversation history exists."""
    print("--- REWRITING QUERY ---")
    if state["conversation_history"]:
        rewriter = create_query_rewriter_chain()
        rewritten = rewriter.invoke(
            {
                "query": state["query"],
                "conversation_history": state["conversation_history"],
            }
        )
        state["rewritten_query"] = rewritten
        print(f"Rewritten Query: {state['rewritten_query']}")
    else:
        state["rewritten_query"] = state["query"]
        print("No conversation history, using original query.")
    return state


def retrieve_and_rerank_documents(state: RAGState) -> RAGState:
    """
    Retrieves documents from MongoDB Atlas and reranks them in a single step.
    """
    print("--- RETRIEVING AND RERANKING DOCUMENTS ---")
    embeddings = get_embeddings_model()
    client = MongoClient(Config.MONGO_URI)
    collection = client[Config.DB_NAME][Config.COLLECTION_NAME]

    # 1. Create the base retriever from MongoDB Atlas
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=Config.VECTOR_SEARCH_INDEX_NAME,
    )
    base_retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": Config.RETRIEVAL_K}
    )

    # 2. Create the reranking retriever which wraps the base retriever
    reranking_retriever = create_reranking_retriever(base_retriever)

    # 3. Invoke the retriever to get the final, reranked documents
    # This single call now performs both retrieval and reranking
    reranked_docs = reranking_retriever.invoke(state["rewritten_query"])

    # We now store the final documents directly in a single state key
    state["retrieved_docs"] = reranked_docs
    print(f"Retrieved and reranked to {len(reranked_docs)} documents.")
    return state


def generate_answer(state: RAGState) -> RAGState:
    """Generates the final answer and formats the context for citation."""
    print("--- GENERATING ANSWER ---")

    # The reranked documents are now in 'retrieved_docs'
    docs_to_generate_from = state["retrieved_docs"]

    context_with_sources = []
    for doc in docs_to_generate_from:
        source = doc.metadata.get("source", "Unknown Source")
        source_name = os.path.basename(source)
        page = doc.metadata.get("page")
        source_info = f"Source: `{source_name}`"
        if page is not None:
            source_info += f", Page: {page + 1}"
        context_with_sources.append(
            f"<{source_info}>\n{doc.page_content}\n</{source_info}>"
        )
    context = "\n\n---\n\n".join(context_with_sources)

    answer_generator = create_answer_generation_chain()
    answer = answer_generator.invoke(
        {"context": context, "query": state["rewritten_query"]}
    )

    state["answer"] = answer
    state["context"] = context
    print(f"Generated Answer: {answer[:200]}...")
    return state


def build_rag_graph():
    """Builds the simplified LangGraph for the RAG pipeline."""
    workflow = StateGraph(RAGState)

    # Add the nodes to the graph
    workflow.add_node("rewrite_query", rewrite_query)

    # We use our new consolidated function for the 'retrieve' node
    workflow.add_node("retrieve", retrieve_and_rerank_documents)
    workflow.add_node("generate", generate_answer)

    # Define the graph's flow
    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")

    # The 'retrieve' node now directly connects to 'generate'
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
