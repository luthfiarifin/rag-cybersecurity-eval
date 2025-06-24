import streamlit as st
import time

from src.rag_pipeline.graph import (
    rewrite_query,
    retrieve_and_rerank_documents,
    generate_answer,
)
from src.config import Config
from src.rag_pipeline.state import RAGState

st.set_page_config(page_title="Cybersecurity InstructRAG Assistant", layout="wide")
st.title("Cybersecurity InstructRAG Assistant")
st.info(
    """
**Welcome!** This assistant uses the state-of-the-art **InstructRAG** pipeline to answer your cybersecurity questions based on your provided PDF documents.\

**How it works:**
- Retrieves and reranks the most relevant passages from your documents.
- Instructs the language model to answer *only* using the retrieved context, reducing hallucinations and improving factual accuracy.
- Powered by open models for embedding, reranking, and generation, orchestrated by LangGraph.

*Ask a question about the topics covered in your PDFs to get started!*
"""
)


def ensure_config_loaded():
    """Ensures config is loaded from secrets if not set."""
    if not Config.GROQ_API_KEY:
        Config.GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
    if not Config.MONGO_URI:
        Config.MONGO_URI = st.secrets.get("MONGO_URI", None)
    if not Config.GROQ_API_KEY:
        st.error(
            "GROQ_API_KEY environment variable not set! Please set it in your .env file."
        )
        st.stop()
    if not Config.MONGO_URI:
        st.error(
            "MONGO_URI environment variable not set! Please set it in your .env file."
        )
        st.stop()


ensure_config_loaded()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def run_rag_pipeline(prompt: str):
    """Runs the RAG pipeline step by step, updating the UI for each stage."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        answer_box = st.empty()

        status.info("(1/3) Rewriting query...")
        with st.spinner("Thinking: Rewriting query..."):
            state: RAGState = {
                "query": prompt,
                "conversation_history": "\n".join(
                    f"{msg['role']}: {msg['content']}"
                    for msg in st.session_state.messages[-5:]
                ),
                "rewritten_query": "",
                "retrieved_docs": [],
                "reranked_docs": [],
                "answer": "",
                "context": "",
            }
            state = rewrite_query(state)

        status.info("(2/3) Retrieving and reranking documents...")
        with st.spinner("Thinking: Retrieving and reranking documents..."):
            state = retrieve_and_rerank_documents(state)

        status.info("(3/3) Generating answer...")
        with st.spinner("Thinking: Generating answer..."):
            state = generate_answer(state)
            full_response = state.get("answer", "I couldn't find an answer.")
            # Streaming simulation: show answer word by word
            streamed = ""
            for word in full_response.split():
                streamed += word + " "
                time.sleep(0.02)
            answer_box.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        status.empty()


prompt = st.chat_input("Ask a question about cybersecurity...")
if prompt:
    run_rag_pipeline(prompt)
