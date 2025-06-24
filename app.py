import streamlit as st
import sys
import os

from src.rag_pipeline.graph import build_rag_graph
from src.config import Config

# --- App Setup ---
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


@st.cache_resource
def get_rag_app():
    """Builds and returns the RAG graph application."""
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
    return build_rag_graph()


app = get_rag_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about cybersecurity..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Running InstructRAG pipeline..."):
            history = "\n".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in st.session_state.messages[-5:]
                ]
            )
            inputs = {"query": prompt, "conversation_history": history}

            try:
                response = app.invoke(inputs)
                full_response = response.get("answer", "I couldn't find an answer.")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
