import os
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader, WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.config import Config
from tqdm import tqdm


def load_and_chunk_docs() -> List[Document]:
    """
    Loads documents from all configured sources (PDFs, Wikipedia) and
    splits them into chunks.
    """
    all_docs = []

    # Load from PDFs
    if os.path.exists(Config.PDF_DIRECTORY) and os.listdir(Config.PDF_DIRECTORY):
        print(f"Loading PDFs from {Config.PDF_DIRECTORY}...")
        pdf_loader = PyPDFDirectoryLoader(Config.PDF_DIRECTORY)
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
        print(f"Loaded {len(pdf_docs)} pages from PDFs.")
    else:
        print(
            f"Warning: PDF directory '{Config.PDF_DIRECTORY}' is empty or does not exist."
        )

    # Load from Wikipedia
    wikipedia_keywords_path = "src/data_processing/wikipedia_keywords.txt"
    wikipedia_keywords = []
    with open(wikipedia_keywords_path, "r") as f:
        wikipedia_keywords = [line.strip() for line in f.readlines()]
    if not wikipedia_keywords:
        print(
            f"Warning: No keywords found in '{wikipedia_keywords_path}'. "
            "Please add keywords to load Wikipedia articles."
        )
        return []

    print(f"Loading {len(wikipedia_keywords)} articles from Wikipedia...")
    for keyword in tqdm(wikipedia_keywords, desc="Loading Wikipedia Articles"):
        try:
            wiki_loader = WikipediaLoader(
                query=keyword,
                load_max_docs=1,
                doc_content_chars_max=50000,
                load_all_available_meta=True,
            )
            wiki_docs = wiki_loader.load()
            all_docs.extend(wiki_docs)
        except Exception as e:
            print(f"Could not load Wikipedia page for '{keyword}': {e}")

    if not all_docs:
        print("No documents loaded from any source.")
        return []

    # Chunk all documents
    print(f"\nChunking {len(all_docs)} total documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
    )
    chunked_documents = text_splitter.split_documents(all_docs)
    print(f"Successfully chunked documents into {len(chunked_documents)} chunks.")

    return chunked_documents
