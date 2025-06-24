from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import Config
from src.data_processing.loader import load_and_chunk_docs


def get_embeddings_model():
    """Initializes and returns the embeddings model."""
    print(f"Initializing embedding model: {Config.EMBEDDING_MODEL}")
    # Using 'cpu' for broader compatibility. Change to 'cuda' if a GPU is available.
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store():
    """
    Builds and populates the MongoDB Atlas vector store.
    This function will clear the existing collection and create a new vector search index
    if it doesn't already exist.
    """
    if not Config.MONGO_URI:
        print("ERROR: MONGO_URI is not set in the .env file. Aborting.")
        return

    print("--- Starting MongoDB Atlas Vector Store Build Process ---")

    # Initialize MongoDB client and collection
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.DB_NAME]
    collection = db[Config.COLLECTION_NAME]

    # Create the Vector Search Index if it doesn't exist
    index_name = Config.VECTOR_SEARCH_INDEX_NAME
    try:
        if not any(
            index["name"] == index_name for index in collection.list_search_indexes()
        ):
            print(f"Vector search index '{index_name}' not found. Creating it now...")

            # Get the embedding dimension from the model
            embeddings = get_embeddings_model()
            embedding_dimension = len(embeddings.embed_query("test"))

            index_definition = {
                "name": index_name,
                "definition": {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": embedding_dimension,
                                "similarity": "cosine",
                            },
                            "metadata.source": {"type": "string"},
                            "metadata.page": {"type": "number"},
                        },
                    }
                },
            }
            collection.create_search_index(model=index_definition)
            print(
                f"Successfully started creation of index '{index_name}'. It may take a few minutes to become ready."
            )
        else:
            print(f"Vector search index '{index_name}' already exists.")
    except Exception as e:
        print(f"An error occurred while checking or creating the search index: {e}")
        print("Please ensure you have the necessary permissions in MongoDB Atlas.")
        return

    # Clear existing documents from the collection
    print(
        f"Clearing all existing documents from the '{Config.COLLECTION_NAME}' collection..."
    )
    collection.delete_many({})
    print("Collection cleared.")

    # Load and chunk documents
    documents = load_and_chunk_docs()
    if not documents:
        print("No documents to process. Vector store build complete.")
        return

    # Get embeddings model (if not already loaded)
    if "embeddings" not in locals():
        embeddings = get_embeddings_model()

    # Populate the vector store
    print(f"Populating vector store with {len(documents)} document chunks...")
    MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=index_name,
    )

    print("\n--- Vector Store Build Process Finished Successfully ---")
    print(f"Collection '{Config.COLLECTION_NAME}' is populated.")
