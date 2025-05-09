import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    METADATA_SOURCE_KEY,
)


@st.cache_resource(show_spinner="Initializing Embeddings...")
def get_embedding_function():
    """Initializes and returns the embedding function."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY
        )
        st.success("✅ Embedding function initialized.")
        return embeddings
    except Exception as e:
        st.error(f"🔴 Failed to initialize embedding function: {e}")
        return None


# Vector Store Initialization
@st.cache_resource(show_spinner="Connecting to Vector Store...")
def initialize_vector_store(_embeddings):
    """
    Initializes connection to ChromaDB and returns the LangChain Chroma object.
    Ensures the collection exists. Does NOT add documents here.
    """
    if _embeddings is None:
        st.error("🔴 Cannot initialize vector store without embeddings.")
        return None
    try:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=_embeddings,
        )
        vector_store._client.heartbeat()
        st.success(f"✅ Connected to ChromaDB collection '{COLLECTION_NAME}'.")
        return vector_store
    except Exception as e:
        st.error(f"🔴 Failed to connect/initialize ChromaDB: {e}")
        st.info(f"Attempted to use directory: {CHROMA_PERSIST_DIR}")
        return None


# Add Documents
def add_documents_to_store(vector_store, chunks_with_metadata):
    """Adds document chunks (with metadata) to the Chroma vector store."""
    if not vector_store or not chunks_with_metadata:
        st.warning("⚠️ Add documents skipped: No vector store or chunks provided.")
        return False
    try:
        st.info(f"⏳ Adding {len(chunks_with_metadata)} new chunks to vector store...")
        vector_store.add_documents(chunks_with_metadata)

        st.success(f"✅ Added {len(chunks_with_metadata)} chunks.")
        return True
    except Exception as e:
        st.error(f"🔴 Error adding documents to vector store: {e}")
        return False


# Delete Documents by Filename
def delete_documents_from_store(vector_store, filenames_to_delete):
    """Deletes documents from Chroma based on filename metadata."""
    if not vector_store or not filenames_to_delete:
        st.warning("⚠️ Delete documents skipped: No vector store or filenames provided.")
        return False
    try:
        st.info(
            f"⏳ Deleting documents from vector store: {', '.join(filenames_to_delete)}"
        )

        vector_store.delete(where={METADATA_SOURCE_KEY: {"$in": filenames_to_delete}})
        st.success(f"✅ Deleted documents: {', '.join(filenames_to_delete)}")
        return True
    except Exception as e:
        st.error(f"🔴 Error deleting documents: {e}")
        return False


# Clear Entire Collection
def clear_vector_store_collection(vector_store):
    """Deletes all items from the specified collection."""
    if not vector_store:
        st.warning("⚠️ Clear collection skipped: Vector store not available.")
        return False
    try:
        collection_name = vector_store._collection.name
        st.info(f"⏳ Clearing all embeddings from collection '{collection_name}'...")

        existing_items = vector_store.get()
        if existing_items and existing_items.get("ids"):
            vector_store.delete(ids=existing_items["ids"])
            st.success(
                f"✅ Cleared all {len(existing_items['ids'])} items from '{collection_name}'."
            )
        else:
            st.info("Collection was already empty.")
        return True

    except Exception as e:
        st.error(f"🔴 Error clearing collection '{vector_store._collection.name}': {e}")
        return False
