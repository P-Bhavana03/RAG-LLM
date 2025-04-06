import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_uploads")

CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "document_collection"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-exp-03-07"
LLM_MODEL_NAME = "gemini-2.5-pro-preview-03-25"
LLM_TEMPERATURE = 0.7
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

RETRIEVER_SEARCH_K = 3
METADATA_SOURCE_KEY = "source_doc_filename"


def check_api_key():
    """Checks if the Google API key is available."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "🔴 Google API Key not found. Please set the GOOGLE_API_KEY environment variable in your .env file."
        )
    print("✅ Google API Key configured.")
