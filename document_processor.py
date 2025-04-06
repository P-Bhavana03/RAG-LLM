from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import streamlit as st

from config import METADATA_SOURCE_KEY


def load_and_split_uploaded_pdf(uploaded_file, chunk_size, chunk_overlap, temp_dir):
    """
    Loads an uploaded PDF file, adds filename metadata, and splits it into chunks.

    Args:
        uploaded_file: The file object from st.file_uploader.
        chunk_size (int): Max size of text chunks.
        chunk_overlap (int): Overlap between chunks.
        temp_dir (str): Directory to save temporary files.

    Returns:
        list: A list of LangChain Document objects (chunks) with metadata, or None.
    """
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", dir=temp_dir
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.info(f"⏳ Processing '{file_name}'...")
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        if not documents:
            st.warning(f"⚠️ Could not load content from '{file_name}'.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            st.warning(f"⚠️ Splitting '{file_name}' resulted in zero chunks.")
            return None

        # Add filename metadata to each chunk
        for chunk in chunks:
            chunk.metadata[METADATA_SOURCE_KEY] = file_name
            chunk.metadata["page"] = chunk.metadata.get("page", "N/A") + 1

        st.success(f"✅ Processed '{file_name}' into {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        st.error(f"🔴 Error processing '{file_name}': {e}")
        return None
    finally:
        # Clean up the temporary file
        if "tmp_file_path" in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
