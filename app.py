import streamlit as st
import os
import time

from config import (
    GOOGLE_API_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    UPLOAD_TEMP_DIR,
    METADATA_SOURCE_KEY,
    check_api_key,
)
from document_processor import load_and_split_uploaded_pdf
from vector_store import (
    get_embedding_function,
    initialize_vector_store,
    add_documents_to_store,
    delete_documents_from_store,
    clear_vector_store_collection,
)
from rag_chain import get_llm, create_rag_chain

st.set_page_config(page_title="Dynamic Doc Chatbot", layout="wide")
st.title("📄 Dynamic RAG Chatbot with ChromaDB & Gemini")
st.write("Upload PDF documents, ask questions, and manage embeddings.")

try:
    check_api_key()
    embeddings = get_embedding_function()
    vector_store = initialize_vector_store(embeddings)
    llm = get_llm()
    SYSTEM_READY = all([embeddings, vector_store, llm])
except ValueError as ve:
    st.error(ve)
    SYSTEM_READY = False
except Exception as e:
    st.error(f"An unexpected error occurred during initialization: {e}")
    SYSTEM_READY = False

# Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload documents and ask me questions!"}
    ]
if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = set()
if "current_rag_chain" not in st.session_state:
    st.session_state.current_rag_chain = None

# UI Components

with st.sidebar:
    st.header("📄 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True,
        key="file_uploader",
        disabled=not SYSTEM_READY,
    )

    if st.button(
        "🔄 Process Uploaded Documents",
        key="process_button",
        disabled=not SYSTEM_READY or not uploaded_files,
    ):
        if uploaded_files and vector_store and embeddings:
            current_uploaded_names = {f.name for f in uploaded_files}
            already_processed = st.session_state.processed_filenames

            # 1. Identify Files to Add and Delete
            files_to_add = current_uploaded_names - already_processed
            files_to_delete = already_processed - current_uploaded_names

            # 2. Delete embeddings for removed files
            if files_to_delete:
                delete_documents_from_store(vector_store, list(files_to_delete))
                st.session_state.processed_filenames -= files_to_delete

            # 3. Process and add embeddings for new files
            new_chunks_added = 0
            for uploaded_file in uploaded_files:
                if uploaded_file.name in files_to_add:
                    chunks = load_and_split_uploaded_pdf(
                        uploaded_file, CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_TEMP_DIR
                    )
                    if chunks:
                        if add_documents_to_store(vector_store, chunks):
                            st.session_state.processed_filenames.add(uploaded_file.name)
                            new_chunks_added += len(chunks)

            if new_chunks_added > 0:
                st.sidebar.success(
                    f"Added embeddings for {len(files_to_add)} file(s) ({new_chunks_added} chunks)."
                )

            # 4. Re-create the RAG chain with potentially updated vector store
            st.session_state.current_rag_chain = create_rag_chain(vector_store, llm)
            if st.session_state.current_rag_chain:
                st.sidebar.info("RAG chain updated.")

            st.rerun()

    st.divider()
    st.subheader("⚠️ Danger Zone")
    if st.button(
        "🗑️ Clear All Documents & Embeddings", key="clear_all", disabled=not SYSTEM_READY
    ):
        if vector_store:
            if clear_vector_store_collection(vector_store):
                st.session_state.processed_filenames = set()
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Embeddings cleared. Upload new documents.",
                    }
                ]
                st.session_state.current_rag_chain = None  # Reset chain
                st.rerun()  # Rerun to update UI

    st.divider()
    st.subheader("📊 Status")
    if SYSTEM_READY:
        st.success("Core System Initialized")
        if st.session_state.processed_filenames:
            st.write("**Indexed Documents:**")
            for fname in sorted(list(st.session_state.processed_filenames)):
                st.caption(f"- {fname}")
        else:
            st.info("No documents processed yet.")

        if st.session_state.current_rag_chain:
            st.success("RAG Chain is Active")
        elif st.session_state.processed_filenames:
            st.warning(
                "Documents indexed, but RAG chain needs processing (Click 'Process Uploaded Documents')."
            )
        else:
            st.info("Upload and process documents to activate the RAG chain.")

    else:
        st.error("System Initialization Failed. Check API Key and configurations.")


st.header("💬 Chat Interface")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(
    "Ask a question about the processed documents:",
    disabled=not st.session_state.current_rag_chain,
):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from RAG chain
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        try:
            qa_chain = st.session_state.current_rag_chain
            if not qa_chain:
                raise ValueError("RAG Chain not available. Process documents first.")

            start_time = time.time()
            response = qa_chain.invoke({"query": prompt})
            end_time = time.time()

            answer = response.get("result", "Sorry, I couldn't find an answer.")
            source_docs = response.get("source_documents", [])

            message_placeholder.markdown(
                answer
            )  # Update placeholder with the final answer
            st.caption(f"Response time: {end_time - start_time:.2f} seconds")

            # Display sources
            if source_docs:
                with st.expander("View Sources"):
                    for i, doc in enumerate(source_docs):
                        source_file = doc.metadata.get(METADATA_SOURCE_KEY, "Unknown")
                        page_num = doc.metadata.get("page", "N/A")
                        st.caption(f"Source {i+1}: '{source_file}' (Page: {page_num})")
                        st.write(doc.page_content[:350] + "...")

        except Exception as e:
            error_msg = f"🔴 Error processing your query: {e}"
            message_placeholder.error(error_msg)
            answer = error_msg  # Store error as answer for history

    # Add assistant response (or error) to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

elif not SYSTEM_READY:
    st.warning("System is not ready. Please check sidebar status.")
elif not st.session_state.current_rag_chain and st.session_state.processed_filenames:
    st.info(
        "Please click 'Process Uploaded Documents' in the sidebar to activate chat."
    )
elif not st.session_state.processed_filenames:
    st.info("Please upload and process documents using the sidebar.")
