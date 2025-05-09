import streamlit as st
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
        logging.info("'Process Uploaded Documents' button clicked.")
        if uploaded_files and vector_store and embeddings:
            current_uploaded_names = {f.name for f in uploaded_files}
            already_processed = st.session_state.processed_filenames

            files_to_add = current_uploaded_names - already_processed
            files_to_delete = already_processed - current_uploaded_names

            logging.info(f"Files selected in uploader: {current_uploaded_names}")
            logging.info(f"Files already processed: {already_processed}")
            logging.info(f"Files to add: {files_to_add}")
            logging.info(f"Files to delete: {files_to_delete}")

            made_changes = False

            if files_to_delete:
                try:
                    delete_documents_from_store(vector_store, list(files_to_delete))
                    st.session_state.processed_filenames -= files_to_delete
                    logging.info(
                        f"Successfully deleted embeddings for files: {files_to_delete}"
                    )
                    st.sidebar.info(f"Removed: {', '.join(files_to_delete)}")
                    made_changes = True
                except Exception as e_del:
                    logging.error(
                        f"Error deleting documents {files_to_delete}: {e_del}",
                        exc_info=True,
                    )
                    st.sidebar.error(f"Error removing old files: {e_del}")

            new_chunks_added_count = 0
            successfully_added_files_names = []

            if files_to_add:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in files_to_add:
                        try:
                            logging.info(
                                f"Processing file for addition: {uploaded_file.name}"
                            )
                            chunks = load_and_split_uploaded_pdf(
                                uploaded_file,
                                CHUNK_SIZE,
                                CHUNK_OVERLAP,
                                UPLOAD_TEMP_DIR,
                            )
                            if chunks:
                                if add_documents_to_store(vector_store, chunks):
                                    st.session_state.processed_filenames.add(
                                        uploaded_file.name
                                    )
                                    successfully_added_files_names.append(
                                        uploaded_file.name
                                    )
                                    new_chunks_added_count += len(chunks)
                                    logging.info(
                                        f"Successfully added embeddings for {uploaded_file.name} ({len(chunks)} chunks)."
                                    )
                                    made_changes = True
                                else:
                                    logging.error(
                                        f"Failed to add document chunks to store for {uploaded_file.name}."
                                    )
                                    st.sidebar.error(
                                        f"Failed to add {uploaded_file.name} to vector store."
                                    )
                            else:
                                logging.warning(
                                    f"No chunks created for {uploaded_file.name} (file empty or unreadable?)."
                                )
                                st.sidebar.warning(
                                    f"Could not process {uploaded_file.name} (no content extracted)."
                                )
                        except Exception as e_process:
                            logging.error(
                                f"Error processing file {uploaded_file.name}: {e_process}",
                                exc_info=True,
                            )
                            st.sidebar.error(
                                f"Error processing {uploaded_file.name}: {e_process}"
                            )

            if successfully_added_files_names:
                st.sidebar.success(
                    f"Added: {', '.join(successfully_added_files_names)} ({new_chunks_added_count} new chunks)."
                )
            elif files_to_add:  # Attempted to add but none succeeded
                st.sidebar.warning(
                    "Attempted to process new files, but no new embeddings were added."
                )

            # Re-create the RAG chain if changes were made or if it doesn't exist
            if made_changes or not st.session_state.current_rag_chain:
                logging.info("Rebuilding RAG chain...")
                st.session_state.current_rag_chain = create_rag_chain(vector_store, llm)
                if st.session_state.current_rag_chain:
                    st.sidebar.info("RAG chain (re)built successfully.")
                    logging.info("RAG chain (re)built successfully.")
                else:
                    st.sidebar.error("Failed to (re)build RAG chain.")
                    logging.error("Failed to (re)build RAG chain.")
                st.rerun()
            elif not files_to_add and not files_to_delete:
                st.sidebar.info("No changes to processed documents.")

        else:
            if not uploaded_files:
                st.sidebar.warning("No files selected in uploader to process.")
                logging.warning("Process button clicked, but no files were uploaded.")
            else:
                st.sidebar.error(
                    "System not ready for processing (check vector store/embeddings)."
                )
                logging.error(
                    "Process button clicked, but system (vector_store/embeddings) not ready."
                )

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
                st.session_state.current_rag_chain = None
                st.rerun()

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Placeholder functions for tools
def handle_calculation(query: str) -> str:
    logging.info(f"Calculator received query: {query}")
    if "what is" in query.lower():  # Example: "calculate what is 5 + 3"
        parts = query.lower().split("what is")[-1].strip().replace("?", "")
        try:
            result = eval(parts)
            return f"The result of '{parts}' is {result}."
        except Exception:
            return f"I can attempt basic arithmetic. Could not calculate '{parts}'."
    return f"Calculator tool would process: '{query}'"


def handle_definition(query: str) -> str:
    logging.info(f"Dictionary received query: {query}")
    keyword = query.lower().replace("define", "").strip().replace("?", "")
    if keyword:
        return f"Dictionary tool would define: '{keyword}'."
    return f"Dictionary tool would process: '{query}'"


if prompt := st.chat_input(
    "Ask a question, or try 'calculate ...' or 'define ...'", disabled=not SYSTEM_READY
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        answer = ""
        prompt_lower = prompt.lower()

        try:
            if "calculate" in prompt_lower:
                logging.info(f"Routing to calculator for query: {prompt}")
                answer = handle_calculation(prompt)
                message_placeholder.markdown(answer)
            elif "define" in prompt_lower:
                logging.info(f"Routing to dictionary for query: {prompt}")
                answer = handle_definition(prompt)
                message_placeholder.markdown(answer)
            else:
                logging.info(f"Routing to RAG chain for query: {prompt}")
                if not st.session_state.current_rag_chain:
                    logging.info(
                        "No RAG chain available. Attempting to answer with LLM directly."
                    )
                    if llm:
                        try:
                            start_time = time.time()
                            llm_response = llm.invoke(prompt)
                            end_time = time.time()

                            if hasattr(llm_response, "content"):
                                answer = llm_response.content
                            else:
                                answer = str(llm_response)

                            message_placeholder.markdown(answer)
                            st.caption(
                                f"LLM direct response time: {end_time - start_time:.2f} seconds"
                            )
                            logging.info("Successfully answered with LLM directly.")
                        except Exception as e_llm:
                            error_msg = f"🔴 Error during direct LLM call: {e_llm}"
                            logging.error(error_msg, exc_info=True)
                            message_placeholder.error(error_msg)
                            answer = error_msg
                    else:
                        answer = "LLM not available to answer directly."
                        message_placeholder.error(answer)
                        logging.error(answer)
                else:
                    qa_chain = st.session_state.current_rag_chain
                    start_time = time.time()
                    response = qa_chain.invoke({"query": prompt})
                    end_time = time.time()

                    answer = response.get("result", "Sorry, I couldn't find an answer.")
                    source_docs = response.get("source_documents", [])

                    message_placeholder.markdown(answer)
                    st.caption(
                        f"RAG response time: {end_time - start_time:.2f} seconds"
                    )

                    if source_docs:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(source_docs):
                                source_file = doc.metadata.get(
                                    METADATA_SOURCE_KEY, "Unknown"
                                )
                                page_num = doc.metadata.get("page", "N/A")
                                st.caption(
                                    f"Source {i+1}: '{source_file}' (Page: {page_num})"
                                )
                                st.write(doc.page_content[:350] + "...")
        except Exception as e:
            error_msg = f"🔴 Error processing your query: {e}"
            logging.error(error_msg, exc_info=True)
            message_placeholder.error(error_msg)
            answer = error_msg

    st.session_state.messages.append({"role": "assistant", "content": answer})

elif not SYSTEM_READY:
    st.warning("System is not ready. Please check sidebar status.")
elif not st.session_state.current_rag_chain and st.session_state.processed_filenames:
    st.info(
        "Please click 'Process Uploaded Documents' in the sidebar to activate chat."
    )
elif not st.session_state.processed_filenames:
    st.info("Please upload and process documents using the sidebar.")
