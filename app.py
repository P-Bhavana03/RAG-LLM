import streamlit as st
import os
import time
import logging
import re
import requests

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
            elif files_to_add:
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


# Tool handler functions
def handle_calculation(query: str) -> str:
    logging.info(f"Calculator received query: {query}")
    expression_to_eval = ""
    query_lower = query.lower()

    if "what is" in query_lower:
        expression_to_eval = (
            query_lower.split("what is", 1)[-1].strip().replace("?", "")
        )
    elif query_lower.startswith("calculate "):
        expression_to_eval = (
            query_lower.replace("calculate ", "", 1).strip().replace("?", "")
        )

    if expression_to_eval:
        sanitized_expression = "".join(
            char for char in expression_to_eval if char in "0123456789.+-*/() "
        )

        if not sanitized_expression or not re.fullmatch(
            r"[\d\s\.\+\-\*\/\(\)]+", sanitized_expression
        ):
            logging.warning(
                f"Invalid characters in expression or empty after sanitization: '{expression_to_eval}' -> '{sanitized_expression}'"
            )
            return f"Could not calculate '{expression_to_eval}'. Please use numbers and basic operators (+, -, *, /)."

        if all(c in "0123456789.+-*/() " for c in expression_to_eval):
            try:
                result = eval(sanitized_expression)
                return f"The result of '{expression_to_eval}' is {result}."
            except ZeroDivisionError:
                return f"Error: Cannot divide by zero ('{expression_to_eval}')."
            except SyntaxError:
                return f"Error: Invalid syntax in calculation ('{expression_to_eval}'). Please check your expression."
            except Exception as e:
                logging.error(
                    f"Error evaluating sanitized expression '{sanitized_expression}': {e}",
                    exc_info=True,
                )
                return f"Could not calculate '{expression_to_eval}'. An unexpected error occurred."
        else:
            logging.warning(
                f"Expression contained disallowed characters, rejecting: '{expression_to_eval}'"
            )
            return f"Could not calculate '{expression_to_eval}'. Please use only numbers, spaces, and basic operators (+, -, *, /)."

    return f"Calculator tool would process: '{query}'. (Hint: Try 'calculate 5+3' or 'calculate what is 2*7')"


def handle_definition(query: str) -> str:
    logging.info(f"Dictionary received query: {query}")
    keyword_match = re.search(r"define\s+([\w\s]+)", query, re.IGNORECASE)
    if not keyword_match:
        simple_keyword = query.lower().replace("define", "").strip().replace("?", "")
        if not simple_keyword:
            return "Please specify a word to define after 'define'."
        keyword_to_define = simple_keyword
    else:
        keyword_to_define = keyword_match.group(1).strip()

    if not keyword_to_define:
        return "Please specify a word to define."

    api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{keyword_to_define}"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        data = response.json()
        print(data)

        if isinstance(data, list) and data:
            meanings = data[0].get("meanings", [])
            if meanings:
                definitions_text = []
                for i, meaning in enumerate(meanings[:2]):
                    part_of_speech = meaning.get("partOfSpeech", "N/A")
                    definitions_list = meaning.get("definitions", [])
                    if definitions_list:
                        defs_to_show = [
                            d.get("definition")
                            for d in definitions_list[:2]
                            if d.get("definition")
                        ]
                        if defs_to_show:
                            definitions_text.append(
                                f"**{part_of_speech.capitalize()}**:\n"
                                + "\n".join(f"- {d}" for d in defs_to_show)
                            )

                if definitions_text:
                    return (
                        f"**{data[0].get('word', keyword_to_define).capitalize()}**:\n\n"
                        + "\n\n".join(definitions_text)
                    )
                else:
                    return f"No definitions found for '{keyword_to_define}' in the response."
            else:
                return f"No meanings found for '{keyword_to_define}'."
        elif isinstance(data, dict) and data.get("title") == "No Definitions Found":
            return f"Sorry, I couldn't find a definition for '{keyword_to_define}'. {data.get('message', '')}"
        else:
            logging.warning(
                f"Unexpected API response structure for '{keyword_to_define}': {data}"
            )
            return f"Sorry, received an unexpected response when looking up '{keyword_to_define}'."

    except Exception as e:
        logging.error(
            f"Unexpected error in handle_definition for '{keyword_to_define}': {e}",
            exc_info=True,
        )
        return f"An unexpected error occurred while trying to define '{keyword_to_define}'."


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
