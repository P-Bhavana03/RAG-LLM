import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from config import GOOGLE_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE, RETRIEVER_SEARCH_K


@st.cache_resource(show_spinner="Initializing LLM...")
def get_llm():
    """Initializes and returns the Language Model."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            convert_system_message_to_human=True,
        )
        st.success("✅ Language Model (LLM) initialized.")
        return llm
    except Exception as e:
        st.error(f"🔴 Failed to initialize LLM: {e}")
        return None


def create_rag_chain(vector_store, llm):
    """
    Creates the Retrieval-Augmented Generation (RAG) chain.

    Args:
        vector_store: The initialized and potentially updated vector store object.
        llm: The initialized language model object.

    Returns:
        RetrievalQA: The initialized RAG chain object, or None if an error occurs.
    """
    if vector_store is None or llm is None:
        st.error("🔴 Cannot create RAG chain: Vector store or LLM not available.")
        return None

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        return qa_chain
    except Exception as e:
        st.error(f"🔴 Error creating RAG chain: {e}")
        return None
