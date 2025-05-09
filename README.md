# Dynamic RAG Chatbot with ChromaDB & Gemini

This project implements a dynamic RAG (Retrieval Augmented Generation) chatbot using Streamlit for the user interface, Google Gemini for language models and embeddings, and ChromaDB as the vector store. It allows users to upload PDF documents, ask questions about their content, and also includes basic routing for simple calculator and dictionary queries.

## Features

- **PDF Document Processing**: Upload PDF files for content extraction and chunking.
- **RAG-based Q&A**: Ask questions and get answers based on the content of the uploaded documents.
- **Vector Store**: Utilizes ChromaDB to store and retrieve document embeddings.
- **Google Gemini Integration**: Leverages Google Gemini for powerful language model capabilities (LLM) and text embeddings.
- **Interactive UI**: Built with Streamlit for an easy-to-use chat interface and document management sidebar.
- **Functional Tools**:
  - **Calculator**: Evaluates simple arithmetic expressions (e.g., "calculate 2\*7+3").
  - **Dictionary**: Fetches real word definitions using an online API (e.g., "define synergy").
- **Direct LLM Fallback**: Answers general queries using the LLM directly if no documents are processed and the query isn't for a tool.
- **Dynamic Document Management**: Add or remove documents, and the RAG chain updates accordingly.
- **Logging**: Logs key decisions and processing steps for better observability.

## Technology Stack

- **Python 3.x**
- **Streamlit**: For the web application interface.
- **LangChain**: Framework for building LLM applications.
- **Google Generative AI SDK**: For Gemini LLM and embedding models.
- **ChromaDB**: For the vector store.
- **PyPDFLoader**: For loading PDF document content.
- **Dotenv**: For managing environment variables.
- **Requests**: For making HTTP requests (used by the Dictionary tool).

## Setup and Configuration

1.  **Clone the Repository (if you haven't already):**

    ```bash
    # git clone <repository-url>
    # cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Make sure you have Python and pip installed. Then run:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    - Copy the example environment file:
      ```bash
      cp .env.example .env
      ```
    - Open the `.env` file and add your `GOOGLE_API_KEY`:
      ```
      GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
      ```

## Running the Application

Once the setup is complete, you can run the Streamlit application:

```bash
streamlit run app.py
```

The application should open in your web browser, usually at `http://localhost:8501`.

## How It Works

1.  **Initialization**: The app initializes the Google Gemini embedding model, the LLM, and connects to the ChromaDB vector store.
2.  **Document Upload**: Users can upload PDF files via the sidebar.
3.  **Processing**: When "Process Uploaded Documents" is clicked:
    - PDFs are parsed, split into chunks.
    - Embeddings are generated for these chunks using Gemini.
    - Chunks and their embeddings are stored in ChromaDB.
    - The RAG chain is (re)built using the updated vector store.
4.  **Chat Interface**:
    - User enters a query.
    - If the query contains "calculate", it's routed to an internal calculator that evaluates the arithmetic expression.
    - If the query contains "define", it's routed to a dictionary tool that fetches the definition from an online API.
    - Otherwise, if documents are processed and a RAG chain exists, the query is sent to the RAG chain to retrieve relevant document chunks and generate an answer.
    - If no documents are processed (and it's not a tool query), the query is sent directly to the LLM for a general answer.
    - Responses and (if applicable) sources are displayed.

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── config.py               # Configuration variables (API keys, paths, model names)
├── document_processor.py   # Handles PDF loading, splitting
├── rag_chain.py            # Logic for creating the RAG chain and LLM instance
├── vector_store.py         # Manages ChromaDB vector store interactions
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment file
├── .gitignore              # Specifies intentionally untracked files
├── temp_uploads/           # Temporary directory for uploaded files (auto-created)
└── chroma_db/              # Persistent directory for ChromaDB (auto-created)
```
