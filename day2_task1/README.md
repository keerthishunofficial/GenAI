# Document Indexing and Semantic Retrieval Lab

This web application demonstrates a minimal document indexing pipeline and semantic search interface using a lightweight, open-source stack. The system extracts structured text from a PDF product manual, splits it into semantic chunks, computes vector embeddings, and allows users to perform semantic search queries via vector similarity.

## 🏗 Architecture & Reasoning

The architecture is divided into three core stages:
1. **Document Indexing Pipeline (Data Ingestion):** We use `pypdf` to extract text from a lengthy manual. The text is chunked using LangChain's `RecursiveCharacterTextSplitter` to ensure semantic coherence (preserving paragraphs and sentences).
2. **Embedding & Storage:** We use LangChain's `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) to generate lightweight, dense vector representations of the chunks. These are persisted locally using `ChromaDB`, allowing for fast, offline similarity search without external API costs.
3. **Retrieval Interface:** A decoupled frontend-backend setup. FastAPI handles the core indexing and search workload securely, while Streamlit provides a dynamic layout for students to interact with the retrieved document chunks.

*Scope:* This implementation rigorously emphasizes the retrieval (indexing & search) stages. Generative synthesis (like LLM answer-generation in full RAG) is intentionally excluded to focus on the semantic search groundwork.

---

## 🛠 Environment Setup

### 1. Folder Structure

The project uses the following file structure:

```text
document_retrieval_lab/
│
├── requirements.txt
├── backend.py            # FastAPI service (Extraction, Chunking, Embeddings, ChromaDB)
├── frontend.py           # Streamlit user interface
└── chroma_db/            # (Auto-generated) Vector DB local storage
```

### 2. Dependency Installation

First, create a virtual environment, activate it, and install the required dependencies:

```bash
# Create and activate virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install dependencies using pip
pip install -r requirements.txt
```

*(Note: The HuggingFace open-source packages will download local embedding weights the first time they run. They will perfectly scale for a 100-page PDF document on standard laptop hardware.)*

---

## 💻 Code Explanations

### Backend: FastAPI & ChromaDB (`backend.py`)

This file implements the document indexing pipeline and search logic. It saves the ChromaDB state locally in the `./chroma_db` directory so you don't have to re-index the PDF every time you restart the application.

- **`/index` Endpoint:** Handles PDF ingestion, extracts text via `pypdf`, splits into 1000-character chunks with a 200-character overlap using `RecursiveCharacterTextSplitter`, embeds them via Langchain's `HuggingFaceEmbeddings`, and stores them in ChromaDB.
- **`/search` Endpoint:** Receives a string query, computes its embedding, and matches it against the stored chunks in ChromaDB.

### Frontend: Streamlit (`frontend.py`)

This file provides the student-friendly interface to test the pipeline.
- Contains an uploader to send the PDF to the backend.
- Features a search bar and a slider to select the Top-K results.
- Retrieves and elegantly expands the context from the backend's vector search response.

---

## 🚀 Execution Instructions

To run this application locally, you need to start both services in separate terminal windows.

**Running the Backend:**
Open Terminal 1, activate your environment, and execute:
```bash
uvicorn backend:app --reload
```
*Wait until you see `Uvicorn running on http://127.0.0.1:8000`.*

**Running the Frontend:**
Open Terminal 2, activate your environment, and execute:
```bash
streamlit run frontend.py
```
*This will open the Streamlit interface in your default web browser automatically.*

> [!TIP]
> **Vector Database Persistence**
> The `backend.py` file maps `chromadb.PersistentClient` to the local `./chroma_db` folder. This means you only need to index your 100-page PDF once. If you stop the servers and restart them later, your loaded embeddings are safely loaded from disk! You can immediately start making search queries without re-uploading the file.
