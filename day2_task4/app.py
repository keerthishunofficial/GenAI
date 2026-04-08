"""
Multi-Document RAG Assistant
=============================
Upload PDF, CSV, and TXT files → index into ChromaDB → ask questions answered
by Groq (llama-3.1-8b-instant) using only retrieved context.

Run:  streamlit run app.py
Requires:  GROQ_API_KEY environment variable set.
"""

import os, tempfile, hashlib
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Doc RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Multi-Document RAG Assistant")
st.caption("Upload PDFs, CSVs, or TXT files → ask questions grounded in your documents.")

# ── Sidebar: API key ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password",
                            value=os.environ.get("GROQ_API_KEY", ""))
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "rag_chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

FALLBACK = "I could not find this in the uploaded documents."

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Answer the question ONLY using the context below.\n"
        "If the context does not contain enough information, reply exactly:\n"
        f'"{FALLBACK}"\n\n'
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_file(path: str, filename: str):
    """Return LangChain Documents for a single file."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext == ".csv":
        return CSVLoader(path, encoding="utf-8").load()
    elif ext == ".txt":
        return TextLoader(path, encoding="utf-8").load()
    else:
        st.warning(f"Unsupported file type: {filename}")
        return []


def process_documents(uploaded_files):
    """Load, chunk, embed, and store documents. Returns stats dict."""
    all_docs = []
    for uf in uploaded_files:
        # Write to temp file
        suffix = Path(uf.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uf.read())
        tmp.close()

        docs = load_file(tmp.name, uf.name)
        # Attach metadata
        for d in docs:
            d.metadata["source_filename"] = uf.name
            d.metadata["filetype"] = suffix.lstrip(".")
        all_docs.extend(docs)
        os.unlink(tmp.name)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(all_docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    # Embed & store
    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=CHROMA_DIR
    )

    return {
        "num_files": len(uploaded_files),
        "num_docs": len(all_docs),
        "num_chunks": len(chunks),
        "vectordb": vectordb,
    }


# ── Session state ────────────────────────────────────────────────────────────
if "stats" not in st.session_state:
    st.session_state.stats = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# ── File upload ──────────────────────────────────────────────────────────────
st.subheader("1️⃣ Upload Documents")
uploaded = st.file_uploader(
    "Choose PDF, CSV, or TXT files",
    type=["pdf", "csv", "txt"],
    accept_multiple_files=True,
)

if uploaded:
    st.write(f"**{len(uploaded)} file(s) selected:**")
    for f in uploaded:
        st.write(f"- `{f.name}` ({f.size / 1024:.1f} KB)")

# ── Document preview ─────────────────────────────────────────────────────────
    with st.expander("📄 Preview uploaded files"):
        for f in uploaded:
            ext = Path(f.name).suffix.lower()
            f.seek(0)
            if ext == ".txt":
                st.text_area(f.name, f.read().decode("utf-8", errors="replace"), height=150)
            elif ext == ".csv":
                import pandas as pd
                st.dataframe(pd.read_csv(f), use_container_width=True)
            else:
                st.info(f"{f.name}: PDF preview not available – content will be extracted on indexing.")
            f.seek(0)

# ── Index button ─────────────────────────────────────────────────────────────
st.subheader("2️⃣ Index Documents")
if st.button("🔨 Index Documents", disabled=not uploaded):
    if not api_key:
        st.error("Please provide your Groq API key in the sidebar.")
    else:
        with st.spinner("Processing & indexing…"):
            result = process_documents(uploaded)
            st.session_state.vectordb = result["vectordb"]
            st.session_state.stats = {k: v for k, v in result.items() if k != "vectordb"}

if st.session_state.stats:
    s = st.session_state.stats
    c1, c2, c3 = st.columns(3)
    c1.metric("📁 Files uploaded", s["num_files"])
    c2.metric("📝 Chunks created", s["num_chunks"])
    c3.metric("💾 Embeddings stored", s["num_chunks"])

# ── Question answering ───────────────────────────────────────────────────────
st.subheader("3️⃣ Ask a Question")
question = st.text_input("Type your question about the uploaded documents:")

if question and st.session_state.vectordb:
    if not api_key:
        st.error("Please provide your Groq API key in the sidebar.")
    else:
        llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.2)
        retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": TOP_K})

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT_TEMPLATE},
        )

        with st.spinner("Retrieving & generating answer…"):
            result = qa.invoke({"query": question})

        st.markdown("### 💡 Answer")
        st.write(result["result"])

        with st.expander("🔍 Retrieved Context Chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}** — `{doc.metadata.get('source_filename', '?')}` "
                            f"(type: {doc.metadata.get('filetype', '?')}, "
                            f"chunk_id: {doc.metadata.get('chunk_id', '?')})")
                st.text(doc.page_content[:600])
                st.divider()

elif question and not st.session_state.vectordb:
    st.warning("Please upload and index documents first.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with LangChain · ChromaDB · HuggingFace · Groq · Streamlit")
