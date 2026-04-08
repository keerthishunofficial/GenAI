import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Document Semantic Search", layout="wide")
st.title("📄 Document Indexing & Retrieval")
st.markdown("Upload a PDF product manual, index it via the backend, and perform semantic search queries via vector similarity.")

# --- Document Upload & Indexing Section ---
st.header("1. Upload & Index Document")
uploaded_file = st.file_uploader("Upload a PDF product manual", type="pdf")

if uploaded_file and st.button("Index Document"):
    with st.spinner("Extracting text, chunking, and embedding into ChromaDB..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        try:
            response = requests.post(f"{API_URL}/index", files=files)
            if response.status_code == 200:
                st.success(f"Success! Indexed {response.json().get('chunks_indexed')} semantic chunks.")
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Backend. Is FastAPI running on port 8000?")

st.divider()

# --- Search Section ---
st.header("2. Semantic Search")
query = st.text_input("Enter your question related to the manual:")
top_k = st.slider("Number of relevant chunks", 1, 10, 3)

if st.button("Search") and query:
    with st.spinner("Retrieving relevant semantics..."):
        try:
            response = requests.post(f"{API_URL}/search", json={"query": query, "top_k": top_k})
            if response.status_code == 200:
                results = response.json().get("results", [])
                
                if not results:
                    st.warning("No matches found.")
                else:
                    st.subheader(f"Top {len(results)} Relevant Sections")
                    for i, res in enumerate(results):
                        with st.expander(f"Result {i+1} | Source: {res['metadata']['source']}"):
                            st.write(res["text"])
                            if res.get('score'):
                                st.caption(f"Similarity Distance: {res['score']:.4f}")
            else:
                st.error(f"Backend error occurred. Status: {response.status_code}, Detail: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Backend. Is FastAPI running on port 8000?")
