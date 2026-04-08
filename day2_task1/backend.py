import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

app = FastAPI(title="Document Indexing API")

# 1. Setup ChromaDB for Local Persistence
CHROMA_DATA_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection_name = "product_manuals"
collection = chroma_client.get_or_create_collection(name=collection_name)

# 2. Setup LangChain Embeddings
# Using a lightweight, local model ideal for lab environments
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/index")
async def index_document(file: UploadFile = File(...)):
    """Handles PDF ingestion, text splitting, embedding, and storing in ChromaDB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Step 1 & 2: Load PDF and extract text
        reader = PdfReader(temp_path)
        extracted_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted_text += f"\n--- Page {i+1} ---\n{text}"
                
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found.")

        # Step 3: Split text into semantic chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(extracted_text)
        
        # Step 4 & 5: Generate embeddings and store in ChromaDB
        documents, metadatas, ids = [], [], []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": file.filename, "chunk_id": i})
            ids.append(f"{file.filename}_chunk_{i}")
            
        embedded_docs = embeddings.embed_documents(documents)
        
        collection.upsert(
            documents=documents,
            embeddings=embedded_docs,
            metadatas=metadatas,
            ids=ids
        )
        
        return {"message": "Success", "chunks_indexed": len(chunks)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/search")
async def search_document(request: QueryRequest):
    """Embeds the user query and retrieves matching document chunks."""
    query_embedding = embeddings.embed_query(request.query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.top_k
    )
    
    matches = []
    if results['documents'] and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            matches.append({
                "id": results['ids'][0][i],
                "score": results['distances'][0][i] if 'distances' in results and results['distances'] else None,
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
            
    return {"query": request.query, "results": matches}
