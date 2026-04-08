import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Securely load the API key from environment variable (or let the user provide it)
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.warning("Please set the GROQ_API_KEY environment variable. If you haven't, the app might throw an error.")

# Set up page configurations
st.set_page_config(page_title="FAQ Assistant - RAG", page_icon="🤖", layout="wide")

# --- Initialize Session State ---
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "num_entries" not in st.session_state:
    st.session_state.num_entries = 0
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

# --- Functions ---
@st.cache_resource
def init_pipeline():
    # 1. Load Data
    data_path = "sample_faq.txt"
    if not os.path.exists(data_path):
        st.error(f"Cannot find the FAQ dataset at {data_path}. Please create it first.")
        return None, 0, 0
    
    loader = TextLoader(data_path)
    documents = loader.load()
    
    # Simple count of QA pairs (each paragraph chunk separated by newline)
    with open(data_path, "r") as f:
        content = f.read()
        num_entries = content.count("Q:")
    
    # 2. Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    num_chunks = len(chunks)
    
    # 3. Create Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Store in Local ChromaDB
    # Persist directory
    persist_directory = "./chroma_db"
    
    # Create the vector database
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    return vectordb, num_entries, num_chunks

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def invoke_model(query):
    # 5. Connect Retriever and Groq LLM
    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 2})
    
    # Retrieve documents for observability
    retrieved_docs = retriever.invoke(query)
    
    # Setup Groq LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=256
    )
    
    # Setup prompt with fallback and anti-hallucination instructions
    template = """You are a helpful company HR assistant. Use only the following pieces of retrieved context to answer the question.
If the answer cannot be found in the provided context, you MUST reply EXACTLY with: "I could not find this in the FAQ knowledge base." Do not attempt to guess or use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    # Build chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Run Generation
    answer = rag_chain.invoke(query)
    
    return answer, retrieved_docs

# --- App Initialization ---
if st.session_state.vectordb is None:
    with st.spinner("Initializing Document Processing Pipeline..."):
        db, entries, chunks = init_pipeline()
        if db:
            st.session_state.vectordb = db
            st.session_state.num_entries = entries
            st.session_state.num_chunks = chunks

# --- UI Layout ---
st.title("🧑‍💻 Company FAQ RAG Assistant")
st.markdown("Ask anything about leave policy, remote work, office timings, reimbursement, IT support, holidays, payroll, or onboarding!")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat")
    # Quick question buttons
    st.write("Or try some example questions:")
    btn_cols = st.columns(2)
    with btn_cols[0]:
        q1 = st.button("How many leave days do employees get?", use_container_width=True)
        q2 = st.button("When is salary credited?", use_container_width=True)
    with btn_cols[1]:
        q3 = st.button("What is the remote work policy?", use_container_width=True)
        q4 = st.button("How do I contact IT support?", use_container_width=True)
        q5 = st.button("Do we have free lunch?", use_container_width=True) # Testing Fallback

    # Determine default text
    default_q = ""
    if q1: default_q = "How many leave days do employees get?"
    elif q2: default_q = "When is salary credited?"
    elif q3: default_q = "What is the remote work policy?"
    elif q4: default_q = "How do I contact IT support?"
    elif q5: default_q = "Do we have free lunch?"
    
    query = st.text_input("Type your question here:", value=default_q)
    
    if query:
        with st.spinner("Searching the Knowledge Base..."):
            answer, retrieved_docs = invoke_model(query)
            
            st.markdown("### Answer")
            st.info(answer)
            
            st.markdown("### Transparency & Retrieved Context")
            st.markdown(f"**Retrieved Chunks Used: {len(retrieved_docs)}**")
            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"Chunk {i+1} Preview"):
                    st.write(doc.page_content)

with col2:
    st.subheader("Observability Dashboard")
    st.metric(label="FAQ Entries Indexed", value=st.session_state.num_entries)
    st.metric(label="Embeddings Created (Chunks)", value=st.session_state.num_chunks)
    
    with st.expander("Preview Sample FAQ Dataset"):
        if os.path.exists("sample_faq.txt"):
            with open("sample_faq.txt", "r") as f:
                st.text(f.read())
        else:
            st.write("File not found.")
