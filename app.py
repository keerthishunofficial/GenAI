"""
Hybrid Search E-Commerce Assistant
===================================
A Streamlit app demonstrating Hybrid Search (keyword + vector) with
metadata filtering using ChromaDB, LangChain, and the Groq API.

Run:  streamlit run app.py
"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ──────────────────────────────────────────────
# 1. Sample product knowledge base (20 products)
# ──────────────────────────────────────────────
PRODUCTS = [
    # Electronics
    {"text": "Wireless headphones with noise cancellation and 20-hour battery life.", "category": "Electronics", "type": "Headphones", "price_range": "$50-$100"},
    {"text": "Bluetooth portable speaker with deep bass and waterproof design.", "category": "Electronics", "type": "Speaker", "price_range": "$30-$60"},
    {"text": "4K Ultra HD smart TV with 55-inch display and built-in streaming apps.", "category": "Electronics", "type": "TV", "price_range": "$400-$700"},
    {"text": "Wireless ergonomic mouse with adjustable DPI and silent clicks.", "category": "Electronics", "type": "Mouse", "price_range": "$20-$40"},
    {"text": "Noise-cancelling earbuds with touch controls and 8-hour playtime.", "category": "Electronics", "type": "Earbuds", "price_range": "$40-$80"},
    # Clothing
    {"text": "Men's slim-fit cotton t-shirt available in 10 colors.", "category": "Clothing", "type": "T-Shirt", "price_range": "$15-$25"},
    {"text": "Women's waterproof hiking jacket with breathable membrane.", "category": "Clothing", "type": "Jacket", "price_range": "$80-$150"},
    {"text": "Unisex running shoes with responsive cushioning and lightweight mesh.", "category": "Clothing", "type": "Shoes", "price_range": "$60-$120"},
    {"text": "Classic denim jeans with stretch comfort and straight leg fit.", "category": "Clothing", "type": "Jeans", "price_range": "$30-$50"},
    {"text": "Warm fleece hoodie with kangaroo pocket and adjustable drawstring.", "category": "Clothing", "type": "Hoodie", "price_range": "$25-$45"},
    # Home Appliances
    {"text": "Robot vacuum cleaner with smart mapping and automatic charging.", "category": "Home Appliances", "type": "Vacuum", "price_range": "$200-$400"},
    {"text": "Air fryer with digital display, 5-quart capacity, and 8 preset modes.", "category": "Home Appliances", "type": "Air Fryer", "price_range": "$60-$100"},
    {"text": "Instant pot multi-cooker with pressure cook, slow cook, and sauté functions.", "category": "Home Appliances", "type": "Multi-Cooker", "price_range": "$70-$120"},
    {"text": "Cordless stick vacuum with powerful suction and HEPA filtration.", "category": "Home Appliances", "type": "Vacuum", "price_range": "$150-$300"},
    {"text": "Countertop blender with 1200W motor for smoothies and ice crushing.", "category": "Home Appliances", "type": "Blender", "price_range": "$40-$80"},
    # Books
    {"text": "Productivity guide: Atomic Habits by James Clear – build better routines.", "category": "Books", "type": "Self-Help", "price_range": "$10-$20"},
    {"text": "Science fiction novel exploring AI consciousness and ethical dilemmas.", "category": "Books", "type": "Fiction", "price_range": "$10-$18"},
    {"text": "Beginner's cookbook with 100 quick and healthy weeknight recipes.", "category": "Books", "type": "Cookbook", "price_range": "$12-$25"},
    {"text": "Deep Work by Cal Newport – focused success in a distracted world.", "category": "Books", "type": "Self-Help", "price_range": "$10-$20"},
    {"text": "Children's illustrated encyclopedia covering science, nature, and history.", "category": "Books", "type": "Educational", "price_range": "$15-$30"},
]

ALL_CATEGORIES = ["All", "Electronics", "Clothing", "Home Appliances", "Books"]
FALLBACK_MSG = "I could not find this in the knowledge base."

# ──────────────────────────────────────────────
# 2. Build ChromaDB vector store (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model & building vector store…")
def build_vector_store():
    """Create an in-memory ChromaDB collection from the product catalogue."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [
        Document(
            page_content=p["text"],
            metadata={"category": p["category"], "type": p["type"], "price_range": p["price_range"]},
        )
        for p in PRODUCTS
    ]
    vectorstore = Chroma.from_documents(docs, embeddings, collection_name="products")
    return vectorstore

# ──────────────────────────────────────────────
# 3. Keyword search (simple BM25-style matching)
# ──────────────────────────────────────────────
def keyword_search(query: str, category: str, top_k: int = 10) -> list[Document]:
    """Return products whose text contains ANY query keyword."""
    keywords = query.lower().split()
    results = []
    for p in PRODUCTS:
        text_lower = p["text"].lower()
        if any(kw in text_lower for kw in keywords):
            if category == "All" or p["category"] == category:
                results.append(
                    Document(
                        page_content=p["text"],
                        metadata={"category": p["category"], "type": p["type"], "price_range": p["price_range"]},
                    )
                )
    return results[:top_k]

# ──────────────────────────────────────────────
# 4. Vector search via ChromaDB
# ──────────────────────────────────────────────
def vector_search(query: str, vectorstore, category: str, top_k: int = 10) -> list[Document]:
    """Retrieve semantically similar documents, optionally filtered by category."""
    if category != "All":
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k, "filter": {"category": category}},
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever.invoke(query)

# ──────────────────────────────────────────────
# 5. Merge & deduplicate results
# ──────────────────────────────────────────────
def merge_results(keyword_docs: list[Document], vector_docs: list[Document]) -> list[Document]:
    seen = set()
    merged = []
    for doc in keyword_docs + vector_docs:
        key = doc.page_content
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged

# ──────────────────────────────────────────────
# 6. Generate grounded response via Groq
# ──────────────────────────────────────────────
def generate_response(query: str, context_docs: list[Document], api_key: str) -> str:
    if not context_docs:
        return FALLBACK_MSG

    context = "\n".join(
        f"- {d.page_content} [Category: {d.metadata['category']}, Type: {d.metadata['type']}, Price: {d.metadata['price_range']}]"
        for d in context_docs
    )

    system_prompt = (
        "You are a helpful e-commerce product assistant. "
        "Answer the user's question using ONLY the provided product context below. "
        "If the context does not contain relevant information, respond with: "
        f'"{FALLBACK_MSG}"\n\n'
        f"Product Context:\n{context}"
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, api_key=api_key)
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ])
    return response.content

# ──────────────────────────────────────────────
# 7. Streamlit UI
# ──────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Hybrid Search Assistant", page_icon="🔍", layout="wide")
    st.title("🔍 Hybrid Search E-Commerce Assistant")
    st.caption("Keyword + Vector search with metadata filtering · ChromaDB · LangChain · Groq")

    # Sidebar – API key & category filter
    with st.sidebar:
        st.header("⚙️ Settings")
        groq_key = st.text_input("Groq API Key", type="password", help="Get one free at console.groq.com")
        st.divider()
        st.header("🏷️ Filters")
        category = st.selectbox("Product Category", ALL_CATEGORIES)

    # Build vector store once
    vectorstore = build_vector_store()

    # Search input
    query = st.text_input("Search products…", placeholder="e.g. noise cancelling headphones")
    search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)

    if search_clicked and query:
        if not groq_key:
            st.error("Please enter your Groq API key in the sidebar.")
            return

        # ── Pipeline steps ──
        with st.status("Running hybrid retrieval pipeline…", expanded=True) as status:

            # Step 1: Keyword search
            st.write("**Step 1 →** Keyword search")
            kw_docs = keyword_search(query, category)
            st.write(f"Found **{len(kw_docs)}** keyword matches")

            # Step 2: Vector search
            st.write("**Step 2 →** Vector similarity search")
            vec_docs = vector_search(query, vectorstore, category)
            st.write(f"Found **{len(vec_docs)}** vector matches")

            # Step 3: Merge
            st.write("**Step 3 →** Merging & deduplicating")
            merged = merge_results(kw_docs, vec_docs)
            st.write(f"**{len(merged)}** unique results after merge")

            # Step 4: Metadata filter applied
            st.write(f"**Step 4 →** Category filter: `{category}`")

            status.update(label="Pipeline complete ✅", state="complete")

        # ── Display retrieved documents ──
        st.subheader("📄 Retrieved Documents")
        if merged:
            for i, doc in enumerate(merged, 1):
                with st.expander(f"Result {i}: {doc.metadata['type']} – {doc.metadata['category']}"):
                    st.write(doc.page_content)
                    st.json(doc.metadata)
        else:
            st.info("No documents matched your query and filters.")

        # ── AI Response ──
        st.subheader("🤖 AI-Generated Response")
        with st.spinner("Generating response with Groq…"):
            answer = generate_response(query, merged, groq_key)
        st.markdown(answer)

    elif search_clicked:
        st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()
