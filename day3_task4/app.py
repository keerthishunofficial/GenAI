import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

from workflow import create_workflow
from database import retrieve_facts
from reporting import generate_testing_report, generate_observability_report, generate_infographic

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background: linear-gradient(135deg, #0a1628, #0d2137, #0a2744); }

    /* Sidebar */
    [data-testid="stSidebar"] { background: rgba(255,255,255,0.04); backdrop-filter: blur(12px); }

    /* Agent cards */
    .agent-card {
        border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: 0.6rem;
        border: 1px solid rgba(255,255,255,0.12); backdrop-filter: blur(8px);
    }
    .card-researcher { background: rgba(34,197,94,0.15); }
    .card-writer     { background: rgba(59,130,246,0.18); }
    .card-editor     { background: rgba(168,85,247,0.18); }

    /* Metric boxes */
    .metric-box {
        text-align: center; border-radius: 12px; padding: 1rem; margin: 0.25rem;
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10);
    }
    .metric-box h2 { margin:0; font-size:1.8rem; font-weight:800; }
    .metric-box p  { margin:0; font-size:0.75rem; opacity:0.65; }

    /* Topic chips */
    .topic-chip {
        display:inline-block; background:rgba(34,197,94,0.2);
        border:1px solid rgba(34,197,94,0.5); border-radius:20px;
        padding:3px 14px; margin:3px; font-size:0.8rem;
    }

    /* Report body */
    .report-body {
        background: rgba(255,255,255,0.04); border-radius:12px; padding:1.5rem;
        border:1px solid rgba(255,255,255,0.10); line-height:1.8;
    }

    /* Retrieved context box */
    .context-box {
        background: rgba(34,197,94,0.08); border-radius:10px;
        border-left: 4px solid rgba(34,197,94,0.6);
        padding: 0.8rem 1rem; font-size:0.82rem; line-height:1.6;
    }

    /* Chatbot bubble */
    .chat-user {
        background: rgba(59,130,246,0.25); border-radius:14px 14px 4px 14px;
        padding:0.6rem 1rem; margin:0.4rem 0; text-align:right;
        border:1px solid rgba(59,130,246,0.4);
    }
    .chat-bot {
        background: rgba(34,197,94,0.15); border-radius:14px 14px 14px 4px;
        padding:0.6rem 1rem; margin:0.4rem 0;
        border:1px solid rgba(34,197,94,0.3);
    }
    .chat-disclaimer {
        font-size:0.7rem; opacity:0.55; margin-top:0.3rem;
        border-top:1px solid rgba(255,255,255,0.08); padding-top:0.3rem;
    }

    /* Safety badge */
    .safety-badge {
        display:inline-block; background:rgba(239,68,68,0.2);
        border:1px solid rgba(239,68,68,0.5); border-radius:8px;
        padding:2px 12px; font-size:0.75rem; color:#fca5a5;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg,#1d6348,#0d2137) !important;
        color:white !important; border:1px solid rgba(34,197,94,0.5) !important;
        border-radius:10px !important; font-weight:700 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(34,197,94,0.3) !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg,#1d4ed8,#0a1628) !important;
        color:white !important; border:none !important; border-radius:10px !important;
        font-weight:600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "topic_history" not in st.session_state:
    st.session_state.topic_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ─── Chatbot LLM helper ───────────────────────────────────────────────────────
@st.cache_resource
def get_chat_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

def ask_chatbot(question: str) -> str:
    """RAG chatbot: retrieve facts, then answer using only the retrieved context."""
    facts = retrieve_facts(question, k=4)
    if "No relevant facts" in facts:
        return "I could not find this in the knowledge base."

    chat_llm = get_chat_llm()
    sys_msg = SystemMessage(content=(
        "You are a healthcare knowledge assistant. "
        "Answer the user's question ONLY using the context provided below. "
        "If the context does not contain enough information, respond: "
        "'I could not find this in the knowledge base.' "
        "Do NOT provide diagnosis. Do NOT suggest medications. "
        "Stay educational and informative only.\n\n"
        f"Context:\n{facts}"
    ))
    human_msg = HumanMessage(content=question)
    response = chat_llm.invoke([sys_msg, human_msg])
    return response.content

# ─── Workflow cache ───────────────────────────────────────────────────────────
@st.cache_resource
def get_graph():
    return create_workflow()

graph = get_graph()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Assistant Settings")
    st.markdown("---")

    num_results = st.slider("📚 ChromaDB Results per Query", 2, 8, 4,
                            help="Number of facts retrieved per query.")

    st.markdown("---")
    st.markdown("### 📋 Topic History")
    if st.session_state.topic_history:
        for t in reversed(st.session_state.topic_history[-5:]):
            st.markdown(f'<span class="topic-chip">🏥 {t}</span>', unsafe_allow_html=True)
        if st.button("🗑️ Clear History"):
            st.session_state.topic_history = []
            st.rerun()
    else:
        st.caption("No topics searched yet.")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    st.metric("Pipeline Runs", st.session_state.run_count)
    if st.session_state.last_result:
        total_toks = sum(
            v.get("total_tokens", 0)
            for v in st.session_state.last_result.get("token_usage", {}).values()
        )
        st.metric("Last Run Tokens", f"{total_toks:,}")
        total_lat = sum(st.session_state.last_result.get("latency_stats", {}).values())
        st.metric("Last Run Latency", f"{total_lat:.1f}s")

    st.markdown("---")
    st.markdown('<span class="safety-badge">⚠️ Not a diagnostic tool</span>', unsafe_allow_html=True)
    st.caption("This assistant provides educational health information only. Always consult a qualified healthcare professional.")
    st.markdown("---")
    st.markdown("[🔍 LangSmith Dashboard](https://smith.langchain.com/)")

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:1.5rem 0 0.5rem 0;">
    <h1 style="font-size:2.6rem; font-weight:900;
        background:linear-gradient(135deg,#22c55e,#3b82f6,#a855f7);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🏥 Healthcare Knowledge Assistant
    </h1>
    <p style="opacity:0.6; font-size:0.95rem; margin-top:-0.4rem;">
        LangGraph · Groq · ChromaDB · LangSmith — Retrieval-grounded health education
    </p>
    <span style="background:rgba(239,68,68,0.2);border:1px solid rgba(239,68,68,0.4);
        border-radius:8px;padding:3px 14px;font-size:0.75rem;color:#fca5a5;">
        ⚠️ Educational use only — not a substitute for professional medical advice
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── TWO-COLUMN LAYOUT: Pipeline | Chatbot ───────────────────────────────────
main_col, chat_col = st.columns([3, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Research Pipeline
# ══════════════════════════════════════════════════════════════════════════════
with main_col:
    # ── Input row ────────────────────────────────────────────────────────────
    inp_col, btn_col = st.columns([5, 1])
    with inp_col:
        topic = st.text_input(
            "🔎 Healthcare Topic",
            placeholder="e.g., Hypertension, Type 2 Diabetes, Anemia, Heart Disease...",
            label_visibility="collapsed"
        )
    with btn_col:
        run_clicked = st.button("▶ Analyse", use_container_width=True)

    # ── Quick-topic chips ─────────────────────────────────────────────────────
    st.markdown("**Select a condition:**")
    chip_cols = st.columns(6)
    quick_topics = ["Diabetes", "Hypertension", "Heart Disease", "Anemia", "Nutrition", "Preventive Healthcare"]
    for i, qt in enumerate(quick_topics):
        with chip_cols[i]:
            if st.button(qt, key=f"chip_{i}"):
                topic = qt
                run_clicked = True

    st.markdown("---")

    # ── Pipeline execution ────────────────────────────────────────────────────
    if run_clicked:
        if not topic.strip():
            st.warning("⚠️ Please enter a healthcare topic or select one above.")
        elif not os.getenv("GROQ_API_KEY"):
            st.error("❌ `GROQ_API_KEY` not set in `.env`.")
        else:
            if topic not in st.session_state.topic_history:
                st.session_state.topic_history.append(topic)
            st.session_state.run_count += 1

            progress_bar  = st.progress(0, text="⏳ Initialising healthcare pipeline...")
            status_banner = st.empty()
            status_banner.info("🔄 Starting pipeline — agents are loading...")

            # Agent cards
            st.markdown("### 🤖 Agent Execution")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="agent-card card-researcher">'
                            '<b>🔬 Researcher</b><br>'
                            '<small>Retrieves healthcare facts from ChromaDB</small></div>',
                            unsafe_allow_html=True)
                r_status  = st.empty()
                r_content = st.empty()
            with c2:
                st.markdown('<div class="agent-card card-writer">'
                            '<b>✍️ Writer</b><br>'
                            '<small>Structures into Summary, Symptoms, Risk Factors, Prevention Tips</small></div>',
                            unsafe_allow_html=True)
                w_status  = st.empty()
                w_content = st.empty()
            with c3:
                st.markdown('<div class="agent-card card-editor">'
                            '<b>🎨 Editor</b><br>'
                            '<small>Refines for clarity, safety, and readability</small></div>',
                            unsafe_allow_html=True)
                e_status  = st.empty()
                e_content = st.empty()

            r_status.markdown("⏳ *Waiting...*")
            w_status.markdown("⏳ *Waiting...*")
            e_status.markdown("⏳ *Waiting...*")

            # Initial state
            initial_state = {
                "topic":            topic,
                "research_notes":   "",
                "draft":            "",
                "final_report":     "",
                "retrieved_context":"",
                "latency_stats":    {},
                "token_usage":      {},
                "status":           "running",
                "error":            None,
                "_num_results":     num_results,
            }

            final_state = None
            step = 0

            try:
                for s in graph.stream(initial_state):
                    step += 1
                    progress_bar.progress(min(step / 3, 1.0), text=f"Step {step}/3")

                    if "Researcher" in s:
                        r_status.markdown("✅ *Complete*")
                        snippet = s["Researcher"]["research_notes"][:220]
                        r_content.markdown(
                            f'<div class="report-body"><small>{snippet}…</small></div>',
                            unsafe_allow_html=True
                        )
                        status_banner.info("✍️ Writer is structuring the healthcare explanation…")
                        final_state = s["Researcher"]

                    elif "Writer" in s:
                        w_status.markdown("✅ *Complete*")
                        snippet = s["Writer"]["draft"][:220]
                        w_content.markdown(
                            f'<div class="report-body"><small>{snippet}…</small></div>',
                            unsafe_allow_html=True
                        )
                        status_banner.info("🎨 Editor is refining for safety and clarity…")
                        final_state = s["Writer"]

                    elif "Editor" in s:
                        e_status.markdown("✅ *Complete*")
                        snippet = s["Editor"]["final_report"][:220]
                        e_content.markdown(
                            f'<div class="report-body"><small>{snippet}…</small></div>',
                            unsafe_allow_html=True
                        )
                        status_banner.success("🎉 Healthcare explanation ready!")
                        final_state = s["Editor"]

            except Exception as exc:
                st.error(f"🚨 Pipeline error: {exc}")
                st.stop()

            if not final_state:
                st.error("Pipeline returned no output.")
                st.stop()

            st.session_state.last_result = final_state

            if final_state.get("status") == "failed":
                st.error(f"Pipeline failed: {final_state.get('error')}")
                st.stop()

            # ── Metrics strip ─────────────────────────────────────────────────
            st.markdown("### 📊 Pipeline Metrics")
            lat  = final_state.get("latency_stats", {})
            tok  = final_state.get("token_usage", {})
            total_lat  = sum(lat.values())
            total_toks = sum(v.get("total_tokens", 0) for v in tok.values())
            fastest    = min(lat, key=lat.get) if lat else "—"

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-box"><h2>⏱ {total_lat:.1f}s</h2><p>Total Latency</p></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><h2>🔤 {total_toks:,}</h2><p>Total Tokens</p></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-box"><h2>🔬 {lat.get("Researcher",0):.1f}s</h2><p>Researcher Latency</p></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-box"><h2>⚡ {fastest}</h2><p>Fastest Agent</p></div>', unsafe_allow_html=True)

            with st.expander("📈 Detailed Token Breakdown"):
                tok_c = st.columns(3)
                for idx, (agent, usage) in enumerate(tok.items()):
                    with tok_c[idx]:
                        st.markdown(f"**{agent}**")
                        st.metric("Input",  usage.get("input_tokens", 0))
                        st.metric("Output", usage.get("output_tokens", 0))
                        st.metric("Total",  usage.get("total_tokens", 0))

            # ── Retrieved context ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🗂️ Retrieved Knowledge Base Facts")
            raw_ctx = final_state.get("retrieved_context", "")
            if raw_ctx:
                st.markdown(
                    f'<div class="context-box">{raw_ctx.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("No context retrieved.")

            # ── Final report ──────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📄 Structured Healthcare Explanation")
            report_text = final_state["final_report"]

            view_mode = st.radio("View:", ["📝 Rendered", "🔤 Raw Markdown"], horizontal=True, key="view_mode")
            if view_mode == "📝 Rendered":
                st.markdown(f'<div class="report-body">{report_text}</div>', unsafe_allow_html=True)
            else:
                st.code(report_text, language="markdown")

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Download Explanation (Markdown)",
                    data=report_text,
                    file_name=f"health_{topic.replace(' ','_').lower()}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with dl2:
                json_data = json.dumps(generate_testing_report(final_state), indent=2)
                st.download_button(
                    "⬇️ Download Testing Report (JSON)",
                    data=json_data,
                    file_name=f"test_{topic.replace(' ','_').lower()}.json",
                    mime="application/json",
                    use_container_width=True
                )

            # ── Report tabs ───────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🔍 Observability & Reports")
            tab1, tab2, tab3, tab4 = st.tabs(
                ["🧪 Testing Report", "📡 Observability", "📊 Infographics", "🔎 Full Agent Outputs"]
            )
            with tab1:
                test_report = generate_testing_report(final_state)
                st.success(f"Status: **{test_report['workflow_status'].upper()}**")
                st.json(test_report)

            with tab2:
                st.markdown(generate_observability_report(final_state))
                project = os.getenv("LANGCHAIN_PROJECT", "default")
                st.info(f"📌 Traces logged to LangSmith project: **`{project}`**")
                st.markdown("[🖥️ Open LangSmith Dashboard](https://smith.langchain.com/)")

            with tab3:
                st.markdown(generate_infographic())
                if lat:
                    st.markdown("#### ⏱ Latency per Agent")
                    st.bar_chart({"Researcher": lat.get("Researcher", 0),
                                  "Writer":     lat.get("Writer", 0),
                                  "Editor":     lat.get("Editor", 0)})

            with tab4:
                st.markdown("#### 🔬 Researcher Notes")
                with st.expander("View", expanded=False):
                    st.markdown(final_state.get("research_notes", "N/A"))
                st.markdown("#### ✍️ Writer Draft")
                with st.expander("View", expanded=False):
                    st.markdown(final_state.get("draft", "N/A"))
                st.markdown("#### 🎨 Final Report")
                with st.expander("View", expanded=True):
                    st.markdown(final_state.get("final_report", "N/A"))

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Floating Chatbot Panel
# ══════════════════════════════════════════════════════════════════════════════
with chat_col:
    st.markdown("### 💬 Ask the Assistant")
    st.caption("RAG-grounded · non-diagnostic")
    st.markdown('<span class="safety-badge">⚠️ Educational only</span>', unsafe_allow_html=True)
    st.markdown("")

    # Render existing chat history
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🏥 {msg["content"]}'
                        f'<div class="chat-disclaimer">Not a diagnosis. Educational info only.</div>'
                        f'</div>', unsafe_allow_html=True)

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_q = st.text_area(
            "Your question:",
            placeholder="Ask about symptoms, risk factors, prevention, or conditions...",
            height=90,
            label_visibility="collapsed"
        )
        send = st.form_submit_button("Send ➤", use_container_width=True)

    if send and user_q.strip():
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not set.")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_q.strip()})
            with st.spinner("Searching knowledge base..."):
                answer = ask_chatbot(user_q.strip())
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()

    # Clear chat button
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
