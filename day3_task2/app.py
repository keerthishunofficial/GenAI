import streamlit as st
import pandas as pd
import time
import sqlite3
from typing import TypedDict, List, Dict, Any, Annotated
import operator
import uuid

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ==========================================
# 1. State Definition
# ==========================================
# Use TypedDict to define the state being passed between nodes
class WorkflowState(TypedDict):
    raw_data: List[Dict[str, Any]]
    cleaned_data: List[Dict[str, Any]]
    sentiment_results: List[Dict[str, Any]]
    keyword_results: List[Dict[str, Any]]
    final_results: List[Dict[str, Any]]
    trace: Annotated[List[str], operator.add]  # Append-only list to track execution order

# ==========================================
# 2. Node Functions
# ==========================================
def clean_data_node(state: WorkflowState):
    """Clean the raw data by removing entries with missing feedback text."""
    raw_data = state.get("raw_data", [])
    # Basic cleaning: drop rows missing 'feedback_text'
    cleaned = [row for row in raw_data if pd.notna(row.get("feedback_text")) and str(row.get("feedback_text")).strip() != "nan" and str(row.get("feedback_text")).strip() != ""]
    return {
        "cleaned_data": cleaned,
        "trace": ["clean_data: Removed invalid/empty feedback entries."]
    }

def analyze_sentiment_node(state: WorkflowState):
    """Mock parallel node for sentiment analysis."""
    cleaned_data = state.get("cleaned_data", [])
    results = []
    for row in cleaned_data:
        rating = float(row.get("rating", 0))
        # Simple heuristic
        if rating >= 4:
            sentiment = "Positive"
        elif rating == 3:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"
        results.append({"id": row["id"], "sentiment": sentiment})
    
    # Simulate processing time
    time.sleep(1)
    return {
        "sentiment_results": results,
        "trace": ["analyze_sentiment: Completed sentiment classification."]
    }

def extract_keywords_node(state: WorkflowState):
    """Mock parallel node for extracting keywords."""
    cleaned_data = state.get("cleaned_data", [])
    results = []
    
    # Mock some keywords depending on length
    for row in cleaned_data:
        text = str(row.get("feedback_text", ""))
        words = text.split()
        keywords = [words[i] for i in range(min(2, len(words)))] if words else ["none"]
        results.append({"id": row["id"], "keywords": keywords})
    
    # Simulate processing time
    time.sleep(1)
    return {
        "keyword_results": results,
        "trace": ["extract_keywords: Completed keyword extraction."]
    }

def merge_results_node(state: WorkflowState):
    """Merge parallel results into final format."""
    cleaned_data = state.get("cleaned_data", [])
    sentiments = {res["id"]: res["sentiment"] for res in state.get("sentiment_results", [])}
    keywords = {res["id"]: res["keywords"] for res in state.get("keyword_results", [])}
    
    final = []
    for row in cleaned_data:
        final_row = {
            **row,
            "sentiment": sentiments.get(row["id"], "Unknown"),
            "keywords": keywords.get(row["id"], [])
        }
        final.append(final_row)
        
    return {
        "final_results": final,
        "trace": ["merge_results: Combined sentiment and keywords."]
    }

# ==========================================
# 3. Graph Definition & Setup
# ==========================================
def build_graph():
    builder = StateGraph(WorkflowState)
    
    # Add nodes
    builder.add_node("clean_data", clean_data_node)
    builder.add_node("analyze_sentiment", analyze_sentiment_node)
    builder.add_node("extract_keywords", extract_keywords_node)
    builder.add_node("merge_results", merge_results_node)
    
    # Connect nodes: START -> clean_data
    builder.add_edge(START, "clean_data")
    
    # clean_data branches to both analyze_sentiment AND extract_keywords (Parallel Execution)
    builder.add_edge("clean_data", "analyze_sentiment")
    builder.add_edge("clean_data", "extract_keywords")
    
    # Both parallel nodes merge into merge_results
    builder.add_edge("analyze_sentiment", "merge_results")
    builder.add_edge("extract_keywords", "merge_results")
    
    # End node
    builder.add_edge("merge_results", END)
    
    return builder

# Establish single SQLite connection for Streamlit lifecycle caching
@st.cache_resource
def get_graph_and_memory():
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    builder = build_graph()
    # Compile graph with HITL: Interrupt execution after 'clean_data' completes
    # saving the state in 'memory'.
    graph = builder.compile(checkpointer=memory, interrupt_after=["clean_data"])
    return graph

# ==========================================
# 4. Streamlit UI Components
# ==========================================
def main():
    st.set_page_config(page_title="Customer Feedback ETL", layout="wide")
    st.title("LangGraph Feedback Analytics Workflow")
    st.markdown("Features: **Human-in-the-Loop**, **Checkpointing**, **Parallel Execution**")

    # Initialize a consistent thread ID for checkpoints for this Streamlit session
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    graph = get_graph_and_memory()
    
    # Audit & Observability Report UI
    with st.expander("📊 View Audit & Observability Report"):
        st.write("This report extracts the internal state of all workflow executions stored in the local SQLite checkpoint database.")
        
        # Connect to DB to get all thread IDs
        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        threads = cursor.fetchall()
        
        for t in threads:
            t_id = t[0]
            t_config = {"configurable": {"thread_id": t_id}}
            history = list(graph.get_state_history(t_config))
            history.reverse()
            if not history:
                continue
                
            st.markdown(f"### Session Thread: `{t_id}`")
            st.markdown(f"**Total Workflow Steps:** {len(history)}")
            
            final_state = history[-1].values
            
            # Trace
            traces = final_state.get("trace", [])
            if traces:
                st.code("\n".join(traces), language="text")
                
            # Audit
            raw_count = len(final_state.get("raw_data", []))
            clean_count = len(final_state.get("cleaned_data", []))
            final_count = len(final_state.get("final_results", []))
            
            st.markdown(f"- **Raw Entries Received**: {raw_count}\n- **Entries After Cleaning**: {clean_count} ({(raw_count - clean_count)} dropped)\n- **Final Processed Entries**: {final_count}")
            
            # Progression
            progression = []
            for i, state in enumerate(history):
                node = state.metadata.get("source", "unknown") if state.metadata else "unknown"
                next_nodes = ", ".join(state.next) if state.next else "END"
                progression.append({"Step": i+1, "Node Reached": node, "Next Node(s)": next_nodes})
            st.table(progression)
            st.divider()

    # Left column for Actions, Right column for Progress and Results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Step 1: Upload Data")
        uploaded_file = st.file_uploader("Upload 'sample_feedback.csv'", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            
            if st.button("Start Workflow", type="primary"):
                # Convert DF to list of dicts for the graph state
                raw_data = df.to_dict(orient="records")
                initial_state = {"raw_data": raw_data, "trace": ["Workflow started."]}
                
                with st.spinner("Processing to cleaning phase..."):
                    graph.invoke(initial_state, config=config)
                st.success("Cleaning complete. Waiting for your approval.")
                st.rerun()

    # Determine graph execution status
    snapshot = graph.get_state(config)
    
    with col2:
        if snapshot and snapshot.values.get("trace"):
            st.header("Workflow Trace & Checkpoints")
            # Present trace nicely
            for msg in snapshot.values["trace"]:
                st.caption(f"✓ {msg}")

        # HITL Interruption state visualization
        if snapshot and snapshot.next and "analyze_sentiment" in snapshot.next:
            st.warning("⚠️ Human-in-the-Loop Intervention Required")
            st.markdown("The system has cleaned the data. Review the changes before proceeding to full analysis.")
            
            cleaned_df = pd.DataFrame(snapshot.values["cleaned_data"])
            st.dataframe(cleaned_df)
            
            st.markdown("Do you approve the cleaned data to be sent for Sentiment Analysis and Keyword Extraction?")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Approve & Continue", use_container_width=True):
                    with st.spinner("Running sentiment & keyword analysis in parallel..."):
                        snapshot.values["trace"].append("User approved cleaned data.")
                        # To resume simply call invoke with None. State is loaded from Sqlite checkpoint!
                        graph.invoke(None, config=config)
                    st.success("Workflow finished!")
                    st.rerun()
            with col_b:
                if st.button("❌ Reject & Abort", type="primary", use_container_width=True):
                    snapshot.values["trace"].append("User rejected cleaned data. Workflow aborted.")
                    graph.update_state(config, {"trace": ["Workflow aborted by user."]}) # Just recording the trace
                    st.error("Workflow rejected and stopped.")
                    # In a real app we might reset state or transition to an END node manually.
                    st.session_state.thread_id = str(uuid.uuid4()) # easiest reset
                    st.rerun()

        # Final Results if finished
        if snapshot and snapshot.values.get("final_results"):
            st.success("✅ Workflow Successfully Completed")
            st.subheader("Final Processed Data")
            final_df = pd.DataFrame(snapshot.values["final_results"])
            st.dataframe(final_df, use_container_width=True)
            
            # Simple bar chart on sentiment
            st.subheader("Sentiment Summary")
            sentiment_counts = final_df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            
            if st.button("Start New Workflow"):
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()

if __name__ == "__main__":
    main()
