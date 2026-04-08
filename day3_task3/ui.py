import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Approve / Reject Workflow", layout="wide")

st.title("LangGraph Moderation Dashboard")
st.markdown("This dashboard interacts with our HITL LangGraph + FastAPI backend to process comments.")

# Define tabs for better organization
tab_submit, tab_moderate, tab_report = st.tabs(["📝 Submit & Simulate", "⚖️ Moderation Queue", "📊 Report"])

# --- TAB 1: Submit & Simulate ---
with tab_submit:
    st.header("Simulate Dataset")
    st.write("Push a mixed batch of safe and unsafe comments into the workflow.")
    if st.button("Simulate Dataset"):
        try:
            resp = requests.post(f"{API_URL}/simulate_dataset")
            if resp.status_code == 200:
                st.success("Successfully pushed the simulated dataset into the LangGraph workflow!")
                st.json(resp.json())
            else:
                st.error(f"Failed: {resp.text}")
        except Exception as e:
            st.error(f"Connection Error: Is the FastAPI server running on port 8000? Details: {e}")
            
    st.divider()
    
    st.header("Submit Custom Comment")
    with st.form("comment_form"):
        comment_text = st.text_area("Write a comment", placeholder="Type something here...")
        submitted = st.form_submit_button("Submit")
        if submitted and comment_text:
            try:
                resp = requests.post(f"{API_URL}/comment", json={"text": comment_text})
                if resp.status_code == 200:
                    st.success("Comment submitted to the pipeline!")
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- TAB 2: Moderation Queue ---
with tab_moderate:
    st.header("Pending Approvals")
    st.write("These comments were flagged by the `detect` node and the LangGraph workflow is currently paused waiting for your human-in-the-loop decision.")
    
    # Add a refresh button
    if st.button("Refresh Queue"):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/pending")
        if resp.status_code == 200:
            pending = resp.json().get("pending_approvals", [])
            
            if not pending:
                st.info("No comments are currently waiting for approval. 🎉")
            else:
                for idx, item in enumerate(pending):
                    st.markdown(f"**Thread ID:** `{item['thread_id']}`")
                    st.warning(f"**Flagged Content:** {item['text']}")
                    
                    col1, col2 = st.columns([1, 10])
                    
                    with col1:
                        if st.button("✅ Approve", key=f"approve_{item['thread_id']}"):
                            requests.post(f"{API_URL}/moderate/{item['thread_id']}", json={"decision": "approve"})
                            st.success(f"Thread {item['thread_id']} approved. Resuming workflow...")
                            st.rerun()
                    with col2:
                        if st.button("❌ Reject", key=f"reject_{item['thread_id']}"):
                            requests.post(f"{API_URL}/moderate/{item['thread_id']}", json={"decision": "reject"})
                            st.error(f"Thread {item['thread_id']} rejected. Resuming workflow...")
                            st.rerun()
                            
                    st.divider()
    except Exception as e:
        st.error(f"Could not connect to the backend: {e}")

# --- TAB 3: Report ---
with tab_report:
    st.header("Workflow Report")
    
    if st.button("Refresh Report"):
        st.rerun()
        
    try:
        resp = requests.get(f"{API_URL}/report")
        if resp.status_code == 200:
            data = resp.json()
            st.metric("Total Threads Processed", data.get("total_threads", 0))
            
            # Convert dictionary details to a displayable format
            details = data.get("details", [])
            if details:
                st.dataframe(
                    details,
                    column_config={
                        "thread_id": st.column_config.TextColumn("Thread ID"),
                        "text": st.column_config.TextColumn("Comment Content"),
                        "status": st.column_config.TextColumn("Filter Status"),
                        "decision": st.column_config.TextColumn("Final Decision"),
                        "is_pending": st.column_config.CheckboxColumn("Pending Human?"),
                        "history": st.column_config.ListColumn("Execution Path (Nodes)"),
                    },
                    use_container_width=True
                )
    except Exception as e:
         st.error(f"Could not connect to the backend: {e}")
