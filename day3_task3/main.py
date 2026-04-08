import uuid
from typing import TypedDict, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- State Definition ---
class AgentState(TypedDict):
    comment_id: str
    text: str
    status: str
    decision: Optional[str]
    history: List[str]

# --- Nodes ---
def extract(state: AgentState):
    history = state.get("history", []) + ["extract"]
    return {"history": history}

def detect(state: AgentState):
    history = state.get("history", []) + ["detect"]
    text = state["text"].lower()
    
    # Simple bad words list
    bad_words = ["spam", "hate", "offensive", "badword", "idiot", "scam"]
    is_unsafe = any(word in text for word in bad_words)
    
    status = "flagged" if is_unsafe else "safe"
    return {"status": status, "history": history}

def human_approval(state: AgentState):
    history = state.get("history", []) + ["human_approval"]
    # We do nothing here, the logic is handled by HITL external update
    return {"history": history}

def publish(state: AgentState):
    history = state.get("history", []) + ["publish"]
    decision = state.get("decision")
    if not decision: # If it wasn't flagged or if somehow skipped
        decision = "published_automatically"
    
    return {"decision": decision, "history": history}

# --- Routing ---
def route_after_detect(state: AgentState):
    if state["status"] == "flagged":
        return "human_approval"
    return "publish"

# --- Graph Setup ---
builder = StateGraph(AgentState)
builder.add_node("extract", extract)
builder.add_node("detect", detect)
builder.add_node("human_approval", human_approval)
builder.add_node("publish", publish)

builder.add_edge(START, "extract")
builder.add_edge("extract", "detect")
builder.add_conditional_edges("detect", route_after_detect, {"human_approval": "human_approval", "publish": "publish"})
builder.add_edge("human_approval", "publish")
builder.add_edge("publish", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["human_approval"])

# --- FastAPI App ---
app = FastAPI(title="HITL Content Moderation Pipeline")

# Basic datastore for tracking created threads
threads_store = []

class CommentSubmission(BaseModel):
    text: str

class ModerationAction(BaseModel):
    decision: str  # "approve" or "reject"

@app.post("/simulate_dataset")
def simulate_dataset():
    """Populates the pipeline with a mix of safe and unsafe comments."""
    dataset = [
        "This lesson was incredibly helpful, thanks!",
        "Click here to win a free iphone! 100% not a scam!",
        "The audio quality in the second half is a bit poor.",
        "You are an idiot if you don't understand this."
    ]
    
    results = []
    for text in dataset:
        thread_id = str(uuid.uuid4())
        threads_store.append(thread_id)
        
        initial_state = {
            "comment_id": thread_id,
            "text": text,
            "status": "pending",
            "decision": None,
            "history": []
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the graph. If it hits an interrupt, it will pause.
        graph.invoke(initial_state, config)
        
        results.append({"thread_id": thread_id, "text": text})
        
    return {"message": "Dataset simulated", "threads": results}

@app.post("/comment")
def submit_comment(comment: CommentSubmission):
    thread_id = str(uuid.uuid4())
    threads_store.append(thread_id)
    
    initial_state = {
        "comment_id": thread_id,
        "text": comment.text,
        "status": "pending",
        "decision": None,
        "history": []
    }
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(initial_state, config)
    
    return {"message": "Comment submitted", "thread_id": thread_id}

@app.get("/pending")
def get_pending_approvals():
    """Returns all comments currently paused waiting for human approval."""
    pending = []
    for thread_id in threads_store:
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = graph.get_state(config)
        
        # If the workflow is blocked, the 'next' tuple will contain the node it's waiting to enter
        if state_snapshot.next and state_snapshot.next[0] == "human_approval":
            pending.append({
                "thread_id": thread_id,
                "text": state_snapshot.values.get("text"),
                "status": state_snapshot.values.get("status")
            })
            
    return {"pending_approvals": pending}

@app.post("/moderate/{thread_id}")
def moderate_comment(thread_id: str, action: ModerationAction):
    if thread_id not in threads_store:
        raise HTTPException(status_code=404, detail="Thread not found")
        
    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = graph.get_state(config)
    
    if not state_snapshot.next or state_snapshot.next[0] != "human_approval":
        raise HTTPException(status_code=400, detail="Comment is not waiting for approval")
        
    if action.decision not in ["approve", "reject"]:
        raise HTTPException(status_code=400, detail="Decision must be 'approve' or 'reject'")
    
    decision_text = "published_by_moderator" if action.decision == "approve" else "rejected_by_moderator"
    
    # 1. Update the state with the human decision
    graph.update_state(config, {"decision": decision_text})
    
    # 2. Resume the graph execution by invoking with None
    graph.invoke(None, config)
    
    return {"message": f"Comment {action.decision}d successfully", "thread_id": thread_id}

@app.get("/report")
def get_report():
    """Summary of all processed comments."""
    report_items = []
    
    for thread_id in threads_store:
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = graph.get_state(config)
        vals = state_snapshot.values
        
        # Note: If no graph invoke has occurred yet, vals could be somewhat empty, but here we invoke immediately. 
        is_pending = bool(state_snapshot.next and state_snapshot.next[0] == "human_approval")
        
        report_items.append({
            "thread_id": thread_id,
            "text": vals.get("text"),
            "status": vals.get("status"),
            "decision": vals.get("decision", "pending_moderation" if is_pending else "unknown"),
            "is_pending": is_pending,
            "history": vals.get("history", [])
        })
        
    return {"total_threads": len(threads_store), "details": report_items}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
