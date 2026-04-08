import time
from typing import TypedDict, Optional, Dict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from database import retrieve_facts

# ─── Agent State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    topic: str                          # Healthcare question / condition
    research_notes: str                 # Researcher output
    draft: str                          # Writer output
    final_report: str                   # Editor output
    retrieved_context: str              # Raw ChromaDB facts shown in UI

    latency_stats: Dict[str, float]
    token_usage: Dict[str, Dict[str, int]]
    status: str
    error: Optional[str]

# ─── Token usage helper ───────────────────────────────────────────────────────
def get_usage(ai_message) -> Dict[str, int]:
    usage = ai_message.usage_metadata
    if usage:
        return {
            "input_tokens":  usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens":  usage.get("total_tokens", 0),
        }
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

# ─── LLM ─────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

# ─── AGENT 1: Researcher ─────────────────────────────────────────────────────
def researcher_node(state: AgentState):
    start = time.time()
    topic = state.get("topic", "")

    try:
        # Retrieve relevant healthcare facts from ChromaDB
        facts = retrieve_facts(topic, k=state.get("_num_results", 4))

        sys_msg = SystemMessage(content=(
            "You are a healthcare research assistant. "
            "Your job is to retrieve and summarise healthcare facts from the knowledge base. "
            "Use ONLY the retrieved context provided below. "
            "Do NOT add information that is not in the retrieved context. "
            "Do NOT provide diagnosis or suggest medications."
        ))
        human_msg = HumanMessage(content=(
            f"Healthcare topic: {topic}\n\n"
            f"Retrieved facts from knowledge base:\n{facts}\n\n"
            "Please summarise the retrieved information clearly and factually."
        ))

        response = llm.invoke([sys_msg, human_msg])
        latency  = time.time() - start
        usage    = get_usage(response)

        state["retrieved_context"] = facts
        state["research_notes"]    = response.content
        state["latency_stats"]["Researcher"] = latency
        state["token_usage"]["Researcher"]   = usage
        return state

    except Exception as e:
        state["status"] = "failed"
        state["error"]  = str(e)
        return state

# ─── AGENT 2: Writer ─────────────────────────────────────────────────────────
def writer_node(state: AgentState):
    if state.get("status") == "failed":
        return state

    start = time.time()
    notes = state.get("research_notes", "")
    topic = state.get("topic", "")

    try:
        sys_msg = SystemMessage(content=(
            "You are a healthcare content writer. "
            "Convert the provided research notes into a structured healthcare explanation. "
            "Your response must contain exactly these four sections:\n"
            "  ## Summary\n"
            "  ## Symptoms\n"
            "  ## Risk Factors\n"
            "  ## Prevention Tips\n"
            "Use ONLY the information from the provided research notes. "
            "Do NOT provide diagnosis. Do NOT suggest specific medications."
        ))
        human_msg = HumanMessage(content=(
            f"Healthcare topic: {topic}\n\n"
            f"Research notes:\n{notes}\n\n"
            "Write the structured healthcare explanation."
        ))

        response = llm.invoke([sys_msg, human_msg])
        latency  = time.time() - start
        usage    = get_usage(response)

        state["draft"]  = response.content
        state["latency_stats"]["Writer"] = latency
        state["token_usage"]["Writer"]   = usage
        return state

    except Exception as e:
        state["status"] = "failed"
        state["error"]  = str(e)
        return state

# ─── AGENT 3: Editor ─────────────────────────────────────────────────────────
def editor_node(state: AgentState):
    if state.get("status") == "failed":
        return state

    start = time.time()
    draft = state.get("draft", "")
    topic = state.get("topic", "")

    try:
        sys_msg = SystemMessage(content=(
            "You are a senior healthcare content editor. "
            "Refine the draft healthcare explanation for clarity, safety, and readability. "
            "Ensure the response:\n"
            "  - remains informative and educational\n"
            "  - is non-diagnostic and does not suggest prescriptions\n"
            "  - uses plain language accessible to a general audience\n"
            "  - preserves the four-section structure (Summary, Symptoms, Risk Factors, Prevention Tips)\n"
            "Output a final polished markdown document."
        ))
        human_msg = HumanMessage(content=(
            f"Healthcare topic: {topic}\n\n"
            f"Draft:\n{draft}\n\n"
            "Please refine and produce the final polished healthcare explanation."
        ))

        response = llm.invoke([sys_msg, human_msg])
        latency  = time.time() - start
        usage    = get_usage(response)

        state["final_report"] = response.content
        state["latency_stats"]["Editor"] = latency
        state["token_usage"]["Editor"]   = usage
        state["status"] = "success"
        return state

    except Exception as e:
        state["status"] = "failed"
        state["error"]  = str(e)
        return state

# ─── Build Graph ──────────────────────────────────────────────────────────────
def create_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Writer",     writer_node)
    workflow.add_node("Editor",     editor_node)

    workflow.add_edge(START,        "Researcher")
    workflow.add_edge("Researcher", "Writer")
    workflow.add_edge("Writer",     "Editor")
    workflow.add_edge("Editor",     END)

    return workflow.compile()
