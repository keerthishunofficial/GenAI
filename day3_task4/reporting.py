import json
import os

def generate_testing_report(state: dict) -> dict:
    """Generates a structured JSON evaluation output for testing."""
    return {
        "workflow_status": state.get("status", "unknown"),
        "error": state.get("error"),
        "latency_per_agent_seconds": state.get("latency_stats", {}),
        "token_usage_summary": state.get("token_usage", {}),
        "total_latency": sum(state.get("latency_stats", {}).values()),
        "total_tokens": sum(
            usage.get("total_tokens", 0) 
            for usage in state.get("token_usage", {}).values()
        )
    }

def generate_observability_report(state: dict) -> str:
    """Generates a readable observability summary mapped to LangSmith."""
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    
    report = f"""### Observability Summary
**Project Name:** `{project_name}`
**Status:** `{'Success' if state.get('status') == 'success' else 'Failed'}`

#### Execution Timeline
"""
    
    if "latency_stats" in state:
        for agent, latency in state["latency_stats"].items():
            report += f"- **{agent}:** {latency:.2f}s\n"
            
    report += "\n#### Agent Interaction Sequence\n"
    report += "1. User Input -> **Researcher** (Retrieves facts from ChromaDB, compiles notes)\n"
    report += "2. **Researcher** -> **Writer** (Receives notes, drafts structured report)\n"
    report += "3. **Writer** -> **Editor** (Refines report for clarity)\n"
    report += "4. **Editor** -> Final Output\n"
    
    if state.get("error"):
        report += f"\n#### Failure Handling Logic Triggered\n- **Error:** {state['error']}\n"
    
    report += f"\n*Check [LangSmith Dashboard](https://smith.langchain.com/) for detailed trace links.*"
    return report

def generate_infographic() -> str:
    """Returns markdown-renderable Mermaid charts for the UI."""
    return """
### Workflow Diagram
```mermaid
graph LR
    A[User Query] --> B(Researcher Agent)
    subgraph Operations
        B -->|ChromaDB Retrieval| C{Vector Store}
        C -->|Facts| B
        B -->|Notes| D(Writer Agent)
        D -->|Draft| E(Editor Agent)
    end
    E --> F[Final Polished Report]
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#bfb,stroke:#333,stroke-width:2px
```

### Agent Responsibility Chart
```mermaid
pie title Role Distribution
    "Research & Context Retrieval" : 35
    "Drafting & Structuring" : 40
    "Editing & Refinement" : 25
```
"""
