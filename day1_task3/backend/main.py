from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from analysis import analyze_code_ast, agent_loop

app = FastAPI(title="Self-Reflecting Code Review Agent", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeReviewRequest(BaseModel):
    code: str

@app.post("/review")
async def review_code(request: CodeReviewRequest):
    """Endpoint to review Python code with AST analysis and agent loop."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")

    # Step 1: AST Analysis
    ast_result = analyze_code_ast(request.code)
    if not ast_result["valid"]:
        raise HTTPException(status_code=400, detail=ast_result["issues"])

    # Step 2 & 3: Agent loop (suggestions + reflection)
    feedback = agent_loop(request.code, ast_result)
    
    return {
        "ast_issues": ast_result["issues"],
        "round_1_suggestions": feedback["round_1"],
        "round_2_suggestions": feedback["round_2"]
    }

@app.get("/")
async def root():
    return {"message": "Self-Reflecting Code Review Agent API - Submit Python code to /review endpoint"}