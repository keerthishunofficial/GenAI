from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
from .math_tool import evaluate_math_expression
import sys
from pathlib import Path

# Load environment variables from .env file in the project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="Math CoT Assistant API")

# Get API key and verify it's loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set. Please add it to .env file")

client = Groq(api_key=api_key)

class MathProblem(BaseModel):
    problem: str

class ReasoningStep(BaseModel):
    step_number: int
    reasoning: str
    computation: str = None
    result: str = None

class Solution(BaseModel):
    steps: List[ReasoningStep]
    final_answer: str

@app.post("/solve", response_model=Solution)
async def solve_math_problem(problem: MathProblem):
    try:
        # CoT prompt for multi-step reasoning
        system_prompt = """You are a mathematical reasoning assistant. Solve multi-step math problems using clear, step-by-step Chain-of-Thought reasoning.

For each problem:
1. Break down the problem into logical steps
2. Show your reasoning for each step
3. Perform calculations step by step
4. Provide a clear final answer

Format your response with numbered steps for each stage of reasoning. At the end, provide the final answer clearly."""

        user_prompt = f"Solve this math problem step by step:\n\n{problem.problem}\n\nProvide your answer in this format:\nStep 1: [reasoning]\nStep 2: [reasoning]\n...\nFinal Answer: [answer]"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )

        content = response.choices[0].message.content

        # Parse the content into steps
        steps = []
        step_number = 1
        final_answer = ""
        
        lines = content.split('\n')
        current_step_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a step header
            if line.lower().startswith('step') and ':' in line:
                # Save previous step if exists
                if current_step_text:
                    steps.append(ReasoningStep(
                        step_number=step_number,
                        reasoning=current_step_text.strip()
                    ))
                    step_number += 1
                # Extract step content (everything after the colon)
                current_step_text = line.split(':', 1)[1].strip()
            elif line.lower().startswith('final answer') and ':' in line:
                # Save last reasoning step
                if current_step_text:
                    steps.append(ReasoningStep(
                        step_number=step_number,
                        reasoning=current_step_text.strip()
                    ))
                # Extract final answer
                final_answer = line.split(':', 1)[1].strip()
            else:
                # Continuation of current step
                if current_step_text:
                    current_step_text += " " + line
                else:
                    current_step_text = line
        
        # Handle any remaining step
        if current_step_text and not final_answer:
            steps.append(ReasoningStep(
                step_number=step_number,
                reasoning=current_step_text.strip()
            ))
        
        # If no final answer was extracted, use the last step
        if not final_answer and steps:
            final_answer = steps[-1].reasoning
            # Try to extract just the numeric answer
            import re
            match = re.search(r'(\d+(?:\.\d+)?)\)?$', final_answer)
            if match:
                final_answer = match.group(1)
            else:
                # Get last number-like value
                numbers = re.findall(r'\d+(?:\.\d+)?', final_answer)
                if numbers:
                    final_answer = numbers[-1]

        return Solution(steps=steps if steps else [ReasoningStep(step_number=1, reasoning=content)], final_answer=final_answer if final_answer else "See steps above")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving problem: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Math Chain-of-Thought Assistant API"}