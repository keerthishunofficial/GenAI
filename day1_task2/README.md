# Math Chain-of-Thought Assistant

A minimal web application that solves multi-step math problems using Chain-of-Thought reasoning with the Groq API and Python tool execution.

## Architecture

- **Frontend**: Streamlit web app for user input and result display
- **Backend**: FastAPI server handling API calls to Groq and tool execution
- **LLM**: Groq API with Llama 3.1 model supporting tool calling
- **Tool Layer**: Python math evaluation script for computational steps

## Features

- Accepts multi-step math problems via web interface
- Generates step-by-step reasoning using CoT prompting
- Optionally executes Python math computations as tools
- Returns structured solutions with intermediate steps
- Secure API key management via environment variables

## Setup

1. Clone or download this project
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Groq API key: `GROQ_API_KEY=your_key_here`
4. Run the backend: `uvicorn backend.main:app --reload`
5. Run the frontend: `streamlit run frontend/app.py`

## Usage

1. Open the Streamlit app in your browser
2. Enter a multi-step math problem (e.g., "Solve: 2x + 3 = 7")
3. Click "Solve" to get step-by-step reasoning and final answer

## Example Problems

- "If a train travels 120 km in 2 hours, what is its average speed?"
- "Calculate the area of a triangle with base 10cm and height 5cm"
- "Solve for x: 3x - 5 = 16"