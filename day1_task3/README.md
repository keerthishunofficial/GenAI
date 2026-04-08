# Self-Reflecting Code Review Agent

A simple web app for reviewing Python code using AST analysis and iterative AI feedback.

## Architecture

- **Backend**: FastAPI with Python AST and Hugging Face Transformers for LLM-based suggestions.
- **Frontend**: React with Monaco Editor for code input.
- **Agent Loop**: Iterative refinement of code review suggestions.

## Setup and Run

### Prerequisites
- Docker and Docker Compose
- Or Node.js and Python for local development

### Using Docker Compose (Recommended)
1. Clone or navigate to the project directory.
2. Run `docker-compose up --build`
3. Open http://localhost:3000 for frontend, backend at http://localhost:8000

### Local Development
1. Backend:
   - `cd backend`
   - `pip install -r requirements.txt`
   - `uvicorn main:app --reload`

2. Frontend:
   - `cd frontend`
   - `npm install`
   - `npm start`

## Usage
- Paste Python code in the editor.
- Click "Review Code" to get feedback.
- View AST issues and iterative AI suggestions.

## Tech Stack
- Backend: Python, FastAPI, AST, Transformers
- Frontend: React, Monaco Editor
- AI: T5-small model for generation