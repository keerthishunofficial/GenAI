# ReAct QA Agent

A simple web application implementing a ReAct (Reason + Act) agent that answers questions using step-by-step reasoning and web search.

## Features

- ReAct reasoning loop with Thought → Action → Observation → Final Answer
- Web search integration using Tavily API
- Streaming responses for real-time agent output
- Clean, minimal UI built with Next.js and Tailwind CSS

## Setup

1. Clone or download the project
2. Install dependencies: `npm install`
3. Get API keys:
   - Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Tavily API key from [Tavily](https://tavily.com/)
4. Add keys to `.env.local`:
   ```
   GOOGLE_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   ```
5. Run the app: `npm run dev`
6. Open http://localhost:3000

## Usage

Enter a question in the input field and click "Ask". The agent will reason step-by-step and provide an answer based on web search results when needed.

## Architecture

- **Frontend:** Next.js with React and Tailwind CSS
- **Backend:** Next.js API routes
- **Agent:** LangChain.js with Google Gemini and Tavily tools
- **Streaming:** Server-sent events for real-time responses

## Extending

- Add more tools (calculator, code execution, etc.)
- Implement conversation memory
- Add authentication and rate limiting
- Deploy to Vercel or similar platform