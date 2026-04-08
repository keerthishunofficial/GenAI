import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

import { ChatGroq } from "@langchain/groq";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { tools } from "./tools.js";

const llm = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  apiKey: process.env.GROQ_API_KEY,
  temperature: 0.1,
});

export const agent = createReactAgent({
  llm,
  tools,
  messageModifier: `
You are a helpful AI assistant that answers questions by reasoning step-by-step.
Use the tavily_search tool when you need current or external information.
Always provide sources when using search results.
Keep responses concise but comprehensive.
  `,
});

// For testing, also export the LLM directly
export { llm };