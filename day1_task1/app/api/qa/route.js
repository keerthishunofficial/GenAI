import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

import { llm } from "@/lib/agent.js";
import { HumanMessage } from "@langchain/core/messages";

export async function POST(request) {
  try {
    const { question } = await request.json();
    
    if (!question) {
      return Response.json({ error: "Question is required" }, { status: 400 });
    }

    console.log("Processing question:", question);

    // Use Groq LLM for fast responses (free tier available)
    try {
      const response = await llm.invoke([new HumanMessage(question)]);
      
      const encoder = new TextEncoder();
      const readable = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            content: response.content,
            type: "ai"
          })}\n\n`));
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();
        },
      });

      return new Response(readable, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        },
      });
    } catch (llmError) {
      console.error("LLM error:", llmError.message);
      
      return Response.json({ 
        error: "LLM call failed",
        details: llmError.message 
      }, { status: 500 });
    }
  } catch (error) {
    console.error("Agent error:", error);
    return Response.json({ 
      error: "Failed to process question",
      details: error.message 
    }, { status: 500 });
  }
}