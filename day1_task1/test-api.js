import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

import { GoogleGenerativeAI } from "@google/generative-ai";

const apiKey = process.env.GOOGLE_API_KEY;
console.log("API Key exists:", !!apiKey);
console.log("API Key value:", apiKey);
console.log("API Key length:", apiKey?.length);

const genAI = new GoogleGenerativeAI(apiKey);

async function testModels() {
  try {
    // Try listing available models
    const models = await genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
    console.log("Gemini 2.0 Flash available");
  } catch (e) {
    console.log("Gemini 2.0 Flash error:", e.message);
  }

  try {
    const models = await genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    console.log("Gemini 1.5 Flash available");
  } catch (e) {
    console.log("Gemini 1.5 Flash error:", e.message);
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
    const result = await model.generateContent("2+2");
    console.log("Test successful:", result.response.text());
  } catch (error) {
    console.error("Error:", error.message);
  }
}

testModels();
