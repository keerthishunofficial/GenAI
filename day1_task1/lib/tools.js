import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

export const tools = [
  new TavilySearchResults({
    maxResults: 3,
    apiKey: process.env.TAVILY_API_KEY,
  }),
];