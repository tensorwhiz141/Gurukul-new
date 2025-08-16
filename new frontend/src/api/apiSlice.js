import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { API_BASE_URL } from "../config";

// Log the API base URL for debugging
console.log("API Base URL:", API_BASE_URL);

// Create a base API slice with shared configuration
export const apiSlice = createApi({
  reducerPath: "api",
  baseQuery: fetchBaseQuery({
    baseUrl: API_BASE_URL,
    prepareHeaders: (headers, { getState }) => {
      // You can add auth headers here if needed
      return headers;
    },
    timeout: 600000, // 10 minute timeout for Knowledge Store operations
  }),
  tagTypes: [
    "Subjects",
    "Lectures",
    "Tests",
    "ChatHistory",
    "Summary",
    "AgentOutput",
    "AgentLogs",
    "User",
    "LearningTask",
    "FinancialSimulation",
    "PdfChat",
  ],
  endpoints: () => ({}),
});

// Export hooks for usage in components
export const {
  // These will be filled by the injected endpoints
} = apiSlice;
