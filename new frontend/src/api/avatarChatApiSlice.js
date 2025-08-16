import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { CHAT_API_BASE_URL } from "../config";

// Log the Chat API base URL for debugging
console.log("Avatar Chat API Base URL:", CHAT_API_BASE_URL);

/**
 * Avatar Chat API Slice
 * Uses the same chat API endpoint as the main chatbot (port 8001)
 * with avatar-specific functionality and context-aware messaging
 */
export const avatarChatApiSlice = createApi({
  reducerPath: "avatarChatApi",
  baseQuery: fetchBaseQuery({
    baseUrl: CHAT_API_BASE_URL,
    prepareHeaders: (headers, { getState }) => {
      // You can add auth headers here if needed
      return headers;
    },
    timeout: 30000, // 30 second timeout for chat responses
  }),
  tagTypes: ["AvatarChatHistory"],
  endpoints: (builder) => ({
    // Send avatar chat message with context
    sendAvatarChatMessage: builder.mutation({
      query: ({ message, userId, llmModel = "grok", pageContext = null }) => {
        // Enhance the message with page context for better AI responses
        const contextualMessage = pageContext 
          ? `${pageContext}\n\nUser Message: ${message}`
          : message;

        return {
          url: `/chatpost?user_id=${userId || "guest-user"}`,
          method: "POST",
          body: {
            message: contextualMessage,
            llm: llmModel,
            type: "avatar_chat_message",
            context: pageContext ? "page_aware" : "general",
          },
        };
      },
      invalidatesTags: ["AvatarChatHistory"],
    }),

    // Fetch avatar chat response
    fetchAvatarChatResponse: builder.query({
      query: (userId) => ({
        url: "/chatbot",
        params: {
          user_id: userId || "guest-user",
          timestamp: new Date().toISOString(),
          type: "avatar_chat_message",
        },
      }),
      transformResponse: (response) => {
        // Handle the case where the API returns an error for first-time users
        if (
          response.error &&
          (response.error === "No queries yet" ||
            response.error === "No queries found for this user")
        ) {
          return {
            message: "Hi! I'm your AI avatar assistant. I can help you with anything on this page. What would you like to know?",
            isFirstQuery: true,
          };
        }
        return response;
      },
      providesTags: ["AvatarChatHistory"],
    }),

    // Get avatar chat history (separate from main chatbot)
    getAvatarChatHistory: builder.query({
      query: (userId) => ({
        url: "/chatbot",
        params: {
          user_id: userId || "guest-user",
          type: "avatar_chat_history",
          timestamp: new Date().toISOString(),
        },
      }),
      transformResponse: (response) => {
        // Handle the case where the API returns an error for first-time users
        if (
          response.error &&
          (response.error === "No queries yet" ||
            response.error === "No queries found for this user")
        ) {
          return {
            messages: [],
            isFirstQuery: true,
          };
        }
        return response;
      },
      providesTags: ["AvatarChatHistory"],
    }),

    // Clear avatar chat history
    clearAvatarChatHistory: builder.mutation({
      query: (userId) => ({
        url: `/chatbot/clear?user_id=${userId || "guest-user"}`,
        method: "DELETE",
        body: {
          type: "avatar_chat_history",
        },
      }),
      invalidatesTags: ["AvatarChatHistory"],
    }),
  }),
});

export const {
  useSendAvatarChatMessageMutation,
  useLazyFetchAvatarChatResponseQuery,
  useGetAvatarChatHistoryQuery,
  useClearAvatarChatHistoryMutation,
} = avatarChatApiSlice;

/**
 * Helper function to format messages for avatar chat
 */
export const formatAvatarMessage = (message, role = "user", model = "grok") => ({
  id: Date.now() + Math.random(),
  role,
  content: message,
  model,
  timestamp: new Date().toISOString(),
  type: "avatar_chat",
});

/**
 * Helper function to create contextual system prompts for avatar chat
 */
export const createAvatarSystemPrompt = (pageContext) => {
  const basePrompt = `You are a helpful AI avatar assistant integrated into the Gurukul learning platform. You appear as a floating 3D avatar and help users with their current tasks. Be friendly, concise, and contextually aware.`;

  if (!pageContext) {
    return basePrompt + " Provide general assistance to the user.";
  }

  const { routeContext, pageData } = pageContext;

  const contextualPrompts = {
    dashboard: "Help the user navigate the dashboard and understand their learning progress.",
    academic: "Assist with subject selection and academic planning.",
    learning: "Help with document analysis, summarization, and learning activities.",
    ai_interaction: "Provide guidance on AI features and chat functionality.",
    assessment: "Support the user with tests and assessments.",
    customization: "Help with avatar selection and personalization options.",
    configuration: "Assist with settings and platform configuration.",
    information: "Provide information about platform features and capabilities.",
  };

  const specificPrompt = contextualPrompts[routeContext.type] || "Provide general assistance.";

  return `${basePrompt} ${specificPrompt} Current context: ${routeContext.description}`;
};
