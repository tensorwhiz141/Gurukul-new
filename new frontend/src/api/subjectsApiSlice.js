import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { API_BASE_URL } from "../config";

// Create a separate API slice for lesson endpoints using the same base URL
export const lessonApiSlice = createApi({
  reducerPath: "lessonApi",
  baseQuery: fetchBaseQuery({
    baseUrl: API_BASE_URL,
    timeout: 600000, // 10 minute timeout for Knowledge Store (can take 6+ minutes)
  }),
  tagTypes: ["Lessons"],
  endpoints: (builder) => ({
    // Get existing lesson data
    getLessonData: builder.query({
      query: ({ subject, topic }) => ({
        url: `/lessons/${encodeURIComponent(subject)}/${encodeURIComponent(
          topic
        )}`,
        method: "GET",
      }),
      providesTags: (result, error, { subject, topic }) => [
        { type: "Lessons", id: `${subject}-${topic}` },
      ],
    }),

    // Generate lesson using GET endpoint with parameters
    generateLesson: builder.query({
      query: ({
        subject,
        topic,
        include_wikipedia = true,
        use_knowledge_store = true,
      }) => {
        // Log the request for debugging
        console.log("Generating lesson with parameters:", {
          subject,
          topic,
          include_wikipedia,
          use_knowledge_store,
        });

        const params = new URLSearchParams({
          subject: subject,
          topic: topic,
          include_wikipedia: include_wikipedia.toString(),
          use_knowledge_store: use_knowledge_store.toString(),
        });

        const requestConfig = {
          url: `/generate_lesson?${params.toString()}`,
          method: "GET",
        };

        console.log("API Request config:", requestConfig);
        console.log("Full URL:", `${API_BASE_URL}${requestConfig.url}`);

        return requestConfig;
      },
      transformResponse: (response, meta, arg) => {
        console.log("API Raw Response:", response);
        console.log("API Meta:", meta);
        console.log("API Args:", arg);
        return response;
      },
      providesTags: (result, error, { subject, topic }) => [
        { type: "Lessons", id: `${subject}-${topic}` },
      ],
    }),

    // Create new lesson
    createLesson: builder.mutation({
      query: ({
        subject,
        topic,
        user_id,
        include_wikipedia = true,
        force_regenerate = false,
      }) => {
        const userId = user_id || "guest-user";

        // Log the request for debugging
        console.log("Creating lesson with data:", {
          subject,
          topic,
          user_id: userId,
          include_wikipedia,
          force_regenerate,
        });

        return {
          url: "/lessons",
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: {
            subject,
            topic,
            user_id: userId,
            include_wikipedia,
            force_regenerate,
          },
        };
      },
      invalidatesTags: (result, error, { subject, topic }) => [
        { type: "Lessons", id: `${subject}-${topic}` },
      ],
    }),
  }),
});

// Keep the original subjects API slice for backward compatibility
import { apiSlice } from "./apiSlice";

export const subjectsApiSlice = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    getSubjects: builder.query({
      query: () => "/subjects",
      transformResponse: (response) => {
        // You can transform the response data here if needed
        return response;
      },
      providesTags: ["Subjects"],
    }),
    getSubjectById: builder.query({
      query: (id) => `/subjects/${id}`,
      providesTags: (result, error, id) => [{ type: "Subjects", id }],
    }),
    addSubject: builder.mutation({
      query: (subject) => ({
        url: "/subjects",
        method: "POST",
        body: subject,
      }),
      invalidatesTags: ["Subjects"],
    }),
    updateSubject: builder.mutation({
      query: ({ id, ...subject }) => ({
        url: `/subjects/${id}`,
        method: "PUT",
        body: subject,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: "Subjects", id }],
    }),
    deleteSubject: builder.mutation({
      query: (id) => ({
        url: `/subjects/${id}`,
        method: "DELETE",
      }),
      invalidatesTags: ["Subjects"],
    }),
  }),
});

// Export hooks from lesson API slice
export const {
  useGetLessonDataQuery,
  useLazyGetLessonDataQuery,
  useGenerateLessonQuery,
  useLazyGenerateLessonQuery,
  useCreateLessonMutation,
} = lessonApiSlice;

// Export hooks from subjects API slice (for backward compatibility)
export const {
  useGetSubjectsQuery,
  useGetSubjectByIdQuery,
  useAddSubjectMutation,
  useUpdateSubjectMutation,
  useDeleteSubjectMutation,
} = subjectsApiSlice;
