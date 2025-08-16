import { apiSlice } from "./apiSlice";

export const learningApiSlice = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    // Send learning data and get task ID
    sendLearningData: builder.mutation({
      query: ({ user_id, query, pdf_id }) => ({
        url: "/user/learning",
        method: "POST",
        body: { user_id, query, ...(pdf_id && { pdf_id }) },
        params: { wait: false }, // Use async processing
      }),
      invalidatesTags: ["LearningTask"],
    }),

    // Check learning task status and get response
    getLearningTaskStatus: builder.query({
      query: (taskId) => `/user/learning/${taskId}`,
      providesTags: (result, error, taskId) => [
        { type: "LearningTask", id: taskId },
      ],
    }),

    // Upload PDF for chat
    uploadPdfForChat: builder.mutation({
      query: ({ user_id, pdf_file }) => {
        const formData = new FormData();
        formData.append("user_id", user_id);
        formData.append("pdf_file", pdf_file);
        
        return {
          url: "/pdf/chat",
          method: "POST",
          body: formData,
        };
      },
      invalidatesTags: ["PdfChat"],
    }),

    // Notify PDF removed
    notifyPdfRemoved: builder.mutation({
      query: ({ user_id, pdf_id }) => ({
        url: "/pdf/removed",
        method: "POST",
        body: { user_id, pdf_id },
      }),
      invalidatesTags: ["PdfChat"],
    }),
  }),
});

export const {
  useSendLearningDataMutation,
  useGetLearningTaskStatusQuery,
  useLazyGetLearningTaskStatusQuery,
  useUploadPdfForChatMutation,
  useNotifyPdfRemovedMutation,
} = learningApiSlice;
