import { apiSlice } from './apiSlice';

export const testsApiSlice = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    getTests: builder.query({
      query: () => '/test_dummy',
      transformResponse: (response) => {
        // You can transform the response data here if needed
        return response;
      },
      providesTags: ['Tests'],
    }),
    getTestById: builder.query({
      query: (id) => `/test_dummy/${id}`,
      providesTags: (result, error, id) => [{ type: 'Tests', id }],
    }),
    addTest: builder.mutation({
      query: (test) => ({
        url: '/test_dummy',
        method: 'POST',
        body: test,
      }),
      invalidatesTags: ['Tests'],
    }),
    updateTest: builder.mutation({
      query: ({ id, ...test }) => ({
        url: `/test_dummy/${id}`,
        method: 'PUT',
        body: test,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'Tests', id }],
    }),
    deleteTest: builder.mutation({
      query: (id) => ({
        url: `/test_dummy/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Tests'],
    }),
  }),
});

export const {
  useGetTestsQuery,
  useGetTestByIdQuery,
  useAddTestMutation,
  useUpdateTestMutation,
  useDeleteTestMutation,
} = testsApiSlice;
