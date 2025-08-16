import { apiSlice } from './apiSlice';

export const lecturesApiSlice = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    getLectures: builder.query({
      query: () => '/lecture_dummy',
      transformResponse: (response) => {
        // You can transform the response data here if needed
        return response;
      },
      providesTags: ['Lectures'],
    }),
    getLectureById: builder.query({
      query: (id) => `/lecture_dummy/${id}`,
      providesTags: (result, error, id) => [{ type: 'Lectures', id }],
    }),
    addLecture: builder.mutation({
      query: (lecture) => ({
        url: '/lecture_dummy',
        method: 'POST',
        body: lecture,
      }),
      invalidatesTags: ['Lectures'],
    }),
    updateLecture: builder.mutation({
      query: ({ id, ...lecture }) => ({
        url: `/lecture_dummy/${id}`,
        method: 'PUT',
        body: lecture,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'Lectures', id }],
    }),
    deleteLecture: builder.mutation({
      query: (id) => ({
        url: `/lecture_dummy/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Lectures'],
    }),
  }),
});

export const {
  useGetLecturesQuery,
  useGetLectureByIdQuery,
  useAddLectureMutation,
  useUpdateLectureMutation,
  useDeleteLectureMutation,
} = lecturesApiSlice;
