import { apiSlice } from "./apiSlice";

export const agentApiSlice = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    getAgentOutputs: builder.query({
      query: () => "/get_agent_output",
      providesTags: ["AgentOutput"],
      // Set up polling for real-time updates
      keepUnusedDataFor: 30, // Keep data for 30 seconds
    }),

    getAgentLogs: builder.query({
      query: () => "/agent_logs",
      providesTags: ["AgentLogs"],
      // Set up polling for real-time updates
      keepUnusedDataFor: 30, // Keep data for 30 seconds
    }),

    sendAgentMessage: builder.mutation({
      query: ({ message, agentId, userId }) => ({
        url: "/agent_message",
        method: "POST",
        body: {
          message,
          agent_id: agentId,
          user_id: userId || "guest-user",
          timestamp: new Date().toISOString(),
        },
      }),
      invalidatesTags: ["AgentOutput", "AgentLogs"],
    }),

    startAgentSimulation: builder.mutation({
      query: ({ agentId, userId }) => ({
        url: "/start_agent_simulation",
        method: "POST",
        body: {
          agent_id: agentId,
          user_id: userId || "guest-user",
          timestamp: new Date().toISOString(),
        },
      }),
      invalidatesTags: ["AgentOutput", "AgentLogs"],
    }),

    stopAgentSimulation: builder.mutation({
      query: ({ agentId, userId }) => ({
        url: "/stop_agent_simulation",
        method: "POST",
        body: {
          agent_id: agentId,
          user_id: userId || "guest-user",
          timestamp: new Date().toISOString(),
        },
      }),
      invalidatesTags: ["AgentOutput", "AgentLogs"],
    }),

    resetAgentSimulation: builder.mutation({
      query: ({ userId }) => ({
        url: "/reset_agent_simulation",
        method: "POST",
        body: {
          user_id: userId || "guest-user",
          timestamp: new Date().toISOString(),
        },
      }),
      invalidatesTags: ["AgentOutput", "AgentLogs"],
    }),
  }),
});

export const {
  useGetAgentOutputsQuery,
  useGetAgentLogsQuery,
  useSendAgentMessageMutation,
  useStartAgentSimulationMutation,
  useStopAgentSimulationMutation,
  useResetAgentSimulationMutation,
} = agentApiSlice;
