import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { API_BASE_URL } from "../config";
import ttsService from "../services/ttsService";
import { 
  createBHIVRequest, 
  createAgentDirectives,
  isBHIVCoreResponse,
  ClassificationType,
  UrgencyLevel,
  TriggerModule,
  OutputFormat,
  InputMode,
  UserPersona
} from "../types/bhivCoreSchema";

// Create BHIV Core API slice for the new schema
export const bhivCoreApiSlice = createApi({
  reducerPath: "bhivCoreApi",
  baseQuery: fetchBaseQuery({
    baseUrl: `${API_BASE_URL}:8010`, // BHIV Core API port
    timeout: 120000, // 2 minute timeout for complex processing
    prepareHeaders: (headers, { getState }) => {
      headers.set('Content-Type', 'application/json');
      
      // Add auth token if available
      const auth = getState()?.auth;
      if (auth?.token) {
        headers.set('Authorization', `Bearer ${auth.token}`);
      }
      
      return headers;
    },
  }),
  tagTypes: [
    "BHIVSession", 
    "BHIVResponse", 
    "BHIVStats",
    "BHIVHealth",
    "BHIVClassification"
  ],
  endpoints: (builder) => ({
    
    // Main BHIV Core processing endpoint
    processBHIVRequest: builder.mutation({
      query: (bhivRequest) => {
        console.log("🧠 Processing BHIV Core request:", bhivRequest);
        
        return {
          url: "/api/v1/bhiv-core/process",
          method: "POST",
          body: bhivRequest,
        };
      },
      async onQueryStarted(arg, { queryFulfilled }) {
        try {
          const { data } = await queryFulfilled;
          
          // Validate response
          if (!isBHIVCoreResponse(data)) {
            console.warn("⚠️ Invalid BHIV Core response format:", data);
            return;
          }
          
          // Trigger automatic TTS if voice is enabled and response has audio URL
          if (data.input?.voice_enabled && data.output?.voice_response_url) {
            console.log("🔊 BHIV TTS: Playing voice response from URL");
            
            try {
              // Play audio from URL
              const audio = new Audio(data.output.voice_response_url);
              audio.play().catch(error => {
                console.warn("🔊 BHIV TTS: Failed to play audio from URL:", error);
              });
            } catch (error) {
              console.warn("🔊 BHIV TTS: Audio playback error:", error);
            }
          } 
          // Fallback to TTS service for text
          else if (data.input?.voice_enabled && data.output?.response_text) {
            console.log("🔊 BHIV TTS: Generating TTS for response text");
            
            const textToSpeak = data.output.response_text
              .replace(/\n\n/g, '. ')
              .replace(/\n/g, ' ')
              .replace(/\*\*(.*?)\*\*/g, '$1')
              .replace(/\*(.*?)\*/g, '$1');
            
            setTimeout(() => {
              ttsService.autoPlayAI(textToSpeak, {
                delay: 500,
                volume: 0.8
              }).catch(error => {
                console.warn("🔊 BHIV TTS: Fallback TTS failed:", error.message);
              });
            }, 300);
          }
          
        } catch (error) {
          console.warn("🔊 BHIV TTS: Failed to process response for TTS:", error);
        }
      },
      invalidatesTags: (result, error, { user_info, session_id }) => [
        { type: "BHIVSession", id: session_id },
        { type: "BHIVResponse", id: user_info?.user_id },
        "BHIVStats"
      ],
    }),

    // Quick query endpoint for simple requests
    quickQuery: builder.mutation({
      query: ({ userId, query, voiceEnabled = false, lang = "en-IN" }) => {
        console.log("⚡ BHIV Quick Query:", { userId, query, voiceEnabled, lang });
        
        return {
          url: "/api/v1/bhiv-core/quick-query",
          method: "POST",
          body: {
            user_id: userId,
            query: query,
            voice_enabled: voiceEnabled,
            lang: lang
          },
        };
      },
      async onQueryStarted(arg, { queryFulfilled }) {
        try {
          const { data } = await queryFulfilled;
          
          // Handle TTS for quick queries
          if (arg.voiceEnabled) {
            if (data.voice_response_url) {
              const audio = new Audio(data.voice_response_url);
              audio.play().catch(console.warn);
            } else if (data.response_text) {
              const textToSpeak = data.response_text
                .replace(/\n\n/g, '. ')
                .replace(/\n/g, ' ');
              
              setTimeout(() => {
                ttsService.autoPlayAI(textToSpeak, { delay: 300, volume: 0.8 })
                  .catch(console.warn);
              }, 200);
            }
          }
          
        } catch (error) {
          console.warn("⚡ Quick Query TTS error:", error);
        }
      },
      invalidatesTags: (result, error, { userId }) => [
        { type: "BHIVResponse", id: userId },
        "BHIVStats"
      ],
    }),

    // Classify query without full processing
    classifyQuery: builder.mutation({
      query: ({ userId, query, lang = "en-IN" }) => {
        console.log("🔍 BHIV Classify Query:", { userId, query, lang });
        
        return {
          url: "/api/v1/bhiv-core/classify",
          method: "POST",
          body: {
            user_id: userId,
            query: query,
            lang: lang
          },
        };
      },
      invalidatesTags: ["BHIVClassification"],
    }),

    // Get session information
    getSessionInfo: builder.query({
      query: (sessionId) => {
        console.log("📋 Getting BHIV session info:", sessionId);
        return `/api/v1/bhiv-core/session/${sessionId}`;
      },
      providesTags: (result, error, sessionId) => [
        { type: "BHIVSession", id: sessionId }
      ],
    }),

    // Get BHIV Core statistics
    getBHIVStats: builder.query({
      query: () => {
        console.log("📊 Fetching BHIV Core statistics");
        return "/api/v1/bhiv-core/stats";
      },
      providesTags: ["BHIVStats"],
    }),

    // Health check for BHIV Core
    checkBHIVHealth: builder.query({
      query: () => {
        console.log("❤️ Checking BHIV Core health");
        return "/health";
      },
      providesTags: ["BHIVHealth"],
    }),

  }),
});

// Export hooks for use in components
export const {
  useProcessBHIVRequestMutation,
  useQuickQueryMutation,
  useClassifyQueryMutation,
  useGetSessionInfoQuery,
  useLazyGetSessionInfoQuery,
  useGetBHIVStatsQuery,
  useLazyGetBHIVStatsQuery,
  useCheckBHIVHealthQuery,
  useLazyCheckBHIVHealthQuery,
} = bhivCoreApiSlice;

// Helper functions for creating BHIV requests

export const createQuickBHIVRequest = (
  userId,
  query,
  options = {}
) => {
  const {
    voiceEnabled = false,
    mode = InputMode.CHAT,
    persona = UserPersona.STUDENT,
    lang = "en-IN",
    authToken = "default_token",
    course = null,
    previousMessageId = null
  } = options;

  return createBHIVRequest(userId, query, {
    authToken,
    voiceEnabled,
    mode,
    persona,
    lang
  });
};

export const createEducationalBHIVRequest = (
  userId,
  query,
  options = {}
) => {
  const baseRequest = createQuickBHIVRequest(userId, query, {
    ...options,
    persona: UserPersona.STUDENT
  });

  // Add educational-specific directives
  baseRequest.agent_directives = createAgentDirectives(
    ClassificationType.LEARNING_QUERY,
    "EduMentor",
    {
      urgency: UrgencyLevel.NORMAL,
      triggerModule: TriggerModule.KNOWLEDGEBASE,
      outputFormat: OutputFormat.RICH_TEXT_WITH_TTS
    }
  );

  return baseRequest;
};

export const createWellnessBHIVRequest = (
  userId,
  query,
  options = {}
) => {
  const baseRequest = createQuickBHIVRequest(userId, query, options);

  baseRequest.agent_directives = createAgentDirectives(
    ClassificationType.WELLNESS_QUERY,
    "WellnessBot",
    {
      urgency: options.urgency || UrgencyLevel.NORMAL,
      triggerModule: TriggerModule.WELLNESS_BOT,
      outputFormat: OutputFormat.RICH_TEXT_WITH_TTS
    }
  );

  return baseRequest;
};

export const createSpiritualBHIVRequest = (
  userId,
  query,
  options = {}
) => {
  const baseRequest = createQuickBHIVRequest(userId, query, options);

  baseRequest.agent_directives = createAgentDirectives(
    ClassificationType.SPIRITUAL_QUERY,
    "GuruAgent",
    {
      urgency: UrgencyLevel.NORMAL,
      triggerModule: TriggerModule.SPIRITUAL_BOT,
      outputFormat: OutputFormat.RICH_TEXT_WITH_TTS
    }
  );

  return baseRequest;
};

// Helper function to check if BHIV Core is available
export const checkBHIVAvailability = async (dispatch) => {
  try {
    const result = await dispatch(
      bhivCoreApiSlice.endpoints.checkBHIVHealth.initiate()
    );
    
    if (result.data) {
      const isAvailable = result.data.status === 'healthy' &&
                         result.data.components?.orchestration_engine === 'ready';
      
      console.log("🧠 BHIV Core availability check:", isAvailable);
      return isAvailable;
    }
    
    return false;
  } catch (error) {
    console.warn("⚠️ BHIV Core availability check failed:", error);
    return false;
  }
};

// Helper function to format BHIV response for display
export const formatBHIVResponse = (bhivResponse) => {
  if (!isBHIVCoreResponse(bhivResponse)) {
    console.warn("Invalid BHIV response format");
    return null;
  }

  return {
    // Basic response data
    sessionId: bhivResponse.session_id,
    timestamp: bhivResponse.timestamp,
    responseText: bhivResponse.output.response_text,
    confidenceScore: bhivResponse.output.confidence_score,
    
    // Agent information
    agentUsed: bhivResponse.agent_directives.agent_name,
    classification: bhivResponse.agent_directives.classification,
    urgency: bhivResponse.agent_directives.urgency,
    
    // Additional data
    sources: bhivResponse.output.sources || [],
    followupSuggestions: bhivResponse.output.followup_suggestions || [],
    voiceResponseUrl: bhivResponse.output.voice_response_url,
    
    // Metadata
    processingTimeMs: bhivResponse.logs.processing_time_ms,
    rewardScore: bhivResponse.logs.reward_score,
    errorFlag: bhivResponse.logs.error_flag,
    
    // User context
    userId: bhivResponse.user_info.user_id,
    userLang: bhivResponse.user_info.lang,
    voiceEnabled: bhivResponse.input.voice_enabled,
    
    // Raw response for advanced use
    raw: bhivResponse
  };
};

export default bhivCoreApiSlice;
