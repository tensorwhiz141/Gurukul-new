/**
 * Memory Management API - RTK Query Integration
 * 
 * This file provides RTK Query endpoints for the Memory Management API,
 * including all CRUD operations, search functionality, and persona management.
 */

import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

// Configuration
const MEMORY_API_BASE_URL = process.env.REACT_APP_MEMORY_API_BASE_URL || 'http://localhost:8003';
const MEMORY_API_KEY = process.env.REACT_APP_MEMORY_API_KEY || 'memory_api_key_dev';
const API_TIMEOUT = parseInt(process.env.REACT_APP_MEMORY_API_TIMEOUT) || 30000;

// Base query configuration
const memoryBaseQuery = fetchBaseQuery({
  baseUrl: MEMORY_API_BASE_URL,
  timeout: API_TIMEOUT,
  prepareHeaders: (headers, { getState }) => {
    // Add authentication header
    headers.set('Authorization', `Bearer ${MEMORY_API_KEY}`);
    headers.set('Content-Type', 'application/json');
    
    // Add request ID for tracking
    headers.set('X-Request-ID', `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
    
    return headers;
  },
});

// Enhanced base query with error handling and retry logic
const memoryBaseQueryWithRetry = async (args, api, extraOptions) => {
  const maxRetries = parseInt(process.env.REACT_APP_MEMORY_API_RETRY_ATTEMPTS) || 3;
  const retryDelay = parseInt(process.env.REACT_APP_MEMORY_API_RETRY_DELAY) || 1000;
  
  let result = await memoryBaseQuery(args, api, extraOptions);
  
  // Retry logic for specific error codes
  let retryCount = 0;
  while (result.error && retryCount < maxRetries) {
    const { status } = result.error;
    
    // Retry on server errors (5xx) and rate limiting (429)
    if (status >= 500 || status === 429) {
      retryCount++;
      
      // Exponential backoff
      const delay = retryDelay * Math.pow(2, retryCount - 1);
      await new Promise(resolve => setTimeout(resolve, delay));
      
      result = await memoryBaseQuery(args, api, extraOptions);
    } else {
      break;
    }
  }
  
  return result;
};

// Create the Memory API slice
export const memoryApiSlice = createApi({
  reducerPath: 'memoryApi',
  baseQuery: memoryBaseQueryWithRetry,
  tagTypes: [
    'Memory',
    'Interaction', 
    'PersonaMemories',
    'PersonaSummary',
    'RecentInteractions',
    'SearchResults'
  ],
  endpoints: (builder) => ({
    
    // =========================================================================
    // Memory Storage Endpoints
    // =========================================================================
    
    /**
     * Store a new memory chunk
     */
    storeMemory: builder.mutation({
      query: (memoryData) => ({
        url: '/memory',
        method: 'POST',
        body: memoryData,
      }),
      invalidatesTags: (result, error, arg) => [
        { type: 'PersonaMemories', id: arg.persona_id },
        { type: 'PersonaSummary', id: arg.persona_id },
        'Memory'
      ],
      transformResponse: (response) => response.data,
      transformErrorResponse: (response) => ({
        status: response.status,
        message: response.data?.message || 'Failed to store memory',
        details: response.data?.details || []
      }),
    }),
    
    /**
     * Store a user-agent interaction
     */
    storeInteraction: builder.mutation({
      query: (interactionData) => ({
        url: '/memory/interaction',
        method: 'POST',
        body: interactionData,
      }),
      invalidatesTags: (result, error, arg) => [
        { type: 'PersonaMemories', id: arg.persona_id },
        { type: 'PersonaSummary', id: arg.persona_id },
        { type: 'RecentInteractions', id: arg.user_id },
        'Interaction'
      ],
      transformResponse: (response) => response.data,
    }),
    
    // =========================================================================
    // Memory Retrieval Endpoints
    // =========================================================================
    
    /**
     * Get memories for a specific persona
     */
    getPersonaMemories: builder.query({
      query: ({ 
        personaId, 
        userId, 
        limit = 20, 
        offset = 0, 
        contentTypes = [], 
        minImportance 
      }) => {
        const params = new URLSearchParams({
          persona: personaId,
          limit: limit.toString(),
          offset: offset.toString(),
        });
        
        if (userId) params.append('user_id', userId);
        if (minImportance) params.append('min_importance', minImportance.toString());
        
        // Add multiple content types
        contentTypes.forEach(type => params.append('content_type', type));
        
        return `/memory?${params.toString()}`;
      },
      providesTags: (result, error, arg) => [
        { type: 'PersonaMemories', id: arg.personaId },
        'Memory'
      ],
      transformResponse: (response) => ({
        memories: response.memories || [],
        pagination: {
          totalCount: response.total_count,
          page: response.page,
          pageSize: response.page_size,
          hasNext: response.has_next,
          hasPrevious: response.has_previous
        }
      }),
    }),
    
    /**
     * Get recent interactions for chain-of-thought processing
     */
    getRecentInteractions: builder.query({
      query: ({ limit = 5, userId, personaId }) => {
        const params = new URLSearchParams({
          limit: limit.toString(),
          recent_interactions: 'true'
        });
        
        if (userId) params.append('user_id', userId);
        if (personaId) params.append('persona', personaId);
        
        return `/memory?${params.toString()}`;
      },
      providesTags: (result, error, arg) => [
        { type: 'RecentInteractions', id: arg.userId || 'all' },
        'Interaction'
      ],
      transformResponse: (response) => ({
        interactions: response.interactions || [],
        totalCount: response.total_count
      }),
    }),
    
    /**
     * Get a specific memory by ID
     */
    getMemoryById: builder.query({
      query: (memoryId) => `/memory/${memoryId}`,
      providesTags: (result, error, arg) => [{ type: 'Memory', id: arg }],
    }),
    
    /**
     * Get persona memory summary
     */
    getPersonaSummary: builder.query({
      query: ({ personaId, userId }) => {
        const params = userId ? `?user_id=${userId}` : '';
        return `/memory/persona/${personaId}/summary${params}`;
      },
      providesTags: (result, error, arg) => [
        { type: 'PersonaSummary', id: arg.personaId }
      ],
    }),
    
    // =========================================================================
    // Memory Search Endpoints
    // =========================================================================
    
    /**
     * Search memories by text query
     */
    searchMemories: builder.query({
      query: ({ 
        query, 
        personaId, 
        userId, 
        contentTypes = [], 
        limit = 10 
      }) => {
        const params = new URLSearchParams({
          query,
          limit: limit.toString()
        });
        
        if (personaId) params.append('persona_id', personaId);
        if (userId) params.append('user_id', userId);
        
        // Add multiple content types
        contentTypes.forEach(type => params.append('content_type', type));
        
        return `/memory/search?${params.toString()}`;
      },
      providesTags: ['SearchResults'],
      transformResponse: (response) => ({
        results: response.results || [],
        query: response.query,
        totalResults: response.total_results,
        searchTime: response.search_time,
        suggestions: response.suggestions || []
      }),
    }),
    
    // =========================================================================
    // Memory Management Endpoints
    // =========================================================================
    
    /**
     * Update an existing memory
     */
    updateMemory: builder.mutation({
      query: ({ memoryId, updateData }) => ({
        url: `/memory/${memoryId}`,
        method: 'PUT',
        body: updateData,
      }),
      invalidatesTags: (result, error, arg) => [
        { type: 'Memory', id: arg.memoryId },
        'PersonaMemories',
        'PersonaSummary'
      ],
    }),
    
    /**
     * Delete a memory (soft delete by default)
     */
    deleteMemory: builder.mutation({
      query: ({ memoryId, hardDelete = false }) => ({
        url: `/memory/${memoryId}`,
        method: 'DELETE',
        params: hardDelete ? { hard_delete: 'true' } : {},
      }),
      invalidatesTags: (result, error, arg) => [
        { type: 'Memory', id: arg.memoryId },
        'PersonaMemories',
        'PersonaSummary'
      ],
    }),
    
    // =========================================================================
    // Health Check Endpoint
    // =========================================================================
    
    /**
     * Check API health
     */
    checkHealth: builder.query({
      query: () => '/memory/health',
      transformResponse: (response) => ({
        status: response.status,
        timestamp: response.timestamp,
        database: response.database
      }),
    }),
  }),
});

// Export hooks for use in components
export const {
  // Memory Storage
  useStoreMemoryMutation,
  useStoreInteractionMutation,
  
  // Memory Retrieval
  useGetPersonaMemoriesQuery,
  useGetRecentInteractionsQuery,
  useGetMemoryByIdQuery,
  useGetPersonaSummaryQuery,
  
  // Memory Search
  useSearchMemoriesQuery,
  useLazySearchMemoriesQuery,
  
  // Memory Management
  useUpdateMemoryMutation,
  useDeleteMemoryMutation,
  
  // Health Check
  useCheckHealthQuery,
  
  // Lazy queries for manual triggering
  useLazyGetPersonaMemoriesQuery,
  useLazyGetRecentInteractionsQuery,
  useLazyGetPersonaSummaryQuery,
} = memoryApiSlice;

// Export the reducer
export default memoryApiSlice.reducer;

// =========================================================================
// Utility Functions
// =========================================================================

/**
 * Helper function to build memory data object
 */
export const buildMemoryData = ({
  userId,
  personaId,
  content,
  contentType = 'text',
  tags = [],
  importance = 5,
  topic = null,
  contextType = null,
  source = 'user_input',
  confidence = null
}) => ({
  user_id: userId,
  persona_id: personaId,
  content,
  content_type: contentType,
  metadata: {
    tags,
    importance,
    topic,
    context_type: contextType,
    source,
    confidence
  }
});

/**
 * Helper function to build interaction data object
 */
export const buildInteractionData = ({
  userId,
  personaId,
  userMessage,
  agentResponse,
  sessionId = null,
  conversationTurn = null,
  domain = null,
  intent = null,
  previousContext = null,
  responseTime = null,
  confidence = null,
  modelUsed = null,
  tags = []
}) => ({
  user_id: userId,
  persona_id: personaId,
  user_message: userMessage,
  agent_response: agentResponse,
  context: {
    session_id: sessionId,
    conversation_turn: conversationTurn,
    domain,
    intent,
    previous_context: previousContext
  },
  metadata: {
    response_time: responseTime,
    confidence,
    model_used: modelUsed,
    tags
  }
});

/**
 * Content type constants
 */
export const CONTENT_TYPES = {
  TEXT: 'text',
  INTERACTION: 'interaction',
  CONTEXT: 'context',
  REFLECTION: 'reflection',
  PREFERENCE: 'preference',
  FACT: 'fact'
};

/**
 * Importance level constants
 */
export const IMPORTANCE_LEVELS = {
  VERY_LOW: 1,
  LOW: 2,
  BELOW_AVERAGE: 3,
  AVERAGE: 4,
  ABOVE_AVERAGE: 5,
  MODERATE: 6,
  HIGH: 7,
  VERY_HIGH: 8,
  CRITICAL: 9,
  ESSENTIAL: 10
};
