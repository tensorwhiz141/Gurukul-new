/**
 * Memory Management - React Hooks and Component Examples
 * 
 * This file provides custom React hooks and component examples for integrating
 * the Memory Management API into your React application.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  useStoreMemoryMutation,
  useStoreInteractionMutation,
  useGetPersonaMemoriesQuery,
  useGetRecentInteractionsQuery,
  useGetPersonaSummaryQuery,
  useSearchMemoriesQuery,
  buildMemoryData,
  buildInteractionData,
  CONTENT_TYPES,
  IMPORTANCE_LEVELS
} from './memoryApiSlice';

// =========================================================================
// Custom Hooks for Memory Management
// =========================================================================

/**
 * Hook for managing memory storage with automatic error handling
 */
export const useMemoryStorage = () => {
  const [storeMemory, { isLoading: isStoringMemory, error: memoryError }] = useStoreMemoryMutation();
  const [storeInteraction, { isLoading: isStoringInteraction, error: interactionError }] = useStoreInteractionMutation();

  const storeUserMemory = useCallback(async ({
    userId,
    personaId,
    content,
    contentType = CONTENT_TYPES.TEXT,
    tags = [],
    importance = IMPORTANCE_LEVELS.AVERAGE,
    topic = null,
    source = 'user_input'
  }) => {
    try {
      const memoryData = buildMemoryData({
        userId,
        personaId,
        content,
        contentType,
        tags,
        importance,
        topic,
        source
      });

      const result = await storeMemory(memoryData).unwrap();
      return { success: true, memoryId: result.memory_id };
    } catch (error) {
      console.error('Failed to store memory:', error);
      return { success: false, error: error.message };
    }
  }, [storeMemory]);

  const storeUserInteraction = useCallback(async ({
    userId,
    personaId,
    userMessage,
    agentResponse,
    sessionId,
    domain = 'general',
    intent = null,
    responseTime = null,
    confidence = null,
    modelUsed = null
  }) => {
    try {
      const interactionData = buildInteractionData({
        userId,
        personaId,
        userMessage,
        agentResponse,
        sessionId,
        domain,
        intent,
        responseTime,
        confidence,
        modelUsed
      });

      const result = await storeInteraction(interactionData).unwrap();
      return { success: true, interactionId: result.interaction_id };
    } catch (error) {
      console.error('Failed to store interaction:', error);
      return { success: false, error: error.message };
    }
  }, [storeInteraction]);

  return {
    storeUserMemory,
    storeUserInteraction,
    isStoringMemory,
    isStoringInteraction,
    memoryError,
    interactionError
  };
};

/**
 * Hook for retrieving and managing persona memories
 */
export const usePersonaMemories = (personaId, userId, options = {}) => {
  const {
    limit = 20,
    contentTypes = [],
    minImportance = null,
    autoRefresh = false,
    refreshInterval = 30000
  } = options;

  const {
    data: memoriesData,
    isLoading,
    error,
    refetch
  } = useGetPersonaMemoriesQuery({
    personaId,
    userId,
    limit,
    contentTypes,
    minImportance
  }, {
    skip: !personaId,
    pollingInterval: autoRefresh ? refreshInterval : 0
  });

  const memories = useMemo(() => memoriesData?.memories || [], [memoriesData]);
  const pagination = useMemo(() => memoriesData?.pagination || {}, [memoriesData]);

  // Group memories by content type
  const memoriesByType = useMemo(() => {
    return memories.reduce((acc, memory) => {
      const type = memory.content_type;
      if (!acc[type]) acc[type] = [];
      acc[type].push(memory);
      return acc;
    }, {});
  }, [memories]);

  // Get high-importance memories
  const importantMemories = useMemo(() => {
    return memories.filter(memory => memory.metadata.importance >= 7);
  }, [memories]);

  return {
    memories,
    memoriesByType,
    importantMemories,
    pagination,
    isLoading,
    error,
    refetch
  };
};

/**
 * Hook for managing conversation context with chain-of-thought
 */
export const useConversationContext = (userId, personaId, options = {}) => {
  const { limit = 5, autoUpdate = true } = options;

  const {
    data: interactionsData,
    isLoading,
    error,
    refetch
  } = useGetRecentInteractionsQuery({
    limit,
    userId,
    personaId
  }, {
    skip: !userId,
    pollingInterval: autoUpdate ? 10000 : 0
  });

  const interactions = useMemo(() => interactionsData?.interactions || [], [interactionsData]);

  // Build conversation context for AI
  const conversationContext = useMemo(() => {
    return interactions.map(interaction => ({
      role: 'user',
      content: interaction.user_message,
      timestamp: interaction.timestamp
    })).concat(
      interactions.map(interaction => ({
        role: 'assistant',
        content: interaction.agent_response,
        timestamp: interaction.timestamp
      }))
    ).sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  }, [interactions]);

  // Get context summary
  const contextSummary = useMemo(() => {
    if (interactions.length === 0) return null;
    
    const lastInteraction = interactions[interactions.length - 1];
    return {
      totalInteractions: interactions.length,
      lastUserMessage: lastInteraction?.user_message,
      lastAgentResponse: lastInteraction?.agent_response,
      lastTimestamp: lastInteraction?.timestamp
    };
  }, [interactions]);

  return {
    interactions,
    conversationContext,
    contextSummary,
    isLoading,
    error,
    refetch
  };
};

/**
 * Hook for persona memory summary and statistics
 */
export const usePersonaSummary = (personaId, userId) => {
  const {
    data: summary,
    isLoading,
    error,
    refetch
  } = useGetPersonaSummaryQuery({
    personaId,
    userId
  }, {
    skip: !personaId
  });

  const stats = useMemo(() => {
    if (!summary) return null;

    return {
      totalMemories: summary.total_memories,
      categories: summary.memory_categories,
      recentTopics: summary.recent_topics,
      lastInteraction: summary.last_interaction,
      importanceDistribution: summary.importance_distribution
    };
  }, [summary]);

  return {
    summary,
    stats,
    isLoading,
    error,
    refetch
  };
};

/**
 * Hook for memory search with debouncing
 */
export const useMemorySearch = (options = {}) => {
  const { debounceDelay = 500 } = options;
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, debounceDelay);

    return () => clearTimeout(timer);
  }, [searchQuery, debounceDelay]);

  const {
    data: searchResults,
    isLoading,
    error
  } = useSearchMemoriesQuery({
    query: debouncedQuery,
    ...options
  }, {
    skip: !debouncedQuery || debouncedQuery.length < 2
  });

  const search = useCallback((query) => {
    setSearchQuery(query);
  }, []);

  const clearSearch = useCallback(() => {
    setSearchQuery('');
    setDebouncedQuery('');
  }, []);

  return {
    search,
    clearSearch,
    searchQuery,
    searchResults: searchResults?.results || [],
    totalResults: searchResults?.totalResults || 0,
    searchTime: searchResults?.searchTime || 0,
    isLoading,
    error
  };
};

// =========================================================================
// React Component Examples
// =========================================================================

/**
 * Example: Chat Component with Memory Integration
 */
export const ChatWithMemory = ({ userId, personaId, sessionId }) => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  
  const { storeUserInteraction } = useMemoryStorage();
  const { conversationContext } = useConversationContext(userId, personaId);
  const { memories } = usePersonaMemories(personaId, userId, { limit: 10 });

  const handleSendMessage = async () => {
    if (!message.trim()) return;

    // Add user message to chat
    const userMessage = { role: 'user', content: message, timestamp: new Date() };
    setChatHistory(prev => [...prev, userMessage]);

    // Simulate AI response (replace with your AI logic)
    const agentResponse = await generateAIResponse(message, conversationContext, memories);
    
    // Add agent response to chat
    const agentMessage = { role: 'assistant', content: agentResponse, timestamp: new Date() };
    setChatHistory(prev => [...prev, agentMessage]);

    // Store interaction in memory
    await storeUserInteraction({
      userId,
      personaId,
      userMessage: message,
      agentResponse,
      sessionId,
      domain: 'finance', // or detect automatically
      responseTime: 1.5 // measure actual response time
    });

    setMessage('');
  };

  return (
    <div className="chat-container">
      <div className="chat-history">
        {chatHistory.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'You' : 'Assistant'}:</strong>
            <p>{msg.content}</p>
          </div>
        ))}
      </div>
      
      <div className="chat-input">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Type your message..."
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
};

/**
 * Example: Memory Dashboard Component
 */
export const MemoryDashboard = ({ userId, personaId }) => {
  const { memories, memoriesByType, isLoading } = usePersonaMemories(personaId, userId);
  const { stats } = usePersonaSummary(personaId, userId);
  const { search, searchResults, searchQuery } = useMemorySearch({ personaId, userId });

  if (isLoading) return <div>Loading memories...</div>;

  return (
    <div className="memory-dashboard">
      <h2>Memory Dashboard - {personaId}</h2>
      
      {/* Memory Statistics */}
      <div className="memory-stats">
        <h3>Statistics</h3>
        {stats && (
          <div className="stats-grid">
            <div>Total Memories: {stats.totalMemories}</div>
            <div>Recent Topics: {stats.recentTopics?.join(', ')}</div>
            <div>Categories: {Object.entries(stats.categories).map(([type, count]) => 
              `${type}: ${count}`).join(', ')}</div>
          </div>
        )}
      </div>

      {/* Memory Search */}
      <div className="memory-search">
        <h3>Search Memories</h3>
        <input
          type="text"
          placeholder="Search memories..."
          onChange={(e) => search(e.target.value)}
        />
        {searchQuery && (
          <div className="search-results">
            <h4>Search Results ({searchResults.length})</h4>
            {searchResults.map(memory => (
              <div key={memory.memory_id} className="memory-item">
                <strong>{memory.content_type}:</strong> {memory.content}
                <div className="memory-meta">
                  Tags: {memory.metadata.tags.join(', ')} | 
                  Importance: {memory.metadata.importance}/10
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Memories by Type */}
      <div className="memories-by-type">
        <h3>Memories by Type</h3>
        {Object.entries(memoriesByType).map(([type, typeMemories]) => (
          <div key={type} className="memory-type-section">
            <h4>{type.charAt(0).toUpperCase() + type.slice(1)} ({typeMemories.length})</h4>
            {typeMemories.slice(0, 3).map(memory => (
              <div key={memory.memory_id} className="memory-item">
                {memory.content.substring(0, 100)}...
                <div className="memory-meta">
                  Importance: {memory.metadata.importance}/10 | 
                  {new Date(memory.timestamp).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Example: User Preference Capture Component
 */
export const PreferenceCapture = ({ userId, personaId }) => {
  const [preference, setPreference] = useState('');
  const [importance, setImportance] = useState(IMPORTANCE_LEVELS.AVERAGE);
  const [tags, setTags] = useState('');
  
  const { storeUserMemory, isStoringMemory } = useMemoryStorage();

  const handleSavePreference = async () => {
    if (!preference.trim()) return;

    const result = await storeUserMemory({
      userId,
      personaId,
      content: preference,
      contentType: CONTENT_TYPES.PREFERENCE,
      tags: tags.split(',').map(tag => tag.trim()).filter(Boolean),
      importance,
      topic: 'user_preference',
      source: 'preference_form'
    });

    if (result.success) {
      setPreference('');
      setTags('');
      setImportance(IMPORTANCE_LEVELS.AVERAGE);
      alert('Preference saved successfully!');
    } else {
      alert('Failed to save preference: ' + result.error);
    }
  };

  return (
    <div className="preference-capture">
      <h3>Save Your Preference</h3>
      
      <textarea
        value={preference}
        onChange={(e) => setPreference(e.target.value)}
        placeholder="Describe your preference..."
        rows={4}
        cols={50}
      />
      
      <div className="preference-meta">
        <label>
          Importance (1-10):
          <select value={importance} onChange={(e) => setImportance(parseInt(e.target.value))}>
            {Object.entries(IMPORTANCE_LEVELS).map(([name, value]) => (
              <option key={value} value={value}>{name.replace('_', ' ')} ({value})</option>
            ))}
          </select>
        </label>
        
        <label>
          Tags (comma-separated):
          <input
            type="text"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="investment, conservative, long-term"
          />
        </label>
      </div>
      
      <button 
        onClick={handleSavePreference} 
        disabled={isStoringMemory || !preference.trim()}
      >
        {isStoringMemory ? 'Saving...' : 'Save Preference'}
      </button>
    </div>
  );
};

// =========================================================================
// Utility Functions
// =========================================================================

/**
 * Mock AI response generator (replace with your actual AI logic)
 */
const generateAIResponse = async (userMessage, context, memories) => {
  // This is a mock function - replace with your actual AI integration
  const relevantMemories = memories.filter(memory => 
    memory.content.toLowerCase().includes(userMessage.toLowerCase().split(' ')[0])
  );
  
  if (relevantMemories.length > 0) {
    return `Based on what I know about your preferences: ${relevantMemories[0].content}. Here's my response to "${userMessage}"...`;
  }
  
  return `I understand you're asking about "${userMessage}". Let me help you with that...`;
};

/**
 * Extract tags from text content
 */
export const extractTags = (content) => {
  // Simple keyword extraction - enhance with NLP if needed
  const words = content.toLowerCase().match(/\b\w{4,}\b/g) || [];
  const commonWords = new Set(['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'said', 'each', 'which', 'their']);
  return [...new Set(words.filter(word => !commonWords.has(word)))].slice(0, 5);
};

/**
 * Calculate importance based on content
 */
export const calculateImportance = (content) => {
  // Simple heuristic - enhance with ML if needed
  const importantKeywords = ['important', 'critical', 'essential', 'must', 'never', 'always'];
  const hasImportantKeywords = importantKeywords.some(keyword => 
    content.toLowerCase().includes(keyword)
  );
  
  return hasImportantKeywords ? IMPORTANCE_LEVELS.HIGH : IMPORTANCE_LEVELS.AVERAGE;
};
