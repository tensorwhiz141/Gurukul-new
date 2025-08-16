/**
 * Chat Component with Memory Integration
 * 
 * This component demonstrates how to integrate the Memory Management API
 * into a chat interface for the Gurukul Financial Simulator.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useMemoryStorage, useConversationContext, usePersonaMemories } from '../memoryHooks';
import { CONTENT_TYPES, IMPORTANCE_LEVELS } from '../memoryApiSlice';

const ChatComponent = ({ 
  userId, 
  personaId = 'financial_advisor', 
  sessionId,
  onPersonaChange 
}) => {
  // State management
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [showMemoryPanel, setShowMemoryPanel] = useState(false);
  
  // Refs
  const chatEndRef = useRef(null);
  const messageInputRef = useRef(null);

  // Memory hooks
  const { storeUserInteraction, isStoringMemory } = useMemoryStorage();
  const { conversationContext, contextSummary } = useConversationContext(userId, personaId);
  const { memories, importantMemories } = usePersonaMemories(personaId, userId, {
    limit: 10,
    contentTypes: [CONTENT_TYPES.PREFERENCE, CONTENT_TYPES.FACT]
  });

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // Focus input on mount
  useEffect(() => {
    messageInputRef.current?.focus();
  }, []);

  // Handle message sending
  const handleSendMessage = useCallback(async () => {
    if (!message.trim() || isTyping) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: message.trim(),
      timestamp: new Date().toISOString()
    };

    // Add user message to chat
    setChatHistory(prev => [...prev, userMessage]);
    setMessage('');
    setIsTyping(true);

    try {
      // Build context for AI
      const aiContext = {
        userMessage: userMessage.content,
        conversationHistory: conversationContext.slice(-5), // Last 5 interactions
        userMemories: memories.slice(0, 5), // Top 5 relevant memories
        importantFacts: importantMemories.filter(m => m.content_type === CONTENT_TYPES.FACT),
        userPreferences: importantMemories.filter(m => m.content_type === CONTENT_TYPES.PREFERENCE),
        personaId,
        userId
      };

      // Generate AI response (replace with your AI service)
      const agentResponse = await generateAIResponse(aiContext);

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: agentResponse.content,
        timestamp: new Date().toISOString(),
        confidence: agentResponse.confidence,
        responseTime: agentResponse.responseTime
      };

      // Add assistant message to chat
      setChatHistory(prev => [...prev, assistantMessage]);

      // Store interaction in memory
      await storeUserInteraction({
        userId,
        personaId,
        userMessage: userMessage.content,
        agentResponse: agentResponse.content,
        sessionId,
        domain: getPersonaDomain(personaId),
        intent: agentResponse.detectedIntent,
        responseTime: agentResponse.responseTime,
        confidence: agentResponse.confidence,
        modelUsed: agentResponse.modelUsed
      });

      // Auto-store important user information
      await autoStoreUserInfo(userMessage.content, userId, personaId);

    } catch (error) {
      console.error('Error in chat:', error);
      
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'I apologize, but I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
        isError: true
      };
      
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  }, [message, isTyping, conversationContext, memories, importantMemories, userId, personaId, sessionId, storeUserInteraction]);

  // Handle key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Clear chat history
  const clearChat = () => {
    setChatHistory([]);
  };

  return (
    <div className="chat-container">
      {/* Chat Header */}
      <div className="chat-header">
        <div className="persona-info">
          <h3>{getPersonaDisplayName(personaId)}</h3>
          <span className="persona-status">
            {memories.length} memories • {contextSummary?.totalInteractions || 0} interactions
          </span>
        </div>
        
        <div className="chat-controls">
          <button 
            onClick={() => setShowMemoryPanel(!showMemoryPanel)}
            className={`memory-toggle ${showMemoryPanel ? 'active' : ''}`}
            title="Toggle memory panel"
          >
            🧠 Memory
          </button>
          
          <button onClick={clearChat} className="clear-chat" title="Clear chat">
            🗑️ Clear
          </button>
        </div>
      </div>

      <div className="chat-body">
        {/* Memory Panel */}
        {showMemoryPanel && (
          <div className="memory-panel">
            <h4>Context & Memories</h4>
            
            {/* Important Memories */}
            <div className="memory-section">
              <h5>Key Information</h5>
              {importantMemories.slice(0, 3).map(memory => (
                <div key={memory.memory_id} className="memory-item">
                  <span className="memory-type">{memory.content_type}</span>
                  <span className="memory-content">{memory.content.substring(0, 80)}...</span>
                  <span className="memory-importance">★{memory.metadata.importance}</span>
                </div>
              ))}
            </div>

            {/* Recent Context */}
            {contextSummary && (
              <div className="context-section">
                <h5>Recent Context</h5>
                <div className="context-summary">
                  <p><strong>Last interaction:</strong> {contextSummary.lastUserMessage?.substring(0, 50)}...</p>
                  <p><strong>Total interactions:</strong> {contextSummary.totalInteractions}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Chat Messages */}
        <div className="chat-messages">
          {chatHistory.length === 0 && (
            <div className="welcome-message">
              <h4>Welcome to {getPersonaDisplayName(personaId)}!</h4>
              <p>I remember our previous conversations and your preferences. How can I help you today?</p>
              
              {/* Suggested questions based on memories */}
              {memories.length > 0 && (
                <div className="suggested-questions">
                  <p>Based on our history, you might want to ask:</p>
                  <ul>
                    {generateSuggestedQuestions(memories, personaId).map((question, index) => (
                      <li key={index}>
                        <button 
                          onClick={() => setMessage(question)}
                          className="suggestion-button"
                        >
                          {question}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {chatHistory.map((msg) => (
            <div key={msg.id} className={`message ${msg.role} ${msg.isError ? 'error' : ''}`}>
              <div className="message-header">
                <span className="message-role">
                  {msg.role === 'user' ? 'You' : getPersonaDisplayName(personaId)}
                </span>
                <span className="message-time">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </span>
                {msg.confidence && (
                  <span className="message-confidence" title={`Confidence: ${Math.round(msg.confidence * 100)}%`}>
                    {msg.confidence > 0.8 ? '🟢' : msg.confidence > 0.6 ? '🟡' : '🔴'}
                  </span>
                )}
              </div>
              
              <div className="message-content">
                {msg.content}
              </div>

              {msg.responseTime && (
                <div className="message-meta">
                  Response time: {msg.responseTime.toFixed(1)}s
                </div>
              )}
            </div>
          ))}

          {/* Typing indicator */}
          {isTyping && (
            <div className="message assistant typing">
              <div className="message-header">
                <span className="message-role">{getPersonaDisplayName(personaId)}</span>
              </div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>
      </div>

      {/* Chat Input */}
      <div className="chat-input">
        <div className="input-container">
          <textarea
            ref={messageInputRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={`Ask ${getPersonaDisplayName(personaId)} anything...`}
            rows={1}
            disabled={isTyping}
          />
          
          <button 
            onClick={handleSendMessage}
            disabled={!message.trim() || isTyping}
            className="send-button"
          >
            {isTyping ? '⏳' : '📤'}
          </button>
        </div>

        {/* Memory storage indicator */}
        {isStoringMemory && (
          <div className="storage-indicator">
            💾 Saving to memory...
          </div>
        )}
      </div>
    </div>
  );
};

// Helper functions

const getPersonaDisplayName = (personaId) => {
  const names = {
    'financial_advisor': 'Financial Advisor',
    'budget_planner': 'Budget Planner',
    'investment_coach': 'Investment Coach',
    'gurukul_math_tutor': 'Math Tutor',
    'karma_advisor': 'Karma Advisor',
    'ask_vedas': 'Vedic Scholar'
  };
  return names[personaId] || 'AI Assistant';
};

const getPersonaDomain = (personaId) => {
  if (personaId.includes('financial') || personaId.includes('budget') || personaId.includes('investment')) {
    return 'finance';
  }
  if (personaId.includes('gurukul') || personaId.includes('tutor')) {
    return 'education';
  }
  if (personaId.includes('karma') || personaId.includes('vedas')) {
    return 'spiritual';
  }
  return 'general';
};

const generateSuggestedQuestions = (memories, personaId) => {
  const domain = getPersonaDomain(personaId);
  
  const suggestions = {
    finance: [
      "What's my current investment strategy?",
      "How should I adjust my portfolio?",
      "What are my financial goals?"
    ],
    education: [
      "What topics should we review?",
      "Can you help me with practice problems?",
      "What's my learning progress?"
    ],
    spiritual: [
      "What guidance do you have for me today?",
      "How can I improve my karma?",
      "What wisdom can you share?"
    ]
  };

  return suggestions[domain] || ["How can you help me today?"];
};

const generateAIResponse = async (context) => {
  // Mock AI response - replace with your actual AI service
  const startTime = Date.now();
  
  // Simulate API call delay
  await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));
  
  const responseTime = (Date.now() - startTime) / 1000;
  
  // Build response based on context
  let response = "I understand you're asking about: " + context.userMessage;
  
  if (context.userPreferences.length > 0) {
    response += "\n\nBased on your preferences: " + context.userPreferences[0].content;
  }
  
  if (context.importantFacts.length > 0) {
    response += "\n\nConsidering your situation: " + context.importantFacts[0].content;
  }

  return {
    content: response,
    confidence: 0.85 + Math.random() * 0.15,
    responseTime,
    detectedIntent: 'general_inquiry',
    modelUsed: 'gpt-4'
  };
};

const autoStoreUserInfo = async (userMessage, userId, personaId) => {
  // Auto-detect and store important user information
  const importantPatterns = [
    { pattern: /i earn|my income|i make.*\$/, type: CONTENT_TYPES.FACT, importance: IMPORTANCE_LEVELS.HIGH },
    { pattern: /i prefer|i like|i want/, type: CONTENT_TYPES.PREFERENCE, importance: IMPORTANCE_LEVELS.MODERATE },
    { pattern: /my goal|i plan to|i'm planning/, type: CONTENT_TYPES.FACT, importance: IMPORTANCE_LEVELS.HIGH }
  ];

  for (const { pattern, type, importance } of importantPatterns) {
    if (pattern.test(userMessage.toLowerCase())) {
      // Store this as a memory (implement based on your memory storage hook)
      console.log('Auto-storing user info:', { userMessage, type, importance });
      break;
    }
  }
};

export default ChatComponent;
