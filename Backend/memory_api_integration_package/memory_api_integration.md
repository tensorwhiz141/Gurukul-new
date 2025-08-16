# Memory Management API - Frontend Integration Guide

## Overview

This guide provides complete integration instructions for the Memory Management API into the Gurukul Financial Simulator frontend application. The API enables persona-based memory storage, retrieval, and chain-of-thought processing for AI agents.

## Base URLs

```javascript
const API_BASE_URLS = {
  DEVELOPMENT: "http://localhost:8003",
  STAGING: "http://staging-api.gurukul.com:8003", 
  PRODUCTION: "https://api.gurukul.com"  // Configure as needed
};
```

## Authentication

All API requests require Bearer token authentication:

```javascript
const headers = {
  'Authorization': `Bearer ${process.env.REACT_APP_MEMORY_API_KEY}`,
  'Content-Type': 'application/json'
};
```

## Core API Endpoints

### 1. Memory Storage

#### Store Memory Chunk
```http
POST /memory
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "user_id": "user123",
  "persona_id": "financial_advisor",
  "content": "User prefers conservative investment strategies with low risk tolerance",
  "content_type": "preference",
  "metadata": {
    "tags": ["investment", "conservative", "low-risk"],
    "importance": 8,
    "topic": "investment_strategy",
    "context_type": "user_preference",
    "source": "user_input",
    "confidence": 0.95
  }
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Memory chunk created successfully",
  "data": {
    "memory_id": "550e8400-e29b-41d4-a716-446655440000"
  },
  "timestamp": "2023-12-01T12:00:00Z"
}
```

#### Store User-Agent Interaction
```http
POST /memory/interaction
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "user_id": "user123",
  "persona_id": "financial_advisor", 
  "user_message": "What's the best investment strategy for someone my age?",
  "agent_response": "Based on your conservative preference and 30-year timeline, I recommend a balanced portfolio with 60% bonds and 40% stocks.",
  "context": {
    "session_id": "session_001",
    "conversation_turn": 1,
    "domain": "finance",
    "intent": "investment_advice",
    "previous_context": "User completed risk assessment"
  },
  "metadata": {
    "response_time": 1.5,
    "confidence": 0.92,
    "model_used": "financial_advisor_v1",
    "tags": ["investment", "advice"]
  }
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Interaction created successfully", 
  "data": {
    "interaction_id": "660e8400-e29b-41d4-a716-446655440001"
  },
  "timestamp": "2023-12-01T12:00:00Z"
}
```

### 2. Memory Retrieval

#### Get Memories by Persona
```http
GET /memory?persona={persona_id}&limit={number}&user_id={user_id}
Authorization: Bearer <api_key>

Examples:
/memory?persona=financial_advisor&limit=10
/memory?persona=financial_advisor&user_id=user123&limit=5
/memory?persona=financial_advisor&content_type=preference&min_importance=7
```

**Response (200 OK):**
```json
{
  "memories": [
    {
      "memory_id": "550e8400-e29b-41d4-a716-446655440000",
      "user_id": "user123",
      "persona_id": "financial_advisor",
      "content": "User prefers conservative investment strategies",
      "content_type": "preference",
      "metadata": {
        "tags": ["investment", "conservative"],
        "importance": 8,
        "topic": "investment_strategy"
      },
      "timestamp": "2023-12-01T12:00:00Z",
      "created_at": "2023-12-01T12:00:00Z",
      "is_active": true
    }
  ],
  "total_count": 1,
  "page": 1,
  "page_size": 10,
  "has_next": false,
  "has_previous": false
}
```

#### Get Recent Interactions (Chain-of-Thought)
```http
GET /memory?limit={number}&recent_interactions=true&user_id={user_id}
Authorization: Bearer <api_key>

Examples:
/memory?limit=5&recent_interactions=true&user_id=user123
/memory?limit=3&recent_interactions=true&persona=financial_advisor
```

**Response (200 OK):**
```json
{
  "interactions": [
    {
      "interaction_id": "660e8400-e29b-41d4-a716-446655440001",
      "user_id": "user123",
      "persona_id": "financial_advisor",
      "user_message": "What's the best investment strategy?",
      "agent_response": "Based on your profile, I recommend...",
      "context": {
        "session_id": "session_001",
        "conversation_turn": 1,
        "domain": "finance"
      },
      "timestamp": "2023-12-01T12:00:00Z"
    }
  ],
  "total_count": 1,
  "page": 1,
  "page_size": 5
}
```

### 3. Memory Search

#### Search Memories by Text
```http
GET /memory/search?query={text}&persona_id={persona}&limit={number}
Authorization: Bearer <api_key>

Examples:
/memory/search?query=investment%20strategy&persona_id=financial_advisor&limit=5
/memory/search?query=conservative&user_id=user123&limit=10
```

**Response (200 OK):**
```json
{
  "results": [
    {
      "memory_id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "User prefers conservative investment strategies",
      "content_type": "preference",
      "metadata": {
        "tags": ["investment", "conservative"],
        "importance": 8
      }
    }
  ],
  "query": "investment strategy",
  "total_results": 1,
  "search_time": 0.045
}
```

### 4. Persona Memory Summary

#### Get Persona Memory Summary
```http
GET /memory/persona/{persona_id}/summary?user_id={user_id}
Authorization: Bearer <api_key>

Example:
/memory/persona/financial_advisor/summary?user_id=user123
```

**Response (200 OK):**
```json
{
  "persona_id": "financial_advisor",
  "user_id": "user123",
  "total_memories": 25,
  "memory_categories": {
    "preference": 8,
    "fact": 12,
    "interaction": 3,
    "context": 2
  },
  "last_interaction": "2023-12-01T12:00:00Z",
  "recent_topics": ["investment_strategy", "risk_assessment", "portfolio_management"],
  "updated_at": "2023-12-01T12:00:00Z"
}
```

## Content Types

```javascript
const CONTENT_TYPES = {
  TEXT: "text",           // General text content
  INTERACTION: "interaction", // User-agent interaction
  CONTEXT: "context",     // Contextual information
  REFLECTION: "reflection", // Agent reflection/analysis
  PREFERENCE: "preference", // User preference
  FACT: "fact"           // Factual information
};
```

## Importance Levels

```javascript
const IMPORTANCE_LEVELS = {
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
```

## Persona Mapping

```javascript
const PERSONA_MAPPING = {
  // Financial Simulator Personas
  'financial_advisor': 'financial_advisor',
  'investment_coach': 'investment_coach',
  'budget_planner': 'budget_planner',
  'retirement_advisor': 'retirement_advisor',
  
  // Gurukul Education Personas
  'math_tutor': 'gurukul_math_tutor',
  'science_tutor': 'gurukul_science_tutor',
  'language_tutor': 'gurukul_language_tutor',
  
  // Karma & Vedic Personas
  'karma_agent': 'karma_advisor',
  'vedic_scholar': 'ask_vedas',
  'spiritual_guide': 'spiritual_advisor',
  
  // General Assistant
  'general_assistant': 'general_ai_assistant'
};
```

## Rate Limits

```javascript
const RATE_LIMITS = {
  HOURLY_LIMIT: 1000,     // Requests per hour per API key
  BURST_LIMIT: 100,       // Requests per minute per API key
  IP_LIMIT: 200          // Requests per 5 minutes per IP
};
```

## Error Response Format

```json
{
  "error": "ERROR_TYPE",
  "message": "Human-readable error message",
  "details": [
    {
      "field": "field_name",
      "message": "Field-specific error message",
      "code": "ERROR_CODE"
    }
  ],
  "timestamp": "2023-12-01T12:00:00Z",
  "request_id": "req_123456"
}
```

## Common HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Invalid or missing API key
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Integration Best Practices

### 1. Memory Storage Strategy
```javascript
// Store memories after significant user interactions
const storeUserPreference = async (preference) => {
  await storeMemory({
    user_id: currentUser.id,
    persona_id: currentPersona,
    content: preference,
    content_type: "preference",
    metadata: {
      tags: extractTags(preference),
      importance: calculateImportance(preference),
      topic: detectTopic(preference),
      source: "user_input"
    }
  });
};

// Store interactions after agent responses
const storeInteraction = async (userMessage, agentResponse) => {
  await storeMemoryInteraction({
    user_id: currentUser.id,
    persona_id: currentPersona,
    user_message: userMessage,
    agent_response: agentResponse,
    context: {
      session_id: sessionId,
      conversation_turn: turnNumber,
      domain: getCurrentDomain(),
      intent: detectIntent(userMessage)
    },
    metadata: {
      response_time: responseTime,
      confidence: agentConfidence,
      model_used: currentModel
    }
  });
};
```

### 2. Context Building for Chain-of-Thought
```javascript
const buildConversationContext = async (userId, limit = 5) => {
  const recentInteractions = await getRecentInteractions({
    limit,
    userId,
    persona: currentPersona
  });
  
  return recentInteractions.interactions.map(interaction => ({
    role: 'user',
    content: interaction.user_message,
    timestamp: interaction.timestamp
  })).concat(recentInteractions.interactions.map(interaction => ({
    role: 'assistant', 
    content: interaction.agent_response,
    timestamp: interaction.timestamp
  }))).sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
};
```

### 3. Memory-Enhanced Agent Responses
```javascript
const getPersonaContext = async (personaId, userId) => {
  const [memories, summary] = await Promise.all([
    getPersonaMemories({ personaId, userId, limit: 10 }),
    getPersonaMemorySummary(personaId, userId)
  ]);
  
  return {
    preferences: memories.memories.filter(m => m.content_type === 'preference'),
    facts: memories.memories.filter(m => m.content_type === 'fact'),
    summary: summary,
    recentTopics: summary.recent_topics
  };
};
```
