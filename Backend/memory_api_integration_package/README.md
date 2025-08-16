# Memory Management API - Frontend Integration Package

## 📦 Package Overview

This comprehensive integration package provides everything needed to integrate the Memory Management API into the Gurukul Financial Simulator frontend application. The package includes complete documentation, code examples, testing tools, and error handling strategies.

## 📁 Package Contents

```
memory_api_integration_package/
├── README.md                     # This file - package overview and setup
├── memory_api_integration.md     # Complete API documentation and specifications
├── .env.memory.template          # Environment configuration template
├── memoryApiSlice.js            # RTK Query integration with all endpoints
├── memoryHooks.js               # React hooks and component examples
├── test_requests.http           # Postman/curl test requests
├── error_handling_guide.md      # Comprehensive error handling guide
└── integration_examples/        # Additional integration examples
    ├── ChatComponent.jsx        # Chat component with memory integration
    ├── MemoryDashboard.jsx      # Memory management dashboard
    └── PersonaSelector.jsx      # Persona selection component
```

## 🚀 Quick Start Guide

### 1. Environment Setup

1. **Copy Environment Template**
   ```bash
   cp .env.memory.template your-project/.env.local
   ```

2. **Configure Environment Variables**
   ```bash
   # Edit .env.local with your settings
   REACT_APP_MEMORY_API_BASE_URL=http://localhost:8003
   REACT_APP_MEMORY_API_KEY=your_api_key_here
   REACT_APP_DEFAULT_PERSONA_ID=financial_advisor
   ```

### 2. Install Dependencies

```bash
# Install required packages
npm install @reduxjs/toolkit react-redux
```

### 3. Add RTK Query Integration

1. **Copy API Slice**
   ```bash
   cp memoryApiSlice.js your-project/src/api/
   ```

2. **Configure Store**
   ```javascript
   // store.js
   import { configureStore } from '@reduxjs/toolkit';
   import { memoryApiSlice } from './api/memoryApiSlice';

   export const store = configureStore({
     reducer: {
       [memoryApiSlice.reducerPath]: memoryApiSlice.reducer,
     },
     middleware: (getDefaultMiddleware) =>
       getDefaultMiddleware().concat(memoryApiSlice.middleware),
   });
   ```

### 4. Add React Hooks

```bash
cp memoryHooks.js your-project/src/hooks/
```

### 5. Test Integration

```bash
# Use the test requests file to verify API connectivity
# Import test_requests.http into Postman or use with REST Client extension
```

## 🔧 Integration Steps

### Step 1: Basic Memory Storage

```javascript
import { useMemoryStorage } from './hooks/memoryHooks';

const YourComponent = () => {
  const { storeUserMemory, storeUserInteraction } = useMemoryStorage();

  const handleUserPreference = async (preference) => {
    await storeUserMemory({
      userId: currentUser.id,
      personaId: 'financial_advisor',
      content: preference,
      contentType: 'preference',
      tags: ['investment', 'preference'],
      importance: 8
    });
  };

  // Component JSX...
};
```

### Step 2: Retrieve Persona Context

```javascript
import { usePersonaMemories, useConversationContext } from './hooks/memoryHooks';

const ChatComponent = ({ userId, personaId }) => {
  const { memories } = usePersonaMemories(personaId, userId);
  const { conversationContext } = useConversationContext(userId, personaId);

  // Use memories and context for AI responses
  const generateResponse = async (userMessage) => {
    const context = {
      recentMemories: memories.slice(0, 5),
      conversationHistory: conversationContext,
      userMessage
    };
    
    // Send to your AI service with context
    return await aiService.generateResponse(context);
  };
};
```

### Step 3: Chain-of-Thought Processing

```javascript
import { useGetRecentInteractionsQuery } from './api/memoryApiSlice';

const useChainOfThought = (userId, personaId) => {
  const { data: interactions } = useGetRecentInteractionsQuery({
    limit: 5,
    userId,
    personaId
  });

  const buildContext = () => {
    return interactions?.interactions.map(interaction => ({
      user: interaction.user_message,
      assistant: interaction.agent_response,
      timestamp: interaction.timestamp
    })) || [];
  };

  return { context: buildContext() };
};
```

## 📋 Persona Configuration

### Default Persona Mapping

```javascript
const PERSONA_CONFIG = {
  // Financial Simulator
  'financial_advisor': {
    name: 'Financial Advisor',
    domain: 'finance',
    capabilities: ['investment_advice', 'portfolio_management', 'risk_assessment']
  },
  'budget_planner': {
    name: 'Budget Planner', 
    domain: 'finance',
    capabilities: ['budgeting', 'expense_tracking', 'savings_goals']
  },
  
  // Gurukul Education
  'gurukul_math_tutor': {
    name: 'Math Tutor',
    domain: 'education',
    capabilities: ['math_tutoring', 'problem_solving', 'concept_explanation']
  },
  
  // Karma & Vedic
  'karma_advisor': {
    name: 'Karma Advisor',
    domain: 'spiritual',
    capabilities: ['karma_guidance', 'spiritual_advice', 'life_coaching']
  }
};
```

### Dynamic Persona Selection

```javascript
import { usePersonaSummary } from './hooks/memoryHooks';

const PersonaSelector = ({ userId, onPersonaChange }) => {
  const [selectedPersona, setSelectedPersona] = useState('financial_advisor');
  const { stats } = usePersonaSummary(selectedPersona, userId);

  const handlePersonaChange = (personaId) => {
    setSelectedPersona(personaId);
    onPersonaChange(personaId);
  };

  return (
    <div className="persona-selector">
      {Object.entries(PERSONA_CONFIG).map(([id, config]) => (
        <button
          key={id}
          onClick={() => handlePersonaChange(id)}
          className={selectedPersona === id ? 'active' : ''}
        >
          {config.name}
          {stats && <span>({stats.totalMemories} memories)</span>}
        </button>
      ))}
    </div>
  );
};
```

## 🧪 Testing Your Integration

### 1. Health Check Test

```javascript
import { useCheckHealthQuery } from './api/memoryApiSlice';

const HealthCheck = () => {
  const { data: health, error, isLoading } = useCheckHealthQuery();

  if (isLoading) return <div>Checking API health...</div>;
  if (error) return <div>API Error: {error.message}</div>;
  if (health?.status === 'healthy') return <div>✅ API is healthy</div>;
  
  return <div>❌ API is not responding</div>;
};
```

### 2. Memory Storage Test

```javascript
const TestMemoryStorage = () => {
  const { storeUserMemory, isStoringMemory } = useMemoryStorage();

  const testStorage = async () => {
    const result = await storeUserMemory({
      userId: 'test_user',
      personaId: 'financial_advisor',
      content: 'Test memory content',
      contentType: 'text',
      tags: ['test'],
      importance: 5
    });

    console.log('Storage result:', result);
  };

  return (
    <button onClick={testStorage} disabled={isStoringMemory}>
      {isStoringMemory ? 'Storing...' : 'Test Memory Storage'}
    </button>
  );
};
```

### 3. Memory Retrieval Test

```javascript
const TestMemoryRetrieval = ({ personaId, userId }) => {
  const { memories, isLoading, error } = usePersonaMemories(personaId, userId);

  if (isLoading) return <div>Loading memories...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <h3>Retrieved Memories ({memories.length})</h3>
      {memories.map(memory => (
        <div key={memory.memory_id}>
          <strong>{memory.content_type}:</strong> {memory.content}
        </div>
      ))}
    </div>
  );
};
```

## 🔍 Debugging and Troubleshooting

### Enable Debug Mode

```javascript
// In your .env.local
REACT_APP_MEMORY_DEBUG_MODE=true
REACT_APP_LOG_MEMORY_REQUESTS=true
```

### Common Issues and Solutions

1. **API Key Issues**
   ```javascript
   // Verify API key is working
   const testApiKey = async () => {
     try {
       const response = await fetch(`${API_BASE_URL}/memory/health`, {
         headers: { 'Authorization': `Bearer ${API_KEY}` }
       });
       console.log('API Key Status:', response.ok ? 'Valid' : 'Invalid');
     } catch (error) {
       console.error('API Key Test Failed:', error);
     }
   };
   ```

2. **CORS Issues**
   ```javascript
   // Ensure your API server has CORS configured for your frontend domain
   // Check browser console for CORS errors
   ```

3. **Rate Limiting**
   ```javascript
   // Monitor rate limit headers
   const checkRateLimit = (response) => {
     const remaining = response.headers.get('X-RateLimit-Remaining');
     const reset = response.headers.get('X-RateLimit-Reset');
     console.log(`Rate limit: ${remaining} remaining, resets at ${reset}`);
   };
   ```

## 📚 Additional Resources

### Documentation Files

- **`memory_api_integration.md`** - Complete API endpoint documentation
- **`error_handling_guide.md`** - Comprehensive error handling strategies
- **`test_requests.http`** - Ready-to-use API test requests

### Code Examples

- **`memoryApiSlice.js`** - Complete RTK Query integration
- **`memoryHooks.js`** - React hooks and component examples
- **`integration_examples/`** - Additional component examples

### Environment Configuration

- **`.env.memory.template`** - Complete environment variable template

## 🤝 Support and Contribution

### Getting Help

1. **Check Documentation**: Review all provided documentation files
2. **Test with curl**: Use `test_requests.http` to verify API connectivity
3. **Enable Debug Mode**: Set debug flags in environment variables
4. **Check Error Handling Guide**: Review common error scenarios

### Best Practices

1. **Always validate data** before sending to API
2. **Implement proper error handling** with retry logic
3. **Use debouncing** for search and frequent operations
4. **Cache frequently accessed data** to reduce API calls
5. **Monitor rate limits** and implement backoff strategies

### Performance Tips

1. **Use pagination** for large data sets
2. **Implement request deduplication** for identical requests
3. **Cache persona summaries** for better UX
4. **Batch operations** when possible
5. **Use lazy loading** for non-critical data

## 🎯 Next Steps

1. **Copy files** to your project structure
2. **Configure environment** variables
3. **Test API connectivity** using provided test requests
4. **Implement basic memory storage** in your chat components
5. **Add persona-specific memory retrieval** for context
6. **Implement chain-of-thought** processing for AI responses
7. **Add error handling** and user feedback
8. **Monitor and optimize** performance

## 📞 Contact Information

For technical support or questions about the Memory Management API integration:

- **API Documentation**: Check `/memory/docs` endpoint when API is running
- **Test Requests**: Use the provided `test_requests.http` file
- **Error Handling**: Refer to `error_handling_guide.md`
- **Code Examples**: Review `memoryHooks.js` for implementation patterns

---

**Happy Integrating! 🚀**

This package provides everything needed for a successful Memory Management API integration. Follow the step-by-step guide, test thoroughly, and refer to the comprehensive documentation for any questions.
