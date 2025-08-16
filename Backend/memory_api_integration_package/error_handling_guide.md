# Memory Management API - Error Handling Guide

## Overview

This guide provides comprehensive error handling strategies, common error scenarios, troubleshooting steps, and retry logic for the Memory Management API integration.

## Error Response Format

All API errors follow a consistent format:

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

## HTTP Status Codes

| Status Code | Description | Action Required |
|-------------|-------------|-----------------|
| 200 | OK | Success - no action needed |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Fix request parameters |
| 401 | Unauthorized | Check API key |
| 404 | Not Found | Verify resource exists |
| 422 | Validation Error | Fix request body validation |
| 429 | Rate Limited | Implement retry with backoff |
| 500 | Server Error | Retry request, contact support if persistent |
| 503 | Service Unavailable | Service temporarily down, retry later |

## Common Error Scenarios

### 1. Authentication Errors (401)

#### Missing Authorization Header
```json
{
  "error": "UNAUTHORIZED",
  "message": "Missing authorization credentials",
  "details": [],
  "timestamp": "2023-12-01T12:00:00Z"
}
```

**Solution:**
```javascript
// Ensure Authorization header is included
const headers = {
  'Authorization': `Bearer ${process.env.REACT_APP_MEMORY_API_KEY}`,
  'Content-Type': 'application/json'
};
```

#### Invalid API Key
```json
{
  "error": "UNAUTHORIZED", 
  "message": "Invalid API key",
  "details": [],
  "timestamp": "2023-12-01T12:00:00Z"
}
```

**Solution:**
```javascript
// Verify API key is correct and active
const validateApiKey = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/memory/health`, {
      headers: { 'Authorization': `Bearer ${API_KEY}` }
    });
    return response.ok;
  } catch (error) {
    console.error('API key validation failed:', error);
    return false;
  }
};
```

### 2. Validation Errors (422)

#### Missing Required Fields
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": [
    {
      "field": "user_id",
      "message": "Field required",
      "code": "REQUIRED_FIELD"
    },
    {
      "field": "content",
      "message": "Content cannot be empty",
      "code": "EMPTY_CONTENT"
    }
  ],
  "timestamp": "2023-12-01T12:00:00Z"
}
```

**Solution:**
```javascript
// Validate required fields before sending request
const validateMemoryData = (data) => {
  const errors = [];
  
  if (!data.user_id) errors.push('user_id is required');
  if (!data.persona_id) errors.push('persona_id is required');
  if (!data.content || !data.content.trim()) errors.push('content cannot be empty');
  
  return errors;
};

const storeMemory = async (memoryData) => {
  const validationErrors = validateMemoryData(memoryData);
  if (validationErrors.length > 0) {
    throw new Error(`Validation failed: ${validationErrors.join(', ')}`);
  }
  
  // Proceed with API call
};
```

#### Invalid Content Type
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Invalid content type",
  "details": [
    {
      "field": "content_type",
      "message": "Must be one of: text, interaction, context, reflection, preference, fact",
      "code": "INVALID_ENUM"
    }
  ]
}
```

**Solution:**
```javascript
const VALID_CONTENT_TYPES = ['text', 'interaction', 'context', 'reflection', 'preference', 'fact'];

const validateContentType = (contentType) => {
  if (!VALID_CONTENT_TYPES.includes(contentType)) {
    throw new Error(`Invalid content type. Must be one of: ${VALID_CONTENT_TYPES.join(', ')}`);
  }
};
```

### 3. Rate Limiting Errors (429)

```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded",
  "details": [],
  "timestamp": "2023-12-01T12:00:00Z"
}
```

**Headers:**
```
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640995200
Retry-After: 3600
```

**Solution with Exponential Backoff:**
```javascript
class RateLimitHandler {
  constructor() {
    this.retryAttempts = 0;
    this.maxRetries = 3;
    this.baseDelay = 1000; // 1 second
  }

  async handleRateLimit(error, originalRequest) {
    if (this.retryAttempts >= this.maxRetries) {
      throw new Error('Max retry attempts exceeded');
    }

    const retryAfter = error.headers?.['retry-after'];
    const delay = retryAfter ? 
      parseInt(retryAfter) * 1000 : 
      this.baseDelay * Math.pow(2, this.retryAttempts);

    this.retryAttempts++;
    
    console.log(`Rate limited. Retrying in ${delay}ms (attempt ${this.retryAttempts})`);
    
    await new Promise(resolve => setTimeout(resolve, delay));
    return originalRequest();
  }

  reset() {
    this.retryAttempts = 0;
  }
}

// Usage
const rateLimitHandler = new RateLimitHandler();

const apiCallWithRetry = async (apiCall) => {
  try {
    const result = await apiCall();
    rateLimitHandler.reset();
    return result;
  } catch (error) {
    if (error.status === 429) {
      return rateLimitHandler.handleRateLimit(error, apiCall);
    }
    throw error;
  }
};
```

### 4. Server Errors (500)

```json
{
  "error": "INTERNAL_ERROR",
  "message": "An unexpected error occurred",
  "details": [],
  "timestamp": "2023-12-01T12:00:00Z",
  "request_id": "req_123456"
}
```

**Solution:**
```javascript
const handleServerError = async (error, originalRequest, retryCount = 0) => {
  const maxRetries = 3;
  const baseDelay = 1000;

  if (retryCount >= maxRetries) {
    // Log error for monitoring
    console.error('Server error after max retries:', {
      error: error.message,
      requestId: error.request_id,
      timestamp: new Date().toISOString()
    });
    
    // Show user-friendly message
    throw new Error('Service temporarily unavailable. Please try again later.');
  }

  const delay = baseDelay * Math.pow(2, retryCount);
  console.log(`Server error. Retrying in ${delay}ms (attempt ${retryCount + 1})`);
  
  await new Promise(resolve => setTimeout(resolve, delay));
  
  try {
    return await originalRequest();
  } catch (retryError) {
    return handleServerError(retryError, originalRequest, retryCount + 1);
  }
};
```

### 5. Network Errors

**Connection Timeout:**
```javascript
const handleNetworkError = (error) => {
  if (error.name === 'AbortError' || error.code === 'NETWORK_ERROR') {
    return {
      type: 'NETWORK_ERROR',
      message: 'Network connection failed. Please check your internet connection.',
      retry: true
    };
  }
  
  if (error.name === 'TimeoutError') {
    return {
      type: 'TIMEOUT_ERROR', 
      message: 'Request timed out. Please try again.',
      retry: true
    };
  }
  
  return {
    type: 'UNKNOWN_ERROR',
    message: 'An unexpected error occurred.',
    retry: false
  };
};
```

## Error Handling Patterns

### 1. React Hook Error Handling

```javascript
import { useState, useCallback } from 'react';
import { useStoreMemoryMutation } from './memoryApiSlice';

export const useMemoryWithErrorHandling = () => {
  const [error, setError] = useState(null);
  const [isRetrying, setIsRetrying] = useState(false);
  
  const [storeMemory, { isLoading }] = useStoreMemoryMutation();

  const storeMemoryWithRetry = useCallback(async (memoryData, retryCount = 0) => {
    const maxRetries = 3;
    
    try {
      setError(null);
      const result = await storeMemory(memoryData).unwrap();
      return { success: true, data: result };
    } catch (err) {
      console.error('Memory storage error:', err);
      
      // Handle specific error types
      if (err.status === 429 && retryCount < maxRetries) {
        setIsRetrying(true);
        const delay = 1000 * Math.pow(2, retryCount);
        
        await new Promise(resolve => setTimeout(resolve, delay));
        setIsRetrying(false);
        
        return storeMemoryWithRetry(memoryData, retryCount + 1);
      }
      
      if (err.status >= 500 && retryCount < maxRetries) {
        setIsRetrying(true);
        const delay = 1000 * Math.pow(2, retryCount);
        
        await new Promise(resolve => setTimeout(resolve, delay));
        setIsRetrying(false);
        
        return storeMemoryWithRetry(memoryData, retryCount + 1);
      }
      
      // Set user-friendly error message
      const errorMessage = getErrorMessage(err);
      setError(errorMessage);
      
      return { success: false, error: errorMessage };
    }
  }, [storeMemory]);

  return {
    storeMemoryWithRetry,
    isLoading: isLoading || isRetrying,
    error
  };
};

const getErrorMessage = (error) => {
  switch (error.status) {
    case 401:
      return 'Authentication failed. Please check your API key.';
    case 422:
      return 'Invalid data provided. Please check your input.';
    case 429:
      return 'Too many requests. Please wait a moment and try again.';
    case 500:
      return 'Server error. Please try again later.';
    default:
      return error.message || 'An unexpected error occurred.';
  }
};
```

### 2. Global Error Handler

```javascript
// errorHandler.js
class MemoryAPIErrorHandler {
  constructor() {
    this.errorCallbacks = [];
    this.retryQueue = new Map();
  }

  onError(callback) {
    this.errorCallbacks.push(callback);
  }

  async handleError(error, context = {}) {
    // Log error
    console.error('Memory API Error:', {
      error: error.message,
      status: error.status,
      context,
      timestamp: new Date().toISOString()
    });

    // Notify error callbacks
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error, context);
      } catch (callbackError) {
        console.error('Error in error callback:', callbackError);
      }
    });

    // Handle retryable errors
    if (this.isRetryableError(error)) {
      return this.scheduleRetry(error, context);
    }

    throw error;
  }

  isRetryableError(error) {
    return error.status === 429 || error.status >= 500;
  }

  async scheduleRetry(error, context) {
    const retryKey = `${context.endpoint}_${context.requestId}`;
    const retryCount = this.retryQueue.get(retryKey) || 0;
    
    if (retryCount >= 3) {
      this.retryQueue.delete(retryKey);
      throw new Error('Max retry attempts exceeded');
    }

    this.retryQueue.set(retryKey, retryCount + 1);
    
    const delay = 1000 * Math.pow(2, retryCount);
    await new Promise(resolve => setTimeout(resolve, delay));
    
    // Return retry indicator
    return { shouldRetry: true, retryCount: retryCount + 1 };
  }
}

export const memoryErrorHandler = new MemoryAPIErrorHandler();
```

### 3. User Notification System

```javascript
// notificationSystem.js
export const showErrorNotification = (error) => {
  const notifications = {
    401: {
      title: 'Authentication Error',
      message: 'Please check your API credentials and try again.',
      type: 'error',
      duration: 5000
    },
    422: {
      title: 'Invalid Data',
      message: 'Please check your input and try again.',
      type: 'warning',
      duration: 4000
    },
    429: {
      title: 'Rate Limited',
      message: 'Too many requests. Please wait a moment.',
      type: 'warning',
      duration: 3000
    },
    500: {
      title: 'Server Error',
      message: 'Something went wrong. We\'re working to fix it.',
      type: 'error',
      duration: 5000
    }
  };

  const notification = notifications[error.status] || {
    title: 'Error',
    message: 'An unexpected error occurred.',
    type: 'error',
    duration: 4000
  };

  // Show notification (implement based on your notification system)
  showNotification(notification);
};
```

## Monitoring and Debugging

### 1. Error Logging

```javascript
const logError = (error, context) => {
  const errorLog = {
    timestamp: new Date().toISOString(),
    error: {
      message: error.message,
      status: error.status,
      stack: error.stack
    },
    context: {
      userId: context.userId,
      personaId: context.personaId,
      endpoint: context.endpoint,
      requestId: context.requestId
    },
    userAgent: navigator.userAgent,
    url: window.location.href
  };

  // Send to logging service
  if (process.env.NODE_ENV === 'production') {
    sendToLoggingService(errorLog);
  } else {
    console.error('Memory API Error Log:', errorLog);
  }
};
```

### 2. Health Check Monitoring

```javascript
export const useAPIHealthMonitor = () => {
  const [healthStatus, setHealthStatus] = useState('unknown');
  const { data: health, error, refetch } = useCheckHealthQuery(undefined, {
    pollingInterval: 30000 // Check every 30 seconds
  });

  useEffect(() => {
    if (health) {
      setHealthStatus('healthy');
    } else if (error) {
      setHealthStatus('unhealthy');
      console.error('API health check failed:', error);
    }
  }, [health, error]);

  return {
    healthStatus,
    isHealthy: healthStatus === 'healthy',
    checkHealth: refetch
  };
};
```

## Best Practices

### 1. Graceful Degradation

```javascript
const useMemoryWithFallback = (personaId, userId) => {
  const { data: memories, error } = useGetPersonaMemoriesQuery({
    personaId,
    userId
  });

  // Fallback to local storage if API fails
  const fallbackMemories = useMemo(() => {
    if (error && error.status >= 500) {
      const cached = localStorage.getItem(`memories_${personaId}_${userId}`);
      return cached ? JSON.parse(cached) : [];
    }
    return null;
  }, [error, personaId, userId]);

  return {
    memories: memories?.memories || fallbackMemories || [],
    isUsingFallback: !!fallbackMemories,
    error
  };
};
```

### 2. Request Deduplication

```javascript
const requestCache = new Map();

const deduplicateRequest = (key, requestFn) => {
  if (requestCache.has(key)) {
    return requestCache.get(key);
  }

  const promise = requestFn().finally(() => {
    requestCache.delete(key);
  });

  requestCache.set(key, promise);
  return promise;
};
```

### 3. Circuit Breaker Pattern

```javascript
class CircuitBreaker {
  constructor(threshold = 5, timeout = 60000) {
    this.failureThreshold = threshold;
    this.timeout = timeout;
    this.failureCount = 0;
    this.lastFailureTime = null;
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
  }

  async call(fn) {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime > this.timeout) {
        this.state = 'HALF_OPEN';
      } else {
        throw new Error('Circuit breaker is OPEN');
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  onSuccess() {
    this.failureCount = 0;
    this.state = 'CLOSED';
  }

  onFailure() {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    
    if (this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
    }
  }
}
```

## Troubleshooting Checklist

### Before Contacting Support

1. **Check API Key**
   - Verify API key is correct and active
   - Ensure proper Authorization header format

2. **Validate Request Data**
   - Check required fields are present
   - Verify data types and formats
   - Ensure content length limits

3. **Check Network Connectivity**
   - Test basic internet connection
   - Verify API base URL is accessible
   - Check for firewall/proxy issues

4. **Review Rate Limits**
   - Check current usage against limits
   - Implement proper retry logic
   - Monitor rate limit headers

5. **Test with Simple Request**
   - Try health check endpoint first
   - Use minimal request data
   - Test with curl or Postman

### Support Information to Provide

When contacting support, include:
- Request ID from error response
- Full error message and status code
- Request payload (sanitized)
- Timestamp of the error
- Steps to reproduce
- Browser/environment information
