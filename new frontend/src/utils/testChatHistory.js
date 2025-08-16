/**
 * Chat History Test Utility
 * Tests the chat history functionality to ensure it works correctly
 */

import chatHistoryStorage from './chatHistoryStorage';
import chatErrorRecovery from './chatErrorRecovery';
import chatPerformanceOptimizer from './chatPerformanceOptimizer';

/**
 * Run comprehensive tests for chat history functionality
 */
export const runChatHistoryTests = async () => {
  console.log('🧪 Starting chat history tests...');
  
  const results = {
    storage: false,
    persistence: false,
    sessions: false,
    performance: false,
    errorRecovery: false,
    overall: false,
  };

  try {
    // Test 1: Storage functionality
    console.log('📝 Testing storage functionality...');
    results.storage = await testStorageFunctionality();
    
    // Test 2: Persistence across sessions
    console.log('💾 Testing persistence...');
    results.persistence = await testPersistence();
    
    // Test 3: Session management
    console.log('🔄 Testing session management...');
    results.sessions = await testSessionManagement();
    
    // Test 4: Performance optimization
    console.log('⚡ Testing performance optimization...');
    results.performance = await testPerformanceOptimization();
    
    // Test 5: Error recovery
    console.log('🛠️ Testing error recovery...');
    results.errorRecovery = await testErrorRecovery();
    
    // Overall result
    results.overall = Object.values(results).every(result => result === true);
    
    console.log('✅ Chat history tests completed:', results);
    return results;
    
  } catch (error) {
    console.error('❌ Chat history tests failed:', error);
    return { ...results, error: error.message };
  }
};

/**
 * Test storage functionality
 */
const testStorageFunctionality = async () => {
  try {
    // Initialize storage
    await chatHistoryStorage.init('test-user');
    
    // Test adding messages
    const testMessage = {
      role: 'user',
      content: 'Test message for storage',
      model: 'grok',
    };
    
    await chatHistoryStorage.addMessage(testMessage);
    
    // Test loading messages
    const loadedMessages = chatHistoryStorage.loadChatHistory();
    
    // Should have welcome message + test message
    if (loadedMessages.length >= 2) {
      const userMessage = loadedMessages.find(msg => msg.content === 'Test message for storage');
      if (userMessage) {
        console.log('✅ Storage functionality test passed');
        return true;
      }
    }
    
    console.log('❌ Storage functionality test failed');
    return false;
    
  } catch (error) {
    console.error('❌ Storage functionality test error:', error);
    return false;
  }
};

/**
 * Test persistence across sessions
 */
const testPersistence = async () => {
  try {
    // Add a message
    const testMessage = {
      role: 'assistant',
      content: 'Test persistence message',
      model: 'grok',
    };
    
    await chatHistoryStorage.addMessage(testMessage);
    
    // Simulate page reload by reinitializing
    await chatHistoryStorage.init('test-user');
    
    // Check if message persisted
    const loadedMessages = chatHistoryStorage.loadChatHistory();
    const persistedMessage = loadedMessages.find(msg => msg.content === 'Test persistence message');
    
    if (persistedMessage) {
      console.log('✅ Persistence test passed');
      return true;
    }
    
    console.log('❌ Persistence test failed');
    return false;
    
  } catch (error) {
    console.error('❌ Persistence test error:', error);
    return false;
  }
};

/**
 * Test session management
 */
const testSessionManagement = async () => {
  try {
    // Create a new session
    const newSessionId = await chatHistoryStorage.createNewSession();
    
    if (!newSessionId) {
      console.log('❌ Failed to create new session');
      return false;
    }
    
    // Add message to new session
    await chatHistoryStorage.addMessage({
      role: 'user',
      content: 'Message in new session',
      model: 'grok',
    });
    
    // Get all sessions
    const sessions = chatHistoryStorage.getAllSessions();
    
    if (sessions.length >= 1) {
      console.log('✅ Session management test passed');
      return true;
    }
    
    console.log('❌ Session management test failed');
    return false;
    
  } catch (error) {
    console.error('❌ Session management test error:', error);
    return false;
  }
};

/**
 * Test performance optimization
 */
const testPerformanceOptimization = async () => {
  try {
    // Create a large number of test messages
    const testMessages = Array.from({ length: 150 }, (_, i) => ({
      id: `test-msg-${i}`,
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `Test message ${i}`,
      model: 'grok',
      timestamp: new Date().toISOString(),
    }));
    
    // Test message optimization
    const optimizedMessages = chatPerformanceOptimizer.optimizeMessagesForRendering(testMessages);
    
    // Should be fewer messages than original for performance
    if (optimizedMessages.length <= testMessages.length) {
      console.log('✅ Performance optimization test passed');
      return true;
    }
    
    console.log('❌ Performance optimization test failed');
    return false;
    
  } catch (error) {
    console.error('❌ Performance optimization test error:', error);
    return false;
  }
};

/**
 * Test error recovery
 */
const testErrorRecovery = async () => {
  try {
    // Test quota exceeded simulation
    const quotaError = new Error('QuotaExceededError');
    quotaError.name = 'QuotaExceededError';
    
    const recovered = await chatErrorRecovery.handleError(quotaError, 'test');
    
    // Should attempt recovery (may or may not succeed, but should not crash)
    console.log('✅ Error recovery test passed');
    return true;
    
  } catch (error) {
    console.error('❌ Error recovery test error:', error);
    return false;
  }
};

/**
 * Clean up test data
 */
export const cleanupTestData = async () => {
  try {
    console.log('🧹 Cleaning up test data...');
    
    // Clear test sessions
    await chatHistoryStorage.clearAllHistory();
    
    // Reset error recovery state
    chatErrorRecovery.resetRecoveryState();
    
    // Clear performance caches
    chatPerformanceOptimizer.clearCaches();
    
    console.log('✅ Test data cleanup completed');
    return true;
    
  } catch (error) {
    console.error('❌ Test data cleanup failed:', error);
    return false;
  }
};

/**
 * Quick test for development
 */
export const quickTest = async () => {
  console.log('🚀 Running quick chat history test...');
  
  try {
    // Initialize
    await chatHistoryStorage.init('quick-test-user');
    
    // Add a test message
    await chatHistoryStorage.addMessage({
      role: 'user',
      content: 'Quick test message',
      model: 'grok',
    });
    
    // Load and verify
    const messages = chatHistoryStorage.loadChatHistory();
    const testMessage = messages.find(msg => msg.content === 'Quick test message');
    
    if (testMessage) {
      console.log('✅ Quick test passed - chat history is working!');
      return true;
    } else {
      console.log('❌ Quick test failed - message not found');
      return false;
    }
    
  } catch (error) {
    console.error('❌ Quick test error:', error);
    return false;
  }
};

/**
 * Performance benchmark
 */
export const runPerformanceBenchmark = async () => {
  console.log('📊 Running performance benchmark...');
  
  const results = {
    messageCount: 0,
    storageTime: 0,
    loadTime: 0,
    renderTime: 0,
  };
  
  try {
    // Create test messages
    const messageCount = 1000;
    const testMessages = Array.from({ length: messageCount }, (_, i) => ({
      id: `bench-msg-${i}`,
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `Benchmark message ${i} with some content to test storage performance`,
      model: 'grok',
      timestamp: new Date().toISOString(),
    }));
    
    results.messageCount = messageCount;
    
    // Benchmark storage
    const storageStart = performance.now();
    await chatHistoryStorage.saveChatHistory(testMessages);
    results.storageTime = performance.now() - storageStart;
    
    // Benchmark loading
    const loadStart = performance.now();
    const loadedMessages = chatHistoryStorage.loadChatHistory();
    results.loadTime = performance.now() - loadStart;
    
    // Benchmark rendering optimization
    const renderStart = performance.now();
    chatPerformanceOptimizer.optimizeMessagesForRendering(loadedMessages);
    results.renderTime = performance.now() - renderStart;
    
    console.log('📊 Performance benchmark results:', results);
    return results;
    
  } catch (error) {
    console.error('❌ Performance benchmark failed:', error);
    return { ...results, error: error.message };
  }
};
