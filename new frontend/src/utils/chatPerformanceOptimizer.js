/**
 * Chat Performance Optimizer
 * Optimizes chat rendering and storage for large chat histories
 */

// Performance thresholds
const PERFORMANCE_THRESHOLDS = {
  LARGE_HISTORY: 100, // Messages count that's considered large
  HUGE_HISTORY: 500, // Messages count that requires aggressive optimization
  MAX_RENDERED_MESSAGES: 50, // Maximum messages to render at once
  VIRTUALIZATION_THRESHOLD: 100, // When to enable virtualization
  DEBOUNCE_DELAY: 300, // Debounce delay for auto-save
};

class ChatPerformanceOptimizer {
  constructor() {
    this.messageCache = new Map();
    this.renderCache = new Map();
    this.lastRenderTime = 0;
    this.isOptimizationEnabled = true;
  }

  /**
   * Optimize message list for rendering
   */
  optimizeMessagesForRendering(messages, currentIndex = 0) {
    if (!this.isOptimizationEnabled || messages.length <= PERFORMANCE_THRESHOLDS.MAX_RENDERED_MESSAGES) {
      return messages;
    }

    // For large histories, only render recent messages
    const startIndex = Math.max(0, messages.length - PERFORMANCE_THRESHOLDS.MAX_RENDERED_MESSAGES);
    const optimizedMessages = messages.slice(startIndex);

    // Add a placeholder for older messages
    if (startIndex > 0) {
      const olderMessagesPlaceholder = {
        id: 'older-messages-placeholder',
        role: 'system',
        content: `... ${startIndex} older messages (click to load more)`,
        isPlaceholder: true,
        timestamp: messages[0]?.timestamp || new Date().toISOString(),
      };
      
      return [olderMessagesPlaceholder, ...optimizedMessages];
    }

    return optimizedMessages;
  }

  /**
   * Debounced save function to prevent excessive storage writes
   */
  createDebouncedSave(saveFunction) {
    let timeoutId = null;
    let pendingData = null;

    return (data) => {
      pendingData = data;
      
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      timeoutId = setTimeout(async () => {
        if (pendingData) {
          try {
            await saveFunction(pendingData);
          } catch (error) {
            console.error('Debounced save failed:', error);
          }
          pendingData = null;
        }
      }, PERFORMANCE_THRESHOLDS.DEBOUNCE_DELAY);
    };
  }

  /**
   * Optimize message content for storage
   */
  optimizeMessageForStorage(message) {
    if (!message) return message;

    // Create a lightweight version for storage
    const optimized = {
      id: message.id,
      role: message.role,
      content: this.optimizeContent(message.content),
      model: message.model,
      timestamp: message.timestamp,
    };

    // Only include optional fields if they exist
    if (message.isWelcome) optimized.isWelcome = true;
    if (message.isError) optimized.isError = true;
    if (message.isPlaceholder) optimized.isPlaceholder = true;

    return optimized;
  }

  /**
   * Optimize content for storage (remove unnecessary whitespace, etc.)
   */
  optimizeContent(content) {
    if (typeof content !== 'string') {
      return content;
    }

    // Trim whitespace and normalize line breaks
    return content.trim().replace(/\n\s*\n\s*\n/g, '\n\n');
  }

  /**
   * Check if performance optimization should be enabled
   */
  shouldOptimize(messageCount) {
    return messageCount >= PERFORMANCE_THRESHOLDS.LARGE_HISTORY;
  }

  /**
   * Get performance recommendations
   */
  getPerformanceRecommendations(messageCount, storageSize) {
    const recommendations = [];

    if (messageCount >= PERFORMANCE_THRESHOLDS.HUGE_HISTORY) {
      recommendations.push({
        type: 'warning',
        message: 'Large chat history detected. Consider exporting and clearing old sessions.',
        action: 'cleanup',
      });
    }

    if (storageSize > 5 * 1024 * 1024) { // 5MB
      recommendations.push({
        type: 'warning',
        message: 'Storage usage is high. Consider enabling compression or cleaning up old data.',
        action: 'compress',
      });
    }

    if (messageCount >= PERFORMANCE_THRESHOLDS.VIRTUALIZATION_THRESHOLD) {
      recommendations.push({
        type: 'info',
        message: 'Message virtualization is active to improve performance.',
        action: 'virtualize',
      });
    }

    return recommendations;
  }

  /**
   * Measure rendering performance
   */
  measureRenderPerformance(messageCount, callback) {
    const startTime = performance.now();
    
    const result = callback();
    
    const endTime = performance.now();
    const renderTime = endTime - startTime;
    
    // Log performance metrics
    if (renderTime > 100) { // More than 100ms is considered slow
      console.warn(`Slow chat render: ${renderTime.toFixed(2)}ms for ${messageCount} messages`);
    }

    this.lastRenderTime = renderTime;
    
    return result;
  }

  /**
   * Create virtualized message list for very large histories
   */
  createVirtualizedMessageList(messages, visibleRange = { start: 0, end: 50 }) {
    if (messages.length <= PERFORMANCE_THRESHOLDS.VIRTUALIZATION_THRESHOLD) {
      return messages;
    }

    const { start, end } = visibleRange;
    const visibleMessages = messages.slice(start, Math.min(end, messages.length));

    // Add placeholders for non-visible messages
    const result = [];

    if (start > 0) {
      result.push({
        id: 'virtual-placeholder-top',
        role: 'system',
        content: `... ${start} messages above`,
        isPlaceholder: true,
        isVirtual: true,
      });
    }

    result.push(...visibleMessages);

    if (end < messages.length) {
      result.push({
        id: 'virtual-placeholder-bottom',
        role: 'system',
        content: `... ${messages.length - end} messages below`,
        isPlaceholder: true,
        isVirtual: true,
      });
    }

    return result;
  }

  /**
   * Optimize storage operations
   */
  optimizeStorageOperation(operation, data) {
    return new Promise((resolve, reject) => {
      // Use requestIdleCallback if available for non-critical operations
      if (window.requestIdleCallback && operation !== 'critical') {
        window.requestIdleCallback(
          async () => {
            try {
              const result = await data();
              resolve(result);
            } catch (error) {
              reject(error);
            }
          },
          { timeout: 5000 }
        );
      } else {
        // Execute immediately for critical operations
        Promise.resolve(data()).then(resolve).catch(reject);
      }
    });
  }

  /**
   * Batch multiple storage operations
   */
  createBatchedStorageOperation() {
    const operations = [];
    let batchTimeout = null;

    return {
      add: (operation) => {
        operations.push(operation);
        
        if (batchTimeout) {
          clearTimeout(batchTimeout);
        }

        batchTimeout = setTimeout(async () => {
          if (operations.length > 0) {
            try {
              // Execute all operations in batch
              await Promise.all(operations.map(op => op()));
              operations.length = 0; // Clear the array
            } catch (error) {
              console.error('Batch storage operation failed:', error);
            }
          }
        }, 100); // 100ms batch window
      },
      
      flush: async () => {
        if (batchTimeout) {
          clearTimeout(batchTimeout);
        }
        
        if (operations.length > 0) {
          try {
            await Promise.all(operations.map(op => op()));
            operations.length = 0;
          } catch (error) {
            console.error('Batch flush failed:', error);
          }
        }
      }
    };
  }

  /**
   * Memory usage monitoring
   */
  getMemoryUsage() {
    if (performance.memory) {
      return {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit,
        percentage: (performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit) * 100,
      };
    }
    
    return null;
  }

  /**
   * Clean up caches
   */
  clearCaches() {
    this.messageCache.clear();
    this.renderCache.clear();
    console.log('🧹 Performance caches cleared');
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics() {
    return {
      lastRenderTime: this.lastRenderTime,
      cacheSize: this.messageCache.size + this.renderCache.size,
      memoryUsage: this.getMemoryUsage(),
      optimizationEnabled: this.isOptimizationEnabled,
      thresholds: PERFORMANCE_THRESHOLDS,
    };
  }

  /**
   * Enable or disable optimization
   */
  setOptimizationEnabled(enabled) {
    this.isOptimizationEnabled = enabled;
    if (!enabled) {
      this.clearCaches();
    }
  }
}

// Create singleton instance
const chatPerformanceOptimizer = new ChatPerformanceOptimizer();

export default chatPerformanceOptimizer;
export { PERFORMANCE_THRESHOLDS };
