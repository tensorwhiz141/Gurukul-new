import { useEffect, useRef, useCallback } from 'react';

/**
 * Performance monitoring hook
 * Tracks component render times, memory usage, and provides optimization insights
 */
export const usePerformanceMonitor = (componentName, options = {}) => {
  const {
    enableLogging = process.env.NODE_ENV === 'development',
    logThreshold = 16, // Log renders that take longer than 16ms
    memoryThreshold = 50 * 1024 * 1024, // 50MB memory threshold
    trackMemory = true,
    trackRenders = true
  } = options;

  const renderStartTime = useRef(null);
  const renderCount = useRef(0);
  const totalRenderTime = useRef(0);
  const lastMemoryUsage = useRef(0);

  // Start performance measurement
  const startMeasurement = useCallback(() => {
    if (!trackRenders) return;
    renderStartTime.current = performance.now();
  }, [trackRenders]);

  // End performance measurement
  const endMeasurement = useCallback(() => {
    if (!trackRenders || !renderStartTime.current) return;
    
    const renderTime = performance.now() - renderStartTime.current;
    renderCount.current += 1;
    totalRenderTime.current += renderTime;
    
    if (enableLogging && renderTime > logThreshold) {
      console.warn(
        `🐌 Slow render detected in ${componentName}: ${renderTime.toFixed(2)}ms`
      );
    }
    
    renderStartTime.current = null;
  }, [trackRenders, enableLogging, logThreshold, componentName]);

  // Memory usage tracking
  const checkMemoryUsage = useCallback(() => {
    if (!trackMemory || !performance.memory) return;
    
    const currentMemory = performance.memory.usedJSHeapSize;
    const memoryDiff = currentMemory - lastMemoryUsage.current;
    
    if (enableLogging && currentMemory > memoryThreshold) {
      console.warn(
        `🧠 High memory usage in ${componentName}: ${(currentMemory / 1024 / 1024).toFixed(2)}MB`
      );
    }
    
    if (enableLogging && memoryDiff > 10 * 1024 * 1024) { // 10MB increase
      console.warn(
        `📈 Memory increase in ${componentName}: +${(memoryDiff / 1024 / 1024).toFixed(2)}MB`
      );
    }
    
    lastMemoryUsage.current = currentMemory;
  }, [trackMemory, enableLogging, memoryThreshold, componentName]);

  // Get performance stats
  const getStats = useCallback(() => {
    return {
      renderCount: renderCount.current,
      totalRenderTime: totalRenderTime.current,
      averageRenderTime: renderCount.current > 0 ? totalRenderTime.current / renderCount.current : 0,
      currentMemory: performance.memory ? performance.memory.usedJSHeapSize : 0
    };
  }, []);

  // Log performance summary
  const logSummary = useCallback(() => {
    if (!enableLogging) return;
    
    const stats = getStats();
    console.group(`📊 Performance Summary: ${componentName}`);
    console.log(`Renders: ${stats.renderCount}`);
    console.log(`Total render time: ${stats.totalRenderTime.toFixed(2)}ms`);
    console.log(`Average render time: ${stats.averageRenderTime.toFixed(2)}ms`);
    if (performance.memory) {
      console.log(`Current memory: ${(stats.currentMemory / 1024 / 1024).toFixed(2)}MB`);
    }
    console.groupEnd();
  }, [enableLogging, componentName, getStats]);

  // Start measurement on each render
  useEffect(() => {
    startMeasurement();
    return endMeasurement;
  });

  // Check memory usage periodically
  useEffect(() => {
    if (!trackMemory) return;
    
    checkMemoryUsage();
    const interval = setInterval(checkMemoryUsage, 5000); // Check every 5 seconds
    
    return () => clearInterval(interval);
  }, [checkMemoryUsage, trackMemory]);

  // Cleanup and log summary on unmount
  useEffect(() => {
    return () => {
      if (enableLogging && renderCount.current > 0) {
        logSummary();
      }
    };
  }, [enableLogging, logSummary]);

  return {
    getStats,
    logSummary,
    startMeasurement,
    endMeasurement
  };
};

/**
 * Hook for tracking 3D model loading performance
 */
export const useModelLoadingPerformance = (modelPath) => {
  const loadStartTime = useRef(null);
  const loadStats = useRef({
    totalLoads: 0,
    successfulLoads: 0,
    failedLoads: 0,
    totalLoadTime: 0,
    averageLoadTime: 0
  });

  const startLoading = useCallback(() => {
    loadStartTime.current = performance.now();
  }, []);

  const endLoading = useCallback((success = true) => {
    if (!loadStartTime.current) return;
    
    const loadTime = performance.now() - loadStartTime.current;
    const stats = loadStats.current;
    
    stats.totalLoads += 1;
    stats.totalLoadTime += loadTime;
    stats.averageLoadTime = stats.totalLoadTime / stats.totalLoads;
    
    if (success) {
      stats.successfulLoads += 1;
    } else {
      stats.failedLoads += 1;
    }
    
    if (process.env.NODE_ENV === 'development') {
      console.log(
        `🎭 Model loaded: ${modelPath} in ${loadTime.toFixed(2)}ms (${success ? 'success' : 'failed'})`
      );
    }
    
    loadStartTime.current = null;
  }, [modelPath]);

  const getLoadingStats = useCallback(() => {
    return { ...loadStats.current };
  }, []);

  return {
    startLoading,
    endLoading,
    getLoadingStats
  };
};

/**
 * Hook for tracking Redux state update performance
 */
export const useReduxPerformance = (selectorName, selectorValue) => {
  const updateCount = useRef(0);
  const lastUpdateTime = useRef(Date.now());
  const updateFrequency = useRef(0);

  useEffect(() => {
    const now = Date.now();
    const timeSinceLastUpdate = now - lastUpdateTime.current;
    
    updateCount.current += 1;
    updateFrequency.current = updateCount.current / ((now - lastUpdateTime.current) / 1000);
    lastUpdateTime.current = now;
    
    if (process.env.NODE_ENV === 'development' && timeSinceLastUpdate < 100) {
      console.warn(
        `⚡ Frequent Redux updates detected for ${selectorName}: ${updateCount.current} updates, ${updateFrequency.current.toFixed(2)} updates/sec`
      );
    }
  }, [selectorValue, selectorName]);

  return {
    updateCount: updateCount.current,
    updateFrequency: updateFrequency.current
  };
};

/**
 * Hook for debouncing expensive operations
 */
export const useDebounce = (value, delay) => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

/**
 * Hook for throttling expensive operations
 */
export const useThrottle = (value, limit) => {
  const [throttledValue, setThrottledValue] = useState(value);
  const lastRan = useRef(Date.now());

  useEffect(() => {
    const handler = setTimeout(() => {
      if (Date.now() - lastRan.current >= limit) {
        setThrottledValue(value);
        lastRan.current = Date.now();
      }
    }, limit - (Date.now() - lastRan.current));

    return () => {
      clearTimeout(handler);
    };
  }, [value, limit]);

  return throttledValue;
};
