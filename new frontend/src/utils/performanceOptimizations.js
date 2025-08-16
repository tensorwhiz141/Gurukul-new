/**
 * Performance Optimization Utilities
 * Comprehensive performance improvements for AvatarSelection.jsx
 */

// 3D Model optimization utilities
export const optimizeGLTFModel = (gltf) => {
  if (!gltf || !gltf.scene) return gltf;
  
  // Traverse and optimize the scene
  gltf.scene.traverse((child) => {
    if (child.isMesh) {
      // Enable frustum culling
      child.frustumCulled = true;
      
      // Optimize materials
      if (child.material) {
        // Enable material caching
        child.material.needsUpdate = false;
        
        // Optimize texture settings
        if (child.material.map) {
          child.material.map.generateMipmaps = true;
          child.material.map.minFilter = THREE.LinearMipmapLinearFilter;
          child.material.map.magFilter = THREE.LinearFilter;
        }
      }
      
      // Optimize geometry
      if (child.geometry) {
        // Merge vertices if possible
        child.geometry.mergeVertices?.();
        
        // Compute bounding sphere for better culling
        child.geometry.computeBoundingSphere();
      }
    }
  });
  
  return gltf;
};

// Memory management utilities
export const disposeGLTFModel = (gltf) => {
  if (!gltf || !gltf.scene) return;
  
  gltf.scene.traverse((child) => {
    if (child.isMesh) {
      // Dispose geometry
      if (child.geometry) {
        child.geometry.dispose();
      }
      
      // Dispose materials
      if (child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach(material => {
            disposeMaterial(material);
          });
        } else {
          disposeMaterial(child.material);
        }
      }
    }
  });
};

const disposeMaterial = (material) => {
  // Dispose textures
  Object.keys(material).forEach(key => {
    const value = material[key];
    if (value && typeof value.dispose === 'function') {
      value.dispose();
    }
  });
  
  // Dispose material
  material.dispose();
};

// Component optimization utilities
export const shouldComponentUpdate = (prevProps, nextProps, keys) => {
  return keys.some(key => prevProps[key] !== nextProps[key]);
};

export const memoizeProps = (props, dependencies) => {
  return useMemo(() => props, dependencies);
};

// Loading state management
export class LoadingStateManager {
  constructor() {
    this.states = new Map();
    this.listeners = new Set();
  }
  
  setLoading(key, isLoading) {
    this.states.set(key, isLoading);
    this.notifyListeners();
  }
  
  isLoading(key) {
    return this.states.get(key) || false;
  }
  
  isAnyLoading() {
    return Array.from(this.states.values()).some(loading => loading);
  }
  
  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  
  notifyListeners() {
    this.listeners.forEach(listener => listener(this.states));
  }
  
  clear() {
    this.states.clear();
    this.notifyListeners();
  }
}

// Global loading state manager instance
export const globalLoadingManager = new LoadingStateManager();

// Performance metrics collection
export class PerformanceMetrics {
  constructor() {
    this.metrics = {
      renderTimes: [],
      modelLoadTimes: [],
      memoryUsage: [],
      componentUpdates: new Map()
    };
  }
  
  recordRenderTime(componentName, time) {
    this.metrics.renderTimes.push({
      component: componentName,
      time,
      timestamp: Date.now()
    });
    
    // Keep only last 100 entries
    if (this.metrics.renderTimes.length > 100) {
      this.metrics.renderTimes.shift();
    }
  }
  
  recordModelLoadTime(modelPath, time, success) {
    this.metrics.modelLoadTimes.push({
      path: modelPath,
      time,
      success,
      timestamp: Date.now()
    });
    
    // Keep only last 50 entries
    if (this.metrics.modelLoadTimes.length > 50) {
      this.metrics.modelLoadTimes.shift();
    }
  }
  
  recordMemoryUsage() {
    if (performance.memory) {
      this.metrics.memoryUsage.push({
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit,
        timestamp: Date.now()
      });
      
      // Keep only last 20 entries
      if (this.metrics.memoryUsage.length > 20) {
        this.metrics.memoryUsage.shift();
      }
    }
  }
  
  recordComponentUpdate(componentName) {
    const current = this.metrics.componentUpdates.get(componentName) || 0;
    this.metrics.componentUpdates.set(componentName, current + 1);
  }
  
  getAverageRenderTime(componentName) {
    const times = this.metrics.renderTimes
      .filter(entry => entry.component === componentName)
      .map(entry => entry.time);
    
    return times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
  }
  
  getAverageModelLoadTime() {
    const times = this.metrics.modelLoadTimes
      .filter(entry => entry.success)
      .map(entry => entry.time);
    
    return times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
  }
  
  getMemoryTrend() {
    if (this.metrics.memoryUsage.length < 2) return 0;
    
    const recent = this.metrics.memoryUsage.slice(-5);
    const first = recent[0].used;
    const last = recent[recent.length - 1].used;
    
    return last - first;
  }
  
  getSummary() {
    return {
      averageRenderTimes: Array.from(this.metrics.componentUpdates.keys()).reduce((acc, component) => {
        acc[component] = this.getAverageRenderTime(component);
        return acc;
      }, {}),
      averageModelLoadTime: this.getAverageModelLoadTime(),
      memoryTrend: this.getMemoryTrend(),
      componentUpdateCounts: Object.fromEntries(this.metrics.componentUpdates),
      totalMetrics: {
        renders: this.metrics.renderTimes.length,
        modelLoads: this.metrics.modelLoadTimes.length,
        memorySnapshots: this.metrics.memoryUsage.length
      }
    };
  }
  
  clear() {
    this.metrics = {
      renderTimes: [],
      modelLoadTimes: [],
      memoryUsage: [],
      componentUpdates: new Map()
    };
  }
}

// Global performance metrics instance
export const globalPerformanceMetrics = new PerformanceMetrics();

// Optimization recommendations
export const getOptimizationRecommendations = () => {
  const metrics = globalPerformanceMetrics.getSummary();
  const recommendations = [];
  
  // Check render performance
  Object.entries(metrics.averageRenderTimes).forEach(([component, avgTime]) => {
    if (avgTime > 16) { // 60fps threshold
      recommendations.push({
        type: 'performance',
        severity: avgTime > 33 ? 'high' : 'medium',
        component,
        issue: `Slow rendering detected (${avgTime.toFixed(2)}ms average)`,
        suggestion: 'Consider memoization, reducing re-renders, or optimizing heavy computations'
      });
    }
  });
  
  // Check model load performance
  if (metrics.averageModelLoadTime > 2000) {
    recommendations.push({
      type: 'loading',
      severity: 'medium',
      issue: `Slow model loading (${(metrics.averageModelLoadTime / 1000).toFixed(1)}s average)`,
      suggestion: 'Consider model optimization, compression, or progressive loading'
    });
  }
  
  // Check memory usage
  if (metrics.memoryTrend > 10 * 1024 * 1024) { // 10MB increase
    recommendations.push({
      type: 'memory',
      severity: 'high',
      issue: `Memory usage increasing (+${(metrics.memoryTrend / 1024 / 1024).toFixed(1)}MB)`,
      suggestion: 'Check for memory leaks, dispose unused resources, or implement cleanup'
    });
  }
  
  // Check component update frequency
  Object.entries(metrics.componentUpdateCounts).forEach(([component, count]) => {
    if (count > 100) {
      recommendations.push({
        type: 'updates',
        severity: 'medium',
        component,
        issue: `Frequent component updates (${count} times)`,
        suggestion: 'Consider memoization or reducing prop changes'
      });
    }
  });
  
  return recommendations;
};

// Development performance monitor
export const startPerformanceMonitoring = () => {
  if (process.env.NODE_ENV !== 'development') return;
  
  // Monitor memory usage every 10 seconds
  const memoryInterval = setInterval(() => {
    globalPerformanceMetrics.recordMemoryUsage();
  }, 10000);
  
  // Log performance summary every 30 seconds
  const summaryInterval = setInterval(() => {
    const summary = globalPerformanceMetrics.getSummary();
    const recommendations = getOptimizationRecommendations();
    
    console.group('📊 Performance Summary');
    console.table(summary.averageRenderTimes);
    console.log('Average model load time:', (summary.averageModelLoadTime / 1000).toFixed(2) + 's');
    console.log('Memory trend:', (summary.memoryTrend / 1024 / 1024).toFixed(2) + 'MB');
    
    if (recommendations.length > 0) {
      console.warn('⚠️ Performance Recommendations:');
      recommendations.forEach(rec => {
        console.warn(`${rec.severity.toUpperCase()}: ${rec.issue} - ${rec.suggestion}`);
      });
    }
    console.groupEnd();
  }, 30000);
  
  // Cleanup function
  return () => {
    clearInterval(memoryInterval);
    clearInterval(summaryInterval);
  };
};
