import React, { useState, useEffect, useRef } from 'react';
import { globalPerformanceMetrics, getOptimizationRecommendations } from '../utils/performanceOptimizations';

/**
 * Performance Test Component
 * Displays real-time performance metrics and optimization recommendations
 */
export default function PerformanceTest({ isVisible = false }) {
  const [metrics, setMetrics] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [isExpanded, setIsExpanded] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (isVisible) {
      // Update metrics every 2 seconds
      intervalRef.current = setInterval(() => {
        const summary = globalPerformanceMetrics.getSummary();
        const recs = getOptimizationRecommendations();
        setMetrics(summary);
        setRecommendations(recs);
      }, 2000);

      // Initial load
      const summary = globalPerformanceMetrics.getSummary();
      const recs = getOptimizationRecommendations();
      setMetrics(summary);
      setRecommendations(recs);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isVisible]);

  if (!isVisible || !metrics) return null;

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'text-red-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-green-400';
      default: return 'text-white';
    }
  };

  const getSeverityBg = (severity) => {
    switch (severity) {
      case 'high': return 'bg-red-500/20 border-red-500/30';
      case 'medium': return 'bg-yellow-500/20 border-yellow-500/30';
      case 'low': return 'bg-green-500/20 border-green-500/30';
      default: return 'bg-white/5 border-white/10';
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Toggle Button */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`mb-2 px-3 py-2 rounded-lg border transition-all ${
          recommendations.length > 0 
            ? 'bg-yellow-500/20 border-yellow-500/30 text-yellow-400' 
            : 'bg-green-500/20 border-green-500/30 text-green-400'
        }`}
      >
        📊 Performance {recommendations.length > 0 && `(${recommendations.length} issues)`}
      </button>

      {/* Performance Panel */}
      {isExpanded && (
        <div className="bg-black/80 backdrop-blur-sm border border-white/20 rounded-lg p-4 w-80 max-h-96 overflow-y-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-white font-semibold">Performance Metrics</h3>
            <button
              onClick={() => globalPerformanceMetrics.clear()}
              className="text-xs text-white/60 hover:text-white"
            >
              Clear
            </button>
          </div>

          {/* Render Performance */}
          <div className="mb-4">
            <h4 className="text-white/80 text-sm font-medium mb-2">Render Times (avg)</h4>
            <div className="space-y-1">
              {Object.entries(metrics.averageRenderTimes).map(([component, time]) => (
                <div key={component} className="flex justify-between text-xs">
                  <span className="text-white/70">{component}</span>
                  <span className={time > 16 ? 'text-red-400' : 'text-green-400'}>
                    {time.toFixed(1)}ms
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Model Loading */}
          <div className="mb-4">
            <h4 className="text-white/80 text-sm font-medium mb-2">Model Loading</h4>
            <div className="flex justify-between text-xs">
              <span className="text-white/70">Average Load Time</span>
              <span className={metrics.averageModelLoadTime > 2000 ? 'text-yellow-400' : 'text-green-400'}>
                {(metrics.averageModelLoadTime / 1000).toFixed(1)}s
              </span>
            </div>
          </div>

          {/* Memory Usage */}
          <div className="mb-4">
            <h4 className="text-white/80 text-sm font-medium mb-2">Memory</h4>
            <div className="flex justify-between text-xs">
              <span className="text-white/70">Trend</span>
              <span className={metrics.memoryTrend > 0 ? 'text-red-400' : 'text-green-400'}>
                {metrics.memoryTrend > 0 ? '+' : ''}{(metrics.memoryTrend / 1024 / 1024).toFixed(1)}MB
              </span>
            </div>
          </div>

          {/* Component Updates */}
          <div className="mb-4">
            <h4 className="text-white/80 text-sm font-medium mb-2">Component Updates</h4>
            <div className="space-y-1">
              {Object.entries(metrics.componentUpdateCounts).map(([component, count]) => (
                <div key={component} className="flex justify-between text-xs">
                  <span className="text-white/70">{component}</span>
                  <span className={count > 50 ? 'text-yellow-400' : 'text-green-400'}>
                    {count}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Recommendations */}
          {recommendations.length > 0 && (
            <div>
              <h4 className="text-white/80 text-sm font-medium mb-2">Recommendations</h4>
              <div className="space-y-2">
                {recommendations.map((rec, index) => (
                  <div
                    key={index}
                    className={`p-2 rounded border text-xs ${getSeverityBg(rec.severity)}`}
                  >
                    <div className={`font-medium ${getSeverityColor(rec.severity)}`}>
                      {rec.severity.toUpperCase()}: {rec.component && `${rec.component} - `}{rec.issue}
                    </div>
                    <div className="text-white/70 mt-1">{rec.suggestion}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Summary Stats */}
          <div className="mt-4 pt-4 border-t border-white/10">
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="text-center">
                <div className="text-white/50">Renders</div>
                <div className="text-white">{metrics.totalMetrics.renders}</div>
              </div>
              <div className="text-center">
                <div className="text-white/50">Models</div>
                <div className="text-white">{metrics.totalMetrics.modelLoads}</div>
              </div>
              <div className="text-center">
                <div className="text-white/50">Memory</div>
                <div className="text-white">{metrics.totalMetrics.memorySnapshots}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Performance Test Hook
 * Provides easy access to performance testing in development
 */
export function usePerformanceTest() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const handleKeyPress = (event) => {
      // Toggle with Ctrl+Shift+P
      if (event.ctrlKey && event.shiftKey && event.key === 'P') {
        setIsVisible(prev => !prev);
      }
    };

    if (process.env.NODE_ENV === 'development') {
      window.addEventListener('keydown', handleKeyPress);
      return () => window.removeEventListener('keydown', handleKeyPress);
    }
  }, []);

  return { isVisible, setIsVisible };
}

/**
 * Performance Test Provider
 * Wraps the app to provide performance testing capabilities
 */
export function PerformanceTestProvider({ children }) {
  const { isVisible } = usePerformanceTest();

  return (
    <>
      {children}
      {process.env.NODE_ENV === 'development' && (
        <PerformanceTest isVisible={isVisible} />
      )}
    </>
  );
}
