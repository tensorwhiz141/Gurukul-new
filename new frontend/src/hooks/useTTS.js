/**
 * useTTS Hook
 * React hook for easy TTS integration in components
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import ttsService from '../services/ttsService';

/**
 * Custom hook for TTS functionality
 * @param {Object} options - Hook configuration options
 * @returns {Object} TTS hook interface
 */
export function useTTS(options = {}) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState(null);
  const [audioData, setAudioData] = useState(null);
  const [serviceHealthy, setServiceHealthy] = useState(null);
  
  const mountedRef = useRef(true);
  const lastTextRef = useRef('');

  // Configuration
  const {
    autoPlay = false,
    enableQueue = false,
    volume = 0.8,
    onPlayStart,
    onPlayEnd,
    onError,
    debounceMs = 300
  } = options;

  // Check service health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const healthy = await ttsService.checkServiceHealth();
        if (mountedRef.current) {
          setServiceHealthy(healthy);
        }
      } catch (err) {
        if (mountedRef.current) {
          setServiceHealthy(false);
        }
      }
    };

    checkHealth();

    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Configure TTS service
  useEffect(() => {
    ttsService.configure({
      autoPlayEnabled: autoPlay,
      volume: volume
    });
  }, [autoPlay, volume]);

  /**
   * Generate TTS audio from text
   * @param {string} text - Text to convert
   * @param {Object} generateOptions - Generation options
   * @returns {Promise<Object>} Audio data
   */
  const generateTTS = useCallback(async (text, generateOptions = {}) => {
    if (!text || !text.trim()) {
      const error = new Error('Text is required for TTS generation');
      setError(error);
      onError?.(error);
      return null;
    }

    // Avoid duplicate generations
    if (text === lastTextRef.current && isGenerating) {
      return audioData;
    }

    setIsGenerating(true);
    setError(null);
    lastTextRef.current = text;

    try {
      const result = await ttsService.generateTTS(text, generateOptions);
      
      if (mountedRef.current) {
        setAudioData(result);
        setIsGenerating(false);
        return result;
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err);
        setIsGenerating(false);
        onError?.(err);
      }
      return null;
    }
  }, [isGenerating, audioData, onError]);

  /**
   * Play TTS audio
   * @param {string} text - Text to play
   * @param {Object} playOptions - Play options
   */
  const playTTS = useCallback(async (text, playOptions = {}) => {
    if (!serviceHealthy) {
      const error = new Error('TTS service is not available');
      setError(error);
      onError?.(error);
      return;
    }

    setIsPlaying(true);
    setError(null);
    onPlayStart?.(text);

    try {
      await ttsService.playTTS(text, playOptions);
      
      if (mountedRef.current) {
        setIsPlaying(false);
        onPlayEnd?.(text);
      }
    } catch (err) {
      if (mountedRef.current) {
        setIsPlaying(false);
        setError(err);
        onError?.(err);
      }
    }
  }, [serviceHealthy, onPlayStart, onPlayEnd, onError]);

  /**
   * Queue TTS for sequential playback
   * @param {string} text - Text to queue
   * @param {Object} queueOptions - Queue options
   */
  const queueTTS = useCallback(async (text, queueOptions = {}) => {
    if (!enableQueue) {
      return playTTS(text, queueOptions);
    }

    try {
      await ttsService.queueTTS(text, queueOptions);
    } catch (err) {
      setError(err);
      onError?.(err);
    }
  }, [enableQueue, playTTS, onError]);

  /**
   * Auto-play TTS for AI content
   * @param {string} text - AI-generated text
   * @param {Object} autoPlayOptions - Auto-play options
   */
  const autoPlayAI = useCallback(async (text, autoPlayOptions = {}) => {
    if (!autoPlay || !serviceHealthy) {
      return;
    }

    try {
      await ttsService.autoPlayAI(text, autoPlayOptions);
    } catch (err) {
      // Auto-play errors are handled silently to avoid disrupting UX
      console.warn('TTS auto-play failed:', err.message);

      // If autoplay failed due to browser policy, show a subtle notification
      if (err.message.includes('user interaction required')) {
        console.info('💡 TTS: Click anywhere on the page to enable auto-play for future AI responses');
      }
    }
  }, [autoPlay, serviceHealthy]);

  /**
   * Stop current TTS playback
   */
  const stopTTS = useCallback(() => {
    ttsService.stopCurrentAudio();
    setIsPlaying(false);
  }, []);

  /**
   * Clear TTS cache
   */
  const clearCache = useCallback(() => {
    ttsService.clearCache();
  }, []);

  /**
   * Get TTS service status
   */
  const getStatus = useCallback(() => {
    return ttsService.getStatus();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      if (isPlaying) {
        ttsService.stopCurrentAudio();
      }
    };
  }, [isPlaying]);

  return {
    // State
    isGenerating,
    isPlaying,
    error,
    audioData,
    serviceHealthy,
    
    // Actions
    generateTTS,
    playTTS,
    queueTTS,
    autoPlayAI,
    stopTTS,
    clearCache,
    getStatus,
    
    // Service reference for advanced usage
    ttsService
  };
}

/**
 * Hook for auto-playing AI responses
 * @param {string} text - AI response text
 * @param {Object} options - Auto-play options
 */
export function useAutoPlayTTS(text, options = {}) {
  const { autoPlayAI, serviceHealthy } = useTTS({ autoPlay: true, ...options });
  const processedTextRef = useRef('');

  useEffect(() => {
    // Only auto-play new text
    if (text && text !== processedTextRef.current && serviceHealthy) {
      processedTextRef.current = text;
      autoPlayAI(text, options);
    }
  }, [text, autoPlayAI, serviceHealthy, options]);

  return { serviceHealthy };
}

/**
 * Hook for TTS with Jupiter model integration
 * @param {Object} options - Jupiter TTS options
 */
export function useJupiterTTS(options = {}) {
  const ttsHook = useTTS({
    autoPlay: true,
    enableQueue: true,
    volume: 0.9,
    ...options
  });

  /**
   * Handle Jupiter model response with TTS
   * @param {string} response - Jupiter model response
   * @param {Object} jupiterOptions - Jupiter-specific options
   */
  const handleJupiterResponse = useCallback(async (response, jupiterOptions = {}) => {
    if (!response || !response.trim()) {
      return;
    }

    // Add Jupiter-specific formatting
    const formattedResponse = response.replace(/\n\n/g, '. ').replace(/\n/g, ' ');
    
    await ttsHook.autoPlayAI(formattedResponse, {
      delay: 800, // Slightly longer delay for Jupiter responses
      ...jupiterOptions
    });
  }, [ttsHook]);

  return {
    ...ttsHook,
    handleJupiterResponse
  };
}

export default useTTS;
