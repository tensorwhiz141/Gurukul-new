/**
 * useDedicatedChatbotTTS Hook
 * React hook for TTS integration using the dedicated chatbot service
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import dedicatedChatbotTTSService from '../services/dedicatedChatbotTTSService';

/**
 * Custom hook for Dedicated Chatbot TTS functionality
 * @param {Object} options - Hook configuration options
 * @returns {Object} TTS hook interface
 */
export function useDedicatedChatbotTTS(options = {}) {
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
        const healthy = await dedicatedChatbotTTSService.checkServiceHealth();
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
    
    // Periodic health check
    const healthInterval = setInterval(checkHealth, 30000);
    
    return () => {
      clearInterval(healthInterval);
      mountedRef.current = false;
    };
  }, []);

  /**
   * Generate TTS audio
   * @param {string} text - Text to convert
   * @param {Object} generateOptions - Generation options
   */
  const generateTTS = useCallback(async (text, generateOptions = {}) => {
    if (!text || !text.trim()) {
      throw new Error('Text is required for TTS generation');
    }

    setIsGenerating(true);
    setError(null);

    try {
      const result = await dedicatedChatbotTTSService.generateTTS(text, generateOptions);
      
      if (mountedRef.current) {
        setAudioData(result);
        setIsGenerating(false);
      }
      
      return result;
    } catch (err) {
      if (mountedRef.current) {
        setError(err.message);
        setIsGenerating(false);
        if (onError) {
          onError(err);
        }
      }
      throw err;
    }
  }, [onError]);

  /**
   * Play TTS audio
   * @param {string} text - Text to convert and play
   * @param {Object} playOptions - Playback options
   */
  const playTTS = useCallback(async (text, playOptions = {}) => {
    if (!text || !text.trim()) {
      return;
    }

    setError(null);
    
    try {
      setIsPlaying(true);
      
      await dedicatedChatbotTTSService.playTTS(text, {
        volume,
        onPlayStart: (audioUrl) => {
          if (mountedRef.current) {
            setIsPlaying(true);
            if (onPlayStart) {
              onPlayStart(text);
            }
          }
        },
        onPlayEnd: (audioUrl) => {
          if (mountedRef.current) {
            setIsPlaying(false);
            if (onPlayEnd) {
              onPlayEnd(text);
            }
          }
        },
        onError: (err) => {
          if (mountedRef.current) {
            setIsPlaying(false);
            setError(err.message);
            if (onError) {
              onError(err);
            }
          }
        },
        ...playOptions
      });
      
    } catch (err) {
      if (mountedRef.current) {
        setIsPlaying(false);
        setError(err.message);
        if (onError) {
          onError(err);
        }
      }
      throw err;
    }
  }, [volume, onPlayStart, onPlayEnd, onError]);

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
      await dedicatedChatbotTTSService.autoPlayAI(text, {
        volume,
        onPlayStart: (audioUrl) => {
          if (mountedRef.current) {
            setIsPlaying(true);
            if (onPlayStart) {
              onPlayStart(text);
            }
          }
        },
        onPlayEnd: (audioUrl) => {
          if (mountedRef.current) {
            setIsPlaying(false);
            if (onPlayEnd) {
              onPlayEnd(text);
            }
          }
        },
        onError: (err) => {
          if (mountedRef.current) {
            setIsPlaying(false);
            if (onError) {
              onError(err);
            }
          }
        },
        ...autoPlayOptions
      });
    } catch (err) {
      // Auto-play errors are handled silently to avoid disrupting UX
      console.warn('Dedicated Chatbot TTS auto-play failed:', err.message);

      // If autoplay failed due to browser policy, show a subtle notification
      if (err.message.includes('user interaction required')) {
        console.info('🔊 Dedicated Chatbot TTS: Autoplay requires user interaction');
      }
    }
  }, [autoPlay, serviceHealthy, volume, onPlayStart, onPlayEnd, onError]);

  /**
   * Stop current TTS playback
   */
  const stopTTS = useCallback(() => {
    dedicatedChatbotTTSService.stopCurrentAudio();
    if (mountedRef.current) {
      setIsPlaying(false);
    }
  }, []);

  /**
   * Clear TTS cache
   */
  const clearCache = useCallback(() => {
    dedicatedChatbotTTSService.clearCache();
  }, []);

  /**
   * Get TTS service status
   */
  const getStatus = useCallback(() => {
    return dedicatedChatbotTTSService.getStatus();
  }, []);

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
    autoPlayAI,
    stopTTS,
    clearCache,
    getStatus,
    
    // Service reference for advanced usage
    service: dedicatedChatbotTTSService
  };
}

/**
 * Hook for auto-playing AI responses using dedicated chatbot service
 * @param {string} text - AI response text
 * @param {Object} options - Auto-play options
 */
export function useAutoPlayDedicatedChatbotTTS(text, options = {}) {
  const { autoPlayAI, serviceHealthy } = useDedicatedChatbotTTS({ autoPlay: true, ...options });
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
 * Hook for TTS with visual feedback integration
 * @param {Object} options - TTS options with visual feedback
 */
export function useDedicatedChatbotTTSWithVisuals(options = {}) {
  const ttsHook = useDedicatedChatbotTTS({
    autoPlay: true,
    volume: 0.8,
    ...options
  });

  const [isVisuallyActive, setIsVisuallyActive] = useState(false);

  // Enhanced auto-play with visual feedback
  const autoPlayWithVisuals = useCallback(async (text, visualOptions = {}) => {
    if (!text || !text.trim()) {
      return;
    }

    try {
      setIsVisuallyActive(true);
      
      await ttsHook.autoPlayAI(text, {
        onPlayStart: (playedText) => {
          setIsVisuallyActive(true);
          if (options.onPlayStart) {
            options.onPlayStart(playedText);
          }
        },
        onPlayEnd: (playedText) => {
          setIsVisuallyActive(false);
          if (options.onPlayEnd) {
            options.onPlayEnd(playedText);
          }
        },
        onError: (error) => {
          setIsVisuallyActive(false);
          if (options.onError) {
            options.onError(error);
          }
        },
        ...visualOptions
      });
      
    } catch (error) {
      setIsVisuallyActive(false);
      console.warn('TTS with visuals failed:', error);
    }
  }, [ttsHook, options]);

  return {
    ...ttsHook,
    isVisuallyActive,
    autoPlayWithVisuals
  };
}
