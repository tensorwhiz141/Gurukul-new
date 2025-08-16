/**
 * Dedicated Chatbot TTS Service Integration
 * Provides TTS functionality using the dedicated chatbot service with Google TTS
 */

const CHATBOT_TTS_BASE_URL = 'http://localhost:8001';

class DedicatedChatbotTTSService {
  constructor() {
    this.audioCache = new Map();
    this.currentAudio = null;
    this.isPlaying = false;
    this.autoPlayEnabled = true;
    this.volume = 0.8;
    this.playbackQueue = [];
    this.isProcessingQueue = false;
  }

  /**
   * Check if the dedicated chatbot TTS service is healthy
   * @returns {Promise<boolean>} Service health status
   */
  async checkServiceHealth() {
    try {
      const response = await fetch(`${CHATBOT_TTS_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        return data.status === 'healthy';
      }
      return false;
    } catch (error) {
      console.warn('Dedicated Chatbot TTS service health check failed:', error);
      return false;
    }
  }

  /**
   * Generate TTS audio using the dedicated chatbot service
   * @param {string} text - Text to convert to speech
   * @param {Object} options - Additional options
   * @returns {Promise<Object>} TTS generation result
   */
  async generateTTS(text, options = {}) {
    if (!text || !text.trim()) {
      throw new Error('Text is required for TTS generation');
    }

    // Check cache first
    const cacheKey = this.getCacheKey(text, options);
    if (this.audioCache.has(cacheKey)) {
      console.log('🔊 Dedicated Chatbot TTS: Using cached audio for text:', text.substring(0, 50) + '...');
      return this.audioCache.get(cacheKey);
    }

    try {
      console.log('🔊 Dedicated Chatbot TTS: Generating streaming audio for text:', text.substring(0, 100) + '...');

      // Use streaming endpoint
      const response = await fetch(`${CHATBOT_TTS_BASE_URL}/tts/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`TTS streaming failed: ${response.status} - ${errorData.detail || 'Unknown error'}`);
      }

      // Get audio data as blob
      const audioBlob = await response.blob();

      // Create object URL for the blob
      const audioUrl = URL.createObjectURL(audioBlob);

      const result = {
        status: 'success',
        audioUrl: audioUrl,
        fullAudioUrl: audioUrl,
        blob: audioBlob,
        textLength: text.length,
        timestamp: new Date().toISOString(),
        text: text,
        isStreaming: true
      };

      // Cache the result
      this.audioCache.set(cacheKey, result);

      console.log('🔊 Dedicated Chatbot TTS: Streaming audio generated successfully');
      return result;

    } catch (error) {
      console.error('❌ Dedicated Chatbot TTS: Streaming generation error:', error);
      throw error;
    }
  }

  /**
   * Play TTS audio immediately
   * @param {string} text - Text to convert and play
   * @param {Object} options - Playback options
   * @returns {Promise<void>}
   */
  async playTTS(text, options = {}) {
    try {
      const audioData = await this.generateTTS(text, options);
      await this.playAudio(audioData.fullAudioUrl, {
        ...options,
        isStreaming: audioData.isStreaming
      });
    } catch (error) {
      console.error('❌ Dedicated Chatbot TTS: Playback error:', error);
      throw error;
    }
  }

  /**
   * Play audio from URL
   * @param {string} audioUrl - URL of the audio file
   * @param {Object} options - Playback options
   * @returns {Promise<void>}
   */
  async playAudio(audioUrl, options = {}) {
    return new Promise((resolve, reject) => {
      // Stop current audio if playing
      this.stopCurrentAudio();

      const audio = new Audio(audioUrl);
      audio.volume = options.volume || this.volume;
      this.currentAudio = audio;
      this.isPlaying = true;

      // Set up event listeners
      audio.addEventListener('loadstart', () => {
        console.log('🔊 Dedicated Chatbot TTS: Audio loading started');
      });

      audio.addEventListener('canplay', () => {
        console.log('🔊 Dedicated Chatbot TTS: Audio can start playing');
      });

      audio.addEventListener('play', () => {
        console.log('🔊 Dedicated Chatbot TTS: Audio playback started');
        this.isPlaying = true;
        if (options.onPlayStart) {
          options.onPlayStart(audioUrl);
        }
      });

      audio.addEventListener('ended', () => {
        console.log('🔊 Dedicated Chatbot TTS: Audio playback ended');
        this.isPlaying = false;
        this.currentAudio = null;

        // Cleanup blob URL if it's a streaming audio
        if (options.isStreaming && audioUrl.startsWith('blob:')) {
          URL.revokeObjectURL(audioUrl);
          console.log('🔊 Dedicated Chatbot TTS: Blob URL cleaned up after playback');
        }

        if (options.onPlayEnd) {
          options.onPlayEnd(audioUrl);
        }
        resolve();
      });

      audio.addEventListener('error', (e) => {
        console.error('❌ Dedicated Chatbot TTS: Audio playback error:', e);
        this.isPlaying = false;
        this.currentAudio = null;
        if (options.onError) {
          options.onError(e);
        }
        reject(new Error(`Audio playback failed: ${e.message || 'Unknown error'}`));
      });

      audio.addEventListener('canplaythrough', () => {
        console.log('🔊 Dedicated Chatbot TTS: Audio ready to play');
      });

      // Start playback with better error handling
      const playPromise = audio.play();

      if (playPromise !== undefined) {
        playPromise
          .then(() => {
            console.log('🔊 Dedicated Chatbot TTS: Started audio playback successfully');
          })
          .catch((error) => {
            console.error('❌ Dedicated Chatbot TTS: Play promise rejected:', error);
            this.isPlaying = false;
            this.currentAudio = null;
            
            if (error.name === 'NotAllowedError') {
              reject(new Error('Audio autoplay blocked - user interaction required'));
            } else {
              reject(new Error(`Audio playback failed: ${error.message}`));
            }
          });
      }
    });
  }

  /**
   * Auto-play TTS for AI-generated content
   * @param {string} text - AI-generated text
   * @param {Object} options - Auto-play options
   */
  async autoPlayAI(text, options = {}) {
    if (!this.autoPlayEnabled || !text || !text.trim()) {
      return;
    }

    // Add a small delay for better UX
    const delay = options.delay || 500;
    setTimeout(async () => {
      try {
        await this.playTTS(text, { ...options, autoPlay: true });
      } catch (error) {
        // Handle autoplay policy errors gracefully
        if (error.message.includes('user interaction required')) {
          console.info('🔊 Dedicated Chatbot TTS: Autoplay blocked - will enable after user interaction');
          this.enableAutoplayAfterInteraction();
        } else {
          console.warn('⚠️ Dedicated Chatbot TTS: Auto-play failed, continuing silently:', error.message);
        }
      }
    }, delay);
  }

  /**
   * Enable autoplay after user interaction
   */
  enableAutoplayAfterInteraction() {
    const enableAutoplay = () => {
      console.log('🔊 Dedicated Chatbot TTS: User interaction detected, autoplay enabled');
      document.removeEventListener('click', enableAutoplay);
      document.removeEventListener('keydown', enableAutoplay);
      document.removeEventListener('touchstart', enableAutoplay);
    };

    // Listen for any user interaction
    document.addEventListener('click', enableAutoplay, { once: true });
    document.addEventListener('keydown', enableAutoplay, { once: true });
    document.addEventListener('touchstart', enableAutoplay, { once: true });
  }

  /**
   * Stop current audio playback
   */
  stopCurrentAudio() {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
      this.isPlaying = false;
    }
  }

  /**
   * Generate cache key for audio caching
   * @param {string} text - Text content
   * @param {Object} options - Options object
   * @returns {string} Cache key
   */
  getCacheKey(text, options = {}) {
    const optionsString = JSON.stringify(options);
    return `${text.trim()}_${optionsString}`;
  }

  /**
   * Clear audio cache and cleanup blob URLs
   */
  clearCache() {
    // Cleanup blob URLs to prevent memory leaks
    for (const [key, result] of this.audioCache.entries()) {
      if (result.isStreaming && result.audioUrl) {
        URL.revokeObjectURL(result.audioUrl);
      }
    }
    this.audioCache.clear();
    console.log('🔊 Dedicated Chatbot TTS: Cache cleared and blob URLs cleaned up');
  }

  /**
   * Cleanup blob URL for a specific audio result
   * @param {Object} audioResult - Audio result object
   */
  cleanupAudioResult(audioResult) {
    if (audioResult && audioResult.isStreaming && audioResult.audioUrl) {
      URL.revokeObjectURL(audioResult.audioUrl);
      console.log('🔊 Dedicated Chatbot TTS: Blob URL cleaned up');
    }
  }

  /**
   * Get current status
   * @returns {Object} Current service status
   */
  getStatus() {
    return {
      isPlaying: this.isPlaying,
      autoPlayEnabled: this.autoPlayEnabled,
      volume: this.volume,
      cacheSize: this.audioCache.size,
      currentAudio: this.currentAudio ? 'playing' : 'none',
      service: 'Dedicated Chatbot TTS Service'
    };
  }
}

// Create and export singleton instance
const dedicatedChatbotTTSService = new DedicatedChatbotTTSService();
export default dedicatedChatbotTTSService;
