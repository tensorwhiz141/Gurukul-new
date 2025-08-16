/**
 * TTS (Text-to-Speech) Service Integration
 * Provides seamless integration with the Gurukul TTS backend service
 */

const TTS_BASE_URL = 'http://localhost:8007';

class TTSService {
  constructor() {
    this.audioCache = new Map();
    this.currentAudio = null;
    this.isPlaying = false;
    this.autoPlayEnabled = true;
    this.volume = 0.8;
    this.playbackQueue = [];
    this.isProcessingQueue = false;
    this.pendingAudio = null;
    this.autoplayBlocked = false;
    this.userHasInteracted = false;

    // Listen for user interactions to enable autoplay
    this.setupAutoplayUnblock();
  }

  /**
   * Generate TTS audio from text
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
      console.log('🔊 TTS: Using cached audio for text:', text.substring(0, 50) + '...');
      return this.audioCache.get(cacheKey);
    }

    try {
      console.log('🔊 TTS: Generating streaming audio for text:', text.substring(0, 100) + '...');

      // Prepare form data for streaming endpoint
      const formData = new FormData();
      formData.append('text', text);

      const response = await fetch(`${TTS_BASE_URL}/api/generate/stream`, {
        method: 'POST',
        body: formData,
        headers: {
          'User-Agent': 'Gurukul-Frontend-TTS/1.0',
          'X-Source-System': 'gurukul-frontend',
          'X-Request-Type': 'auto-tts-stream'
        }
      });

      if (!response.ok) {
        throw new Error(`TTS streaming failed: ${response.status} ${response.statusText}`);
      }

      // Get audio data as blob
      const audioBlob = await response.blob();

      // Create object URL for the blob
      const audioUrl = URL.createObjectURL(audioBlob);

      const audioData = {
        status: 'success',
        audioUrl: audioUrl,
        fullAudioUrl: audioUrl,
        blob: audioBlob,
        textLength: text.length,
        text: text,
        timestamp: Date.now(),
        isStreaming: true
      };

      this.audioCache.set(cacheKey, audioData);
      console.log('✅ TTS: Streaming audio generated successfully');

      return audioData;
    } catch (error) {
      console.error('❌ TTS: Streaming generation error:', error);
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
      console.error('❌ TTS: Playback error:', error);
      throw error;
    }
  }

  /**
   * Add TTS to playback queue for sequential playing
   * @param {string} text - Text to convert and queue
   * @param {Object} options - Queue options
   */
  async queueTTS(text, options = {}) {
    this.playbackQueue.push({ text, options });
    
    if (!this.isProcessingQueue) {
      this.processQueue();
    }
  }

  /**
   * Process the TTS playback queue
   */
  async processQueue() {
    if (this.isProcessingQueue || this.playbackQueue.length === 0) {
      return;
    }

    this.isProcessingQueue = true;

    while (this.playbackQueue.length > 0) {
      const { text, options } = this.playbackQueue.shift();
      
      try {
        await this.playTTS(text, { ...options, waitForCompletion: true });
      } catch (error) {
        console.error('❌ TTS: Queue processing error:', error);
      }
    }

    this.isProcessingQueue = false;
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
      audio.preload = 'auto';
      audio.crossOrigin = 'anonymous'; // Handle CORS issues

      // Set current audio reference
      this.currentAudio = audio;
      this.isPlaying = true;

      // Event listeners
      audio.addEventListener('ended', () => {
        this.isPlaying = false;
        this.currentAudio = null;

        // Cleanup blob URL if it's a streaming audio
        if (options.isStreaming && audioUrl.startsWith('blob:')) {
          URL.revokeObjectURL(audioUrl);
          console.log('🔊 TTS: Blob URL cleaned up after playback');
        }

        console.log('🔊 TTS: Audio playback completed');
        resolve();
      });

      audio.addEventListener('error', (error) => {
        this.isPlaying = false;
        this.currentAudio = null;
        console.error('❌ TTS: Audio playback error:', error);
        reject(new Error('Audio playback failed'));
      });

      audio.addEventListener('canplaythrough', () => {
        console.log('🔊 TTS: Audio ready to play');
      });

      // Start playback with better error handling
      const playPromise = audio.play();

      if (playPromise !== undefined) {
        playPromise
          .then(() => {
            console.log('🔊 TTS: Started audio playback successfully');
          })
          .catch((error) => {
            this.isPlaying = false;
            this.currentAudio = null;

            // Handle autoplay policy errors gracefully
            if (error.name === 'NotAllowedError') {
              console.warn('🔊 TTS: Autoplay blocked by browser policy. Audio ready for user interaction.');

              // Store the audio for later playback when user interacts
              this.pendingAudio = audio;
              this.isPlaying = false;

              // Show user-friendly message instead of error
              this.showAutoplayNotification();

              // Resolve instead of reject to avoid error propagation
              resolve();
            } else {
              console.error('❌ TTS: Audio play error:', error);
              reject(error);
            }
          });
      }
    });
  }

  /**
   * Stop current audio playback
   */
  stopCurrentAudio() {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.src = '';
      this.currentAudio = null;
      this.isPlaying = false;
      console.log('🔊 TTS: Stopped current audio');
    }
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
          console.info('🔊 TTS: Autoplay blocked - will enable after user interaction');
          this.enableAutoplayAfterInteraction();
        } else {
          console.warn('⚠️ TTS: Auto-play failed, continuing silently:', error.message);
        }
      }
    }, delay);
  }

  /**
   * Enable autoplay after user interaction
   */
  enableAutoplayAfterInteraction() {
    const enableAutoplay = () => {
      console.log('🔊 TTS: User interaction detected, autoplay enabled');
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
   * Generate cache key for audio caching
   * @param {string} text - Text content
   * @param {Object} options - Generation options
   * @returns {string} Cache key
   */
  getCacheKey(text, options = {}) {
    const textHash = this.simpleHash(text);
    const optionsHash = this.simpleHash(JSON.stringify(options));
    return `tts_${textHash}_${optionsHash}`;
  }

  /**
   * Simple hash function for cache keys
   * @param {string} str - String to hash
   * @returns {string} Hash value
   */
  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Check if TTS service is available
   * @returns {Promise<boolean>} Service availability
   */
  async checkServiceHealth() {
    try {
      const response = await fetch(`${TTS_BASE_URL}/api/health`, {
        method: 'GET',
        timeout: 5000
      });
      
      if (response.ok) {
        const health = await response.json();
        console.log('✅ TTS: Service is healthy:', health);
        return true;
      }
      
      return false;
    } catch (error) {
      console.warn('⚠️ TTS: Service health check failed:', error.message);
      return false;
    }
  }

  /**
   * Configure TTS settings
   * @param {Object} settings - TTS settings
   */
  configure(settings = {}) {
    if (typeof settings.autoPlayEnabled === 'boolean') {
      this.autoPlayEnabled = settings.autoPlayEnabled;
    }
    
    if (typeof settings.volume === 'number' && settings.volume >= 0 && settings.volume <= 1) {
      this.volume = settings.volume;
    }

    console.log('🔊 TTS: Configuration updated:', {
      autoPlayEnabled: this.autoPlayEnabled,
      volume: this.volume
    });
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
    console.log('🔊 TTS: Audio cache cleared and blob URLs cleaned up');
  }

  /**
   * Cleanup blob URL for a specific audio result
   * @param {Object} audioResult - Audio result object
   */
  cleanupAudioResult(audioResult) {
    if (audioResult && audioResult.isStreaming && audioResult.audioUrl) {
      URL.revokeObjectURL(audioResult.audioUrl);
      console.log('🔊 TTS: Blob URL cleaned up');
    }
  }

  /**
   * Get service status
   * @returns {Object} Service status
   */
  getStatus() {
    return {
      isPlaying: this.isPlaying,
      autoPlayEnabled: this.autoPlayEnabled,
      volume: this.volume,
      cacheSize: this.audioCache.size,
      queueLength: this.playbackQueue.length,
      isProcessingQueue: this.isProcessingQueue
    };
  }

  /**
   * Setup autoplay unblock listeners
   */
  setupAutoplayUnblock() {
    const enableAutoplay = () => {
      this.userHasInteracted = true;
      this.autoplayBlocked = false;

      // Play pending audio if available
      if (this.pendingAudio) {
        console.log('🔊 TTS: Playing pending audio after user interaction');
        this.pendingAudio.play().catch(console.error);
        this.pendingAudio = null;
      }

      // Remove listeners after first interaction
      document.removeEventListener('click', enableAutoplay);
      document.removeEventListener('keydown', enableAutoplay);
      document.removeEventListener('touchstart', enableAutoplay);
    };

    // Add event listeners for user interaction
    document.addEventListener('click', enableAutoplay, { once: true });
    document.addEventListener('keydown', enableAutoplay, { once: true });
    document.addEventListener('touchstart', enableAutoplay, { once: true });
  }

  /**
   * Show autoplay notification to user
   */
  showAutoplayNotification() {
    if (this.autoplayBlocked) return; // Don't show multiple notifications

    this.autoplayBlocked = true;

    // Create a subtle notification
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 14px;
      z-index: 10000;
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      max-width: 300px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    `;
    notification.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px;">
        <span>🔊</span>
        <span>Click anywhere to enable audio</span>
      </div>
    `;

    document.body.appendChild(notification);

    // Remove notification after 5 seconds or on user interaction
    const removeNotification = () => {
      if (notification.parentNode) {
        notification.remove();
      }
    };

    setTimeout(removeNotification, 5000);

    // Remove on user interaction
    const handleInteraction = () => {
      removeNotification();
      document.removeEventListener('click', handleInteraction);
      document.removeEventListener('keydown', handleInteraction);
      document.removeEventListener('touchstart', handleInteraction);
    };

    document.addEventListener('click', handleInteraction, { once: true });
    document.addEventListener('keydown', handleInteraction, { once: true });
    document.addEventListener('touchstart', handleInteraction, { once: true });
  }
}

// Create and export singleton instance
const ttsService = new TTSService();

export default ttsService;

// Named exports for convenience
export {
  ttsService,
  TTSService
};
