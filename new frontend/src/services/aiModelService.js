/**
 * AI Model Generation Service
 * Integrates with various AI services for text-to-3D and image-to-3D generation
 */

// AI Service Providers Configuration
const AI_PROVIDERS = {
  MESHY: {
    name: 'Meshy AI',
    baseUrl: 'https://api.meshy.ai/v1',
    textTo3D: '/text-to-3d',
    imageTo3D: '/image-to-3d',
    supportedFormats: ['glb', 'gltf', 'obj'],
    maxFileSize: 10 * 1024 * 1024, // 10MB
  },
  TRIPO: {
    name: 'Tripo AI',
    baseUrl: 'https://api.tripo3d.ai/v1',
    textTo3D: '/text-to-model',
    imageTo3D: '/image-to-model',
    supportedFormats: ['glb', 'gltf'],
    maxFileSize: 5 * 1024 * 1024, // 5MB
  },
  STABILITY: {
    name: 'Stability AI',
    baseUrl: 'https://api.stability.ai/v1',
    imageTo3D: '/generation/stable-3d',
    supportedFormats: ['glb'],
    maxFileSize: 8 * 1024 * 1024, // 8MB
  }
};

// Default configuration
const DEFAULT_CONFIG = {
  provider: 'MESHY',
  timeout: 120000, // 2 minutes
  retryAttempts: 3,
  retryDelay: 2000, // 2 seconds
};

/**
 * AI Model Generation Service Class
 */
class AIModelService {
  constructor(config = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.apiKey = null;
    this.currentProvider = AI_PROVIDERS[this.config.provider];
  }

  /**
   * Set API key for the current provider
   */
  setApiKey(apiKey) {
    this.apiKey = apiKey;
  }

  /**
   * Switch AI provider
   */
  setProvider(providerName) {
    if (!AI_PROVIDERS[providerName]) {
      throw new Error(`Unsupported provider: ${providerName}`);
    }
    this.currentProvider = AI_PROVIDERS[providerName];
    this.config.provider = providerName;
  }

  /**
   * Generate 3D model from text prompt
   */
  async generateFromText(prompt, options = {}) {
    if (!this.currentProvider.textTo3D) {
      throw new Error(`${this.currentProvider.name} doesn't support text-to-3D generation`);
    }

    const requestData = {
      prompt: prompt.trim(),
      format: options.format || 'glb',
      quality: options.quality || 'medium',
      style: options.style || 'realistic',
      ...options
    };

    return this._makeRequest('textTo3D', requestData);
  }

  /**
   * Generate 3D model from image
   */
  async generateFromImage(imageFile, options = {}) {
    if (!this.currentProvider.imageTo3D) {
      throw new Error(`${this.currentProvider.name} doesn't support image-to-3D generation`);
    }

    // Validate file size
    if (imageFile.size > this.currentProvider.maxFileSize) {
      throw new Error(`File size exceeds maximum allowed size of ${this.currentProvider.maxFileSize / (1024 * 1024)}MB`);
    }

    // Validate file type
    if (!imageFile.type.startsWith('image/')) {
      throw new Error('Invalid file type. Please upload an image file.');
    }

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('format', options.format || 'glb');
    formData.append('quality', options.quality || 'medium');

    return this._makeRequest('imageTo3D', formData, true);
  }

  /**
   * Check generation status
   */
  async checkStatus(taskId) {
    const endpoint = `${this.currentProvider.baseUrl}/tasks/${taskId}`;
    
    try {
      const response = await fetch(endpoint, {
        method: 'GET',
        headers: this._getHeaders(),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error checking generation status:', error);
      throw error;
    }
  }

  /**
   * Download generated model
   */
  async downloadModel(downloadUrl, filename = 'generated-model.glb') {
    try {
      const response = await fetch(downloadUrl);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      
      // Create download link
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up
      URL.revokeObjectURL(url);
      
      return blob;
    } catch (error) {
      console.error('Error downloading model:', error);
      throw error;
    }
  }

  /**
   * Get supported providers
   */
  static getSupportedProviders() {
    return Object.keys(AI_PROVIDERS).map(key => ({
      id: key,
      name: AI_PROVIDERS[key].name,
      textTo3D: !!AI_PROVIDERS[key].textTo3D,
      imageTo3D: !!AI_PROVIDERS[key].imageTo3D,
      supportedFormats: AI_PROVIDERS[key].supportedFormats,
    }));
  }

  /**
   * Validate prompt for text-to-3D generation
   */
  static validatePrompt(prompt) {
    if (!prompt || typeof prompt !== 'string') {
      return { valid: false, error: 'Prompt is required and must be a string' };
    }

    const trimmedPrompt = prompt.trim();
    if (trimmedPrompt.length < 3) {
      return { valid: false, error: 'Prompt must be at least 3 characters long' };
    }

    if (trimmedPrompt.length > 500) {
      return { valid: false, error: 'Prompt must be less than 500 characters' };
    }

    return { valid: true };
  }

  /**
   * Private method to make API requests
   */
  async _makeRequest(endpoint, data, isFormData = false) {
    if (!this.apiKey) {
      throw new Error('API key is required. Please set it using setApiKey()');
    }

    const url = `${this.currentProvider.baseUrl}${this.currentProvider[endpoint]}`;
    const headers = this._getHeaders(isFormData);

    const requestOptions = {
      method: 'POST',
      headers,
      body: isFormData ? data : JSON.stringify(data),
    };

    try {
      const response = await this._fetchWithRetry(url, requestOptions);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('AI Model Service Error:', error);
      throw error;
    }
  }

  /**
   * Get request headers
   */
  _getHeaders(isFormData = false) {
    const headers = {
      'Authorization': `Bearer ${this.apiKey}`,
    };

    if (!isFormData) {
      headers['Content-Type'] = 'application/json';
    }

    return headers;
  }

  /**
   * Fetch with retry logic
   */
  async _fetchWithRetry(url, options, attempt = 1) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      if (attempt < this.config.retryAttempts && error.name !== 'AbortError') {
        console.warn(`Request failed, retrying... (${attempt}/${this.config.retryAttempts})`);
        await new Promise(resolve => setTimeout(resolve, this.config.retryDelay));
        return this._fetchWithRetry(url, options, attempt + 1);
      }
      throw error;
    }
  }
}

// Export singleton instance
export const aiModelService = new AIModelService();

// Export class for custom instances
export default AIModelService;

// Export utility functions
export const {
  getSupportedProviders,
  validatePrompt,
} = AIModelService;
