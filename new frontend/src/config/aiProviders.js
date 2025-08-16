/**
 * AI Provider Configuration
 * Configure API keys and settings for different AI 3D generation services
 */

// Environment variables for API keys (should be set in .env file)
export const AI_API_KEYS = {
  MESHY: import.meta.env.VITE_MESHY_API_KEY || '',
  TRIPO: import.meta.env.VITE_TRIPO_API_KEY || '',
  STABILITY: import.meta.env.VITE_STABILITY_API_KEY || '',
};

// Default provider settings
export const DEFAULT_PROVIDER_SETTINGS = {
  provider: 'MESHY',
  timeout: 120000, // 2 minutes
  retryAttempts: 3,
  retryDelay: 2000, // 2 seconds
  defaultFormat: 'glb',
  defaultQuality: 'medium',
  defaultStyle: 'realistic',
};

// Provider-specific configurations
export const PROVIDER_CONFIGS = {
  MESHY: {
    name: 'Meshy AI',
    description: 'High-quality 3D model generation with excellent detail',
    website: 'https://meshy.ai',
    features: {
      textTo3D: true,
      imageTo3D: true,
      realtime: false,
      maxFileSize: 10 * 1024 * 1024, // 10MB
    },
    pricing: {
      free: true,
      freeTier: '200 credits/month',
      paid: 'Starting at $20/month',
    },
    formats: ['glb', 'gltf', 'obj'],
    qualities: ['low', 'medium', 'high'],
    styles: ['realistic', 'cartoon', 'stylized', 'low-poly'],
  },
  TRIPO: {
    name: 'Tripo AI',
    description: 'Fast 3D generation with good quality and speed',
    website: 'https://tripo3d.ai',
    features: {
      textTo3D: true,
      imageTo3D: true,
      realtime: true,
      maxFileSize: 5 * 1024 * 1024, // 5MB
    },
    pricing: {
      free: true,
      freeTier: '100 generations/month',
      paid: 'Starting at $15/month',
    },
    formats: ['glb', 'gltf'],
    qualities: ['low', 'medium', 'high'],
    styles: ['realistic', 'cartoon', 'stylized'],
  },
  STABILITY: {
    name: 'Stability AI',
    description: 'Stable Diffusion 3D - reliable and consistent results',
    website: 'https://stability.ai',
    features: {
      textTo3D: false,
      imageTo3D: true,
      realtime: false,
      maxFileSize: 8 * 1024 * 1024, // 8MB
    },
    pricing: {
      free: false,
      freeTier: 'None',
      paid: 'Pay per use - $0.10 per generation',
    },
    formats: ['glb'],
    qualities: ['medium', 'high'],
    styles: ['realistic'],
  },
};

// Sample prompts for text-to-3D generation
export const SAMPLE_PROMPTS = {
  characters: [
    'a futuristic robot with glowing blue eyes',
    'a medieval knight in shining armor',
    'a cute cartoon cat with big eyes',
    'a wise old wizard with a long beard',
    'a cyberpunk hacker with neon accessories',
  ],
  objects: [
    'a vintage wooden chair with intricate carvings',
    'a modern sports car with sleek design',
    'a magical crystal that glows from within',
    'a steampunk mechanical device with gears',
    'a cozy cottage with a thatched roof',
  ],
  environments: [
    'a floating island with waterfalls',
    'a space station orbiting a planet',
    'an underwater coral reef scene',
    'a mystical forest with glowing mushrooms',
    'a desert oasis with palm trees',
  ],
};

// Quality settings and their descriptions
export const QUALITY_DESCRIPTIONS = {
  low: {
    label: 'Low (Fast)',
    description: 'Quick generation, lower detail',
    estimatedTime: '30-60 seconds',
    polygonCount: '< 1K triangles',
  },
  medium: {
    label: 'Medium (Balanced)',
    description: 'Good balance of quality and speed',
    estimatedTime: '1-3 minutes',
    polygonCount: '1K-5K triangles',
  },
  high: {
    label: 'High (Detailed)',
    description: 'Best quality, longer generation time',
    estimatedTime: '3-10 minutes',
    polygonCount: '5K-20K triangles',
  },
};

// Style presets and their characteristics
export const STYLE_DESCRIPTIONS = {
  realistic: {
    label: 'Realistic',
    description: 'Photorealistic appearance with detailed textures',
    bestFor: 'Product visualization, architectural models',
  },
  cartoon: {
    label: 'Cartoon',
    description: 'Stylized, colorful, and playful appearance',
    bestFor: 'Game assets, character design, animations',
  },
  stylized: {
    label: 'Stylized',
    description: 'Artistic interpretation with unique visual style',
    bestFor: 'Creative projects, artistic visualization',
  },
  'low-poly': {
    label: 'Low Poly',
    description: 'Minimalist geometric style with flat surfaces',
    bestFor: 'Game development, mobile applications',
  },
};

// File format information
export const FORMAT_INFO = {
  glb: {
    name: 'GLB (Binary glTF)',
    description: 'Compact binary format, best for web and real-time applications',
    fileSize: 'Small',
    compatibility: 'Excellent',
    recommended: true,
  },
  gltf: {
    name: 'glTF (JSON)',
    description: 'Human-readable format with separate texture files',
    fileSize: 'Medium',
    compatibility: 'Excellent',
    recommended: false,
  },
  obj: {
    name: 'OBJ (Wavefront)',
    description: 'Widely supported legacy format',
    fileSize: 'Large',
    compatibility: 'Universal',
    recommended: false,
  },
};

// Validation rules
export const VALIDATION_RULES = {
  prompt: {
    minLength: 3,
    maxLength: 500,
    required: true,
  },
  image: {
    maxSize: 10 * 1024 * 1024, // 10MB
    allowedTypes: ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'],
    minDimensions: { width: 256, height: 256 },
    maxDimensions: { width: 4096, height: 4096 },
  },
};

// Error messages
export const ERROR_MESSAGES = {
  NO_API_KEY: 'API key is required. Please configure your API key in settings.',
  INVALID_PROMPT: 'Please enter a valid prompt (3-500 characters).',
  INVALID_IMAGE: 'Please upload a valid image file (JPEG, PNG, WebP, or BMP).',
  IMAGE_TOO_LARGE: 'Image file is too large. Maximum size is 10MB.',
  IMAGE_TOO_SMALL: 'Image dimensions are too small. Minimum size is 256x256 pixels.',
  GENERATION_FAILED: 'Model generation failed. Please try again.',
  NETWORK_ERROR: 'Network error. Please check your connection and try again.',
  QUOTA_EXCEEDED: 'API quota exceeded. Please upgrade your plan or try again later.',
};

// Success messages
export const SUCCESS_MESSAGES = {
  GENERATION_STARTED: 'Model generation started successfully!',
  GENERATION_COMPLETED: '3D model generated successfully!',
  MODEL_DOWNLOADED: 'Model downloaded successfully!',
  MODEL_SELECTED: 'Model selected as avatar!',
  SETTINGS_SAVED: 'Settings saved successfully!',
};

// Helper function to get provider configuration
export const getProviderConfig = (providerId) => {
  return PROVIDER_CONFIGS[providerId] || PROVIDER_CONFIGS.MESHY;
};

// Helper function to validate API key
export const validateApiKey = (providerId, apiKey) => {
  if (!apiKey || typeof apiKey !== 'string' || apiKey.trim().length === 0) {
    return { valid: false, error: ERROR_MESSAGES.NO_API_KEY };
  }
  
  // Basic format validation (can be enhanced per provider)
  if (apiKey.length < 10) {
    return { valid: false, error: 'API key appears to be too short.' };
  }
  
  return { valid: true };
};

// Helper function to get available providers
export const getAvailableProviders = () => {
  return Object.keys(PROVIDER_CONFIGS).map(id => ({
    id,
    ...PROVIDER_CONFIGS[id],
    hasApiKey: !!AI_API_KEYS[id],
  }));
};
