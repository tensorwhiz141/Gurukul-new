/**
 * Utility functions for 3D model handling and processing
 */

/**
 * Validate 3D model file format
 */
export const validateModelFormat = (file) => {
  const supportedFormats = ['.glb', '.gltf', '.obj', '.fbx', '.dae'];
  const fileExtension = file.name.toLowerCase().split('.').pop();
  
  return {
    isValid: supportedFormats.includes(`.${fileExtension}`),
    format: fileExtension,
    supportedFormats
  };
};

/**
 * Get model file size in human readable format
 */
export const formatModelSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * Generate thumbnail from 3D model URL
 */
export const generateModelThumbnail = async (modelUrl, options = {}) => {
  const {
    width = 256,
    height = 256,
    backgroundColor = '#000000'
  } = options;

  try {
    // This would typically use Three.js to render a thumbnail
    // For now, return a placeholder
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);
    
    // Add placeholder text
    ctx.fillStyle = '#ffffff';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('3D Model', width / 2, height / 2);
    
    return canvas.toDataURL();
  } catch (error) {
    console.error('Error generating thumbnail:', error);
    return null;
  }
};

/**
 * Cache model data in localStorage
 */
export const cacheModel = (modelId, modelData) => {
  try {
    const cacheKey = `model_cache_${modelId}`;
    const cacheData = {
      ...modelData,
      cachedAt: new Date().toISOString(),
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString() // 24 hours
    };
    
    localStorage.setItem(cacheKey, JSON.stringify(cacheData));
    return true;
  } catch (error) {
    console.error('Error caching model:', error);
    return false;
  }
};

/**
 * Retrieve cached model data
 */
export const getCachedModel = (modelId) => {
  try {
    const cacheKey = `model_cache_${modelId}`;
    const cachedData = localStorage.getItem(cacheKey);
    
    if (!cachedData) return null;
    
    const modelData = JSON.parse(cachedData);
    const now = new Date();
    const expiresAt = new Date(modelData.expiresAt);
    
    if (now > expiresAt) {
      localStorage.removeItem(cacheKey);
      return null;
    }
    
    return modelData;
  } catch (error) {
    console.error('Error retrieving cached model:', error);
    return null;
  }
};

/**
 * Clear expired model cache entries
 */
export const clearExpiredCache = () => {
  try {
    const now = new Date();
    const keysToRemove = [];
    
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      
      if (key && key.startsWith('model_cache_')) {
        try {
          const data = JSON.parse(localStorage.getItem(key));
          const expiresAt = new Date(data.expiresAt);
          
          if (now > expiresAt) {
            keysToRemove.push(key);
          }
        } catch (e) {
          // Invalid cache entry, mark for removal
          keysToRemove.push(key);
        }
      }
    }
    
    keysToRemove.forEach(key => localStorage.removeItem(key));
    return keysToRemove.length;
  } catch (error) {
    console.error('Error clearing expired cache:', error);
    return 0;
  }
};

/**
 * Get all cached models
 */
export const getAllCachedModels = () => {
  try {
    const models = [];
    
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      
      if (key && key.startsWith('model_cache_')) {
        try {
          const data = JSON.parse(localStorage.getItem(key));
          const now = new Date();
          const expiresAt = new Date(data.expiresAt);
          
          if (now <= expiresAt) {
            models.push(data);
          }
        } catch (e) {
          // Skip invalid entries
        }
      }
    }
    
    return models.sort((a, b) => new Date(b.cachedAt) - new Date(a.cachedAt));
  } catch (error) {
    console.error('Error getting cached models:', error);
    return [];
  }
};

/**
 * Optimize model for web display
 */
export const optimizeModelForWeb = (modelData, options = {}) => {
  const {
    maxFileSize = 5 * 1024 * 1024, // 5MB
    targetQuality = 'medium'
  } = options;

  // This would typically involve:
  // 1. Reducing polygon count
  // 2. Compressing textures
  // 3. Optimizing materials
  // 4. Converting to efficient formats
  
  // For now, return the original data with optimization flags
  return {
    ...modelData,
    optimized: true,
    optimizationLevel: targetQuality,
    originalSize: modelData.size,
    optimizedSize: Math.min(modelData.size, maxFileSize)
  };
};

/**
 * Convert model coordinates for different coordinate systems
 */
export const convertModelCoordinates = (position, fromSystem = 'blender', toSystem = 'threejs') => {
  // Blender uses Z-up, Three.js uses Y-up
  if (fromSystem === 'blender' && toSystem === 'threejs') {
    return [position[0], position[2], -position[1]];
  }
  
  if (fromSystem === 'threejs' && toSystem === 'blender') {
    return [position[0], -position[2], position[1]];
  }
  
  return position;
};

/**
 * Calculate model bounding box
 */
export const calculateBoundingBox = (vertices) => {
  if (!vertices || vertices.length === 0) {
    return {
      min: [0, 0, 0],
      max: [0, 0, 0],
      center: [0, 0, 0],
      size: [0, 0, 0]
    };
  }

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < vertices.length; i += 3) {
    const x = vertices[i];
    const y = vertices[i + 1];
    const z = vertices[i + 2];

    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  const min = [minX, minY, minZ];
  const max = [maxX, maxY, maxZ];
  const center = [
    (minX + maxX) / 2,
    (minY + maxY) / 2,
    (minZ + maxZ) / 2
  ];
  const size = [
    maxX - minX,
    maxY - minY,
    maxZ - minZ
  ];

  return { min, max, center, size };
};

/**
 * Generate model metadata
 */
export const generateModelMetadata = (modelData, options = {}) => {
  const {
    includeStats = true,
    includeThumbnail = true
  } = options;

  const metadata = {
    id: modelData.id || Date.now().toString(),
    name: modelData.name || 'Untitled Model',
    format: modelData.format || 'glb',
    createdAt: modelData.createdAt || new Date().toISOString(),
    type: modelData.type || 'unknown',
    url: modelData.url,
    thumbnail: modelData.thumbnail
  };

  if (includeStats && modelData.stats) {
    metadata.stats = {
      vertices: modelData.stats.vertices || 0,
      faces: modelData.stats.faces || 0,
      materials: modelData.stats.materials || 0,
      textures: modelData.stats.textures || 0
    };
  }

  if (modelData.prompt) {
    metadata.prompt = modelData.prompt;
  }

  if (modelData.sourceImage) {
    metadata.sourceImage = modelData.sourceImage;
  }

  return metadata;
};

/**
 * Validate model URL accessibility
 */
export const validateModelUrl = async (url) => {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return {
      isValid: response.ok,
      status: response.status,
      contentType: response.headers.get('content-type'),
      contentLength: response.headers.get('content-length')
    };
  } catch (error) {
    return {
      isValid: false,
      error: error.message
    };
  }
};
