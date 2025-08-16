/**
 * Storage Management Utility
 * Helps manage browser storage quota and prevents QuotaExceededError
 */

/**
 * Check current storage usage
 */
export const getStorageUsage = async () => {
  if ('storage' in navigator && 'estimate' in navigator.storage) {
    try {
      const estimate = await navigator.storage.estimate();
      return {
        used: estimate.usage,
        quota: estimate.quota,
        usedMB: (estimate.usage / 1024 / 1024).toFixed(2),
        quotaMB: (estimate.quota / 1024 / 1024).toFixed(2),
        percentUsed: ((estimate.usage / estimate.quota) * 100).toFixed(1)
      };
    } catch (error) {
      console.error('Error getting storage estimate:', error);
      return null;
    }
  }
  return null;
};

/**
 * Clean up old or unnecessary localStorage items
 */
export const cleanupStorage = () => {
  try {
    const keysToRemove = [];
    const now = Date.now();
    const oneWeekAgo = now - (7 * 24 * 60 * 60 * 1000); // 7 days ago

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key) continue;

      // Remove blob URLs (they're temporary anyway)
      if (key.includes('blob:')) {
        keysToRemove.push(key);
        continue;
      }

      // Remove temporary cache items
      if (key.includes('temp_') || key.includes('cache_')) {
        keysToRemove.push(key);
        continue;
      }

      // Remove old timestamped items
      if (key.includes('_timestamp_')) {
        try {
          const item = localStorage.getItem(key);
          const data = JSON.parse(item);
          if (data.timestamp && new Date(data.timestamp).getTime() < oneWeekAgo) {
            keysToRemove.push(key);
          }
        } catch (e) {
          // If we can't parse it, it might be corrupted, remove it
          keysToRemove.push(key);
        }
      }
    }

    // Remove identified keys
    keysToRemove.forEach(key => {
      try {
        localStorage.removeItem(key);
      } catch (error) {
        console.error(`Error removing key ${key}:`, error);
      }
    });

    console.log(`Cleaned up ${keysToRemove.length} storage items`);
    return keysToRemove.length;
  } catch (error) {
    console.error('Error during storage cleanup:', error);
    return 0;
  }
};

/**
 * Check if storage has enough space for an operation
 */
export const hasStorageSpace = async (estimatedSizeBytes = 1024 * 1024) => { // Default 1MB
  const usage = await getStorageUsage();
  if (!usage) return true; // If we can't check, assume it's fine

  const remainingSpace = usage.quota - usage.used;
  return remainingSpace > estimatedSizeBytes;
};

/**
 * Safe storage setter that handles quota errors
 */
export const safeSetItem = async (key, value) => {
  try {
    localStorage.setItem(key, value);
    return { success: true };
  } catch (error) {
    if (error.name === 'QuotaExceededError') {
      console.warn('Storage quota exceeded, attempting cleanup...');
      
      // Try cleanup and retry
      const cleanedItems = cleanupStorage();
      
      if (cleanedItems > 0) {
        try {
          localStorage.setItem(key, value);
          return { success: true, cleaned: cleanedItems };
        } catch (retryError) {
          return { 
            success: false, 
            error: 'Storage quota exceeded even after cleanup',
            cleaned: cleanedItems 
          };
        }
      } else {
        return { 
          success: false, 
          error: 'Storage quota exceeded and no items to clean' 
        };
      }
    } else {
      return { success: false, error: error.message };
    }
  }
};

/**
 * Get storage usage summary for debugging
 */
export const getStorageSummary = async () => {
  const usage = await getStorageUsage();
  const localStorageSize = new Blob(Object.values(localStorage)).size;
  
  return {
    ...usage,
    localStorageSize: (localStorageSize / 1024 / 1024).toFixed(2) + 'MB',
    itemCount: localStorage.length
  };
};

/**
 * Emergency storage clear (use with caution)
 */
export const emergencyStorageClear = () => {
  try {
    // Clear everything except essential auth data
    const essentialKeys = ['supabase.auth.token', 'persist:auth'];
    const backup = {};
    
    // Backup essential data
    essentialKeys.forEach(key => {
      const value = localStorage.getItem(key);
      if (value) backup[key] = value;
    });
    
    // Clear all storage
    localStorage.clear();
    sessionStorage.clear();
    
    // Restore essential data
    Object.entries(backup).forEach(([key, value]) => {
      try {
        localStorage.setItem(key, value);
      } catch (error) {
        console.error(`Failed to restore ${key}:`, error);
      }
    });
    
    console.log('Emergency storage clear completed');
    return true;
  } catch (error) {
    console.error('Emergency storage clear failed:', error);
    return false;
  }
};
