/**
 * Storage Quota Manager
 * Handles localStorage quota exceeded errors and provides cleanup utilities
 */

export class StorageQuotaManager {
  static async checkStorageUsage() {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      try {
        const estimate = await navigator.storage.estimate();
        const usagePercent = estimate.usage / estimate.quota;
        const usageMB = (estimate.usage / 1024 / 1024).toFixed(2);
        const quotaMB = (estimate.quota / 1024 / 1024).toFixed(2);
        
        console.log(`📊 Storage usage: ${usageMB}MB / ${quotaMB}MB (${(usagePercent * 100).toFixed(1)}%)`);
        
        return {
          usage: estimate.usage,
          quota: estimate.quota,
          usagePercent,
          usageMB: parseFloat(usageMB),
          quotaMB: parseFloat(quotaMB)
        };
      } catch (error) {
        console.error('Error checking storage usage:', error);
        return null;
      }
    }
    return null;
  }

  static clearStorageSelectively() {
    try {
      console.log('🧹 Starting selective storage cleanup...');
      
      // Keep essential auth and system data
      const keysToKeep = [
        'supabase.auth.token',
        'sb-auth-token',
        'gurukul_avatar_version',
        'gurukul_last_selected_avatar' // Keep last selected for restoration
      ];
      
      // Get all keys first
      const allKeys = Object.keys(localStorage);
      let removedCount = 0;
      
      // Remove non-essential keys
      allKeys.forEach(key => {
        const shouldKeep = keysToKeep.some(keepKey => 
          key.includes(keepKey) || key === keepKey
        );
        
        if (!shouldKeep) {
          try {
            localStorage.removeItem(key);
            removedCount++;
          } catch (error) {
            console.warn(`Failed to remove key ${key}:`, error);
          }
        }
      });
      
      console.log(`✅ Removed ${removedCount} storage items, kept ${keysToKeep.length} essential items`);
      return true;
    } catch (error) {
      console.error('Error during selective cleanup:', error);
      return false;
    }
  }

  static clearStorageCompletely() {
    try {
      console.log('🚨 Performing complete storage cleanup...');
      
      // Save essential auth data
      const authData = {};
      const essentialKeys = [
        'supabase.auth.token',
        'sb-auth-token'
      ];
      
      essentialKeys.forEach(key => {
        const value = localStorage.getItem(key);
        if (value) {
          authData[key] = value;
        }
      });
      
      // Clear everything
      localStorage.clear();
      
      // Restore auth data
      Object.entries(authData).forEach(([key, value]) => {
        try {
          localStorage.setItem(key, value);
        } catch (error) {
          console.warn(`Failed to restore ${key}:`, error);
        }
      });
      
      console.log('✅ Complete cleanup done, auth data restored');
      return true;
    } catch (error) {
      console.error('Error during complete cleanup:', error);
      // Last resort
      localStorage.clear();
      return false;
    }
  }

  static async handleQuotaExceeded() {
    console.warn('🚨 Storage quota exceeded, attempting recovery...');
    
    // Try selective cleanup first
    const selectiveSuccess = this.clearStorageSelectively();
    
    if (selectiveSuccess) {
      // Check if we have enough space now
      const usage = await this.checkStorageUsage();
      if (usage && usage.usagePercent < 0.8) {
        console.log('✅ Selective cleanup successful');
        return 'selective';
      }
    }
    
    // If selective cleanup didn't work, do complete cleanup
    console.warn('🔄 Selective cleanup insufficient, doing complete cleanup...');
    const completeSuccess = this.clearStorageCompletely();
    
    if (completeSuccess) {
      console.log('✅ Complete cleanup successful');
      return 'complete';
    }
    
    console.error('❌ All cleanup attempts failed');
    return 'failed';
  }

  static testStorageWrite() {
    try {
      const testKey = 'storage_quota_test';
      const testValue = 'test';
      localStorage.setItem(testKey, testValue);
      localStorage.removeItem(testKey);
      return true;
    } catch (error) {
      if (error.name === 'QuotaExceededError') {
        return false;
      }
      throw error;
    }
  }

  static async monitorAndCleanup() {
    try {
      // Check if we can write to storage
      if (!this.testStorageWrite()) {
        await this.handleQuotaExceeded();
        return true;
      }
      
      // Check usage percentage
      const usage = await this.checkStorageUsage();
      if (usage && usage.usagePercent > 0.9) {
        console.warn('🚨 Storage usage above 90%, preemptive cleanup...');
        await this.handleQuotaExceeded();
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Error in storage monitoring:', error);
      return false;
    }
  }
}

// Global error handler for quota exceeded errors
window.addEventListener('error', (event) => {
  if (event.error && event.error.name === 'QuotaExceededError') {
    console.warn('🚨 Global quota exceeded error detected');
    StorageQuotaManager.handleQuotaExceeded().then((result) => {
      if (result !== 'failed') {
        console.log('🔄 Reloading page after storage cleanup...');
        setTimeout(() => window.location.reload(), 1000);
      }
    });
  }
});

export default StorageQuotaManager;
