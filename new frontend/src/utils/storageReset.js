/**
 * Storage Reset Utility
 * Clears all storage to fix quota and corruption issues
 */

export const resetAllStorage = async () => {
  console.log('🧹 Resetting all storage...');
  
  try {
    // Clear localStorage
    localStorage.clear();
    console.log('✅ localStorage cleared');
    
    // Clear sessionStorage
    sessionStorage.clear();
    console.log('✅ sessionStorage cleared');
    
    // Clear IndexedDB
    if ('indexedDB' in window) {
      try {
        const databases = await indexedDB.databases();
        await Promise.all(
          databases.map(db => {
            return new Promise((resolve, reject) => {
              const deleteReq = indexedDB.deleteDatabase(db.name);
              deleteReq.onsuccess = () => resolve();
              deleteReq.onerror = () => reject(deleteReq.error);
            });
          })
        );
        console.log('✅ IndexedDB cleared');
      } catch (error) {
        console.warn('⚠️ Could not clear IndexedDB:', error);
      }
    }
    
    // Clear cache storage
    if ('caches' in window) {
      try {
        const cacheNames = await caches.keys();
        await Promise.all(
          cacheNames.map(cacheName => caches.delete(cacheName))
        );
        console.log('✅ Cache storage cleared');
      } catch (error) {
        console.warn('⚠️ Could not clear cache storage:', error);
      }
    }
    
    console.log('🎉 All storage cleared successfully!');
    console.log('🔄 Please refresh the page to restart with clean storage.');
    
    return true;
  } catch (error) {
    console.error('❌ Error clearing storage:', error);
    return false;
  }
};

// Auto-run in development if quota exceeded
if (import.meta.env.DEV) {
  window.resetStorage = resetAllStorage;
  console.log('🔧 Storage reset utility loaded. Run window.resetStorage() to clear all storage.');
}
