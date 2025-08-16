/**
 * Test script to verify Redux persist functionality
 * Run this in browser console to test persistence
 */

export const testPersistence = () => {
  console.log('🧪 Testing Redux Persist Functionality...');

  // Check if persist data exists in localStorage
  const persistKey = 'persist:avatar';
  const persistedData = localStorage.getItem(persistKey);

  console.log('📦 Persisted data found:', !!persistedData);

  // Check for corrupted data and clean it up
  if (persistedData && persistedData.includes('[object Object]')) {
    console.warn('🚨 Corrupted persist data detected, clearing...');
    localStorage.removeItem(persistKey);
    console.log('✅ Corrupted data cleared, please refresh the page');
    return;
  }
  
  if (persistedData) {
    try {
      const parsed = JSON.parse(persistedData);
      console.log('📋 Persisted avatar state:', {
        selectedAvatar: parsed.selectedAvatar ? (typeof parsed.selectedAvatar === 'string' ? JSON.parse(parsed.selectedAvatar) : parsed.selectedAvatar) : null,
        isPinModeEnabled: parsed.isPinModeEnabled ? (typeof parsed.isPinModeEnabled === 'string' ? JSON.parse(parsed.isPinModeEnabled) : parsed.isPinModeEnabled) : false,
        pinnedAvatarPosition: parsed.pinnedAvatarPosition ? (typeof parsed.pinnedAvatarPosition === 'string' ? JSON.parse(parsed.pinnedAvatarPosition) : parsed.pinnedAvatarPosition) : null,
        gridPosition: parsed.gridPosition ? (typeof parsed.gridPosition === 'string' ? JSON.parse(parsed.gridPosition) : parsed.gridPosition) : null,
        pinPosition: parsed.pinPosition ? (typeof parsed.pinPosition === 'string' ? JSON.parse(parsed.pinPosition) : parsed.pinPosition) : null,
        favorites: parsed.favorites ? (typeof parsed.favorites === 'string' ? JSON.parse(parsed.favorites) : parsed.favorites) : [],
      });
    } catch (error) {
      console.error('❌ Error parsing persisted data:', error);
      console.log('🔍 Raw persisted data:', persistedData);
    }
  }
  
  // Test instructions
  console.log(`
🔧 Manual Test Instructions:
1. Select an avatar from favorites
2. Enable pin mode
3. Move the pinned avatar to a specific position
4. Adjust position, rotation, and scale settings
5. Perform hard refresh (Ctrl+F5 or Cmd+Shift+R)
6. Verify:
   ✅ Same avatar is selected
   ✅ Pin mode is still enabled
   ✅ Avatar appears at same screen position
   ✅ All settings are preserved
  `);
};

// Auto-run test when imported
if (typeof window !== 'undefined') {
  // Make test available globally for console access
  window.testPersistence = testPersistence;
  // console.log('🔧 Persistence test available: window.testPersistence()');
}
