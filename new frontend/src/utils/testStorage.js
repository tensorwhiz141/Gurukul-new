/**
 * Test script to verify storage utilities work correctly
 * This can be run in the browser console to test persistence
 */

import { storage } from './storageUtils';

export const testStoragePersistence = () => {
  console.log('🧪 Testing Avatar Storage Persistence...');
  
  // Test 1: Save some test data
  console.log('📝 Step 1: Saving test data...');
  
  const testFavorites = [
    {
      id: 'test-1',
      name: 'Test Avatar 1',
      gridPosition: { x: 1, y: 2, z: 3 },
      pinPosition: { x: 4, y: 5, z: 6 },
      isPinModeEnabled: true,
      pinnedAvatarPosition: { x: 200, y: 300 }
    }
  ];
  
  const testGlobalState = {
    isPinModeEnabled: true,
    pinnedAvatarPosition: { x: 150, y: 250 },
    activeSettingsTab: 'pin',
    gridPosition: { x: 1, y: 1, z: 1 },
    gridRotation: { x: 10, y: 20, z: 30 },
    gridScale: 1.5,
    pinPosition: { x: 2, y: 2, z: 2 },
    pinRotation: { x: 40, y: 50, z: 60 },
    pinScale: 2.0,
  };
  
  // Save test data
  storage.setItem('gurukul_favorite_avatars', JSON.stringify(testFavorites));
  storage.setItem('gurukul_last_selected_avatar', 'test-1');
  storage.saveAvatarGlobalState(testGlobalState);
  
  // Add some auth data that should be cleared
  storage.setItem('supabase.auth.token', 'test-auth-token');
  storage.setItem('sb-auth-token-123', 'test-sb-token');
  storage.setItem('lastVisitedPath', '/test-path');
  
  console.log('✅ Test data saved');
  
  // Test 2: Verify data exists
  console.log('🔍 Step 2: Verifying data exists...');
  const savedData = storage.getAvatarData();
  console.log('Saved avatar data:', savedData);
  
  if (savedData.favorites && savedData.lastSelectedId && savedData.globalState) {
    console.log('✅ All avatar data found');
  } else {
    console.log('❌ Some avatar data missing');
    return false;
  }
  
  // Test 3: Clear auth data (simulate logout)
  console.log('🧹 Step 3: Simulating logout (clearing auth data)...');
  storage.clearExceptPersistent();
  
  // Test 4: Verify avatar data persists but auth data is gone
  console.log('🔍 Step 4: Verifying persistence after logout...');
  const dataAfterLogout = storage.getAvatarData();
  
  const authToken = storage.getItem('supabase.auth.token');
  const sbToken = storage.getItem('sb-auth-token-123');
  const visitedPath = storage.getItem('lastVisitedPath');
  
  console.log('Avatar data after logout:', dataAfterLogout);
  console.log('Auth token after logout:', authToken);
  console.log('SB token after logout:', sbToken);
  console.log('Visited path after logout:', visitedPath);
  
  // Verify results
  const avatarDataPersisted = dataAfterLogout.favorites && 
                             dataAfterLogout.lastSelectedId && 
                             dataAfterLogout.globalState;
  
  const authDataCleared = !authToken && !sbToken && !visitedPath;
  
  if (avatarDataPersisted && authDataCleared) {
    console.log('🎉 SUCCESS: Avatar data persisted through logout!');
    console.log('✅ Avatar favorites preserved');
    console.log('✅ Last selected avatar preserved');
    console.log('✅ Global state (pin mode, positions) preserved');
    console.log('✅ Auth data properly cleared');
    return true;
  } else {
    console.log('❌ FAILURE: Storage persistence test failed');
    if (!avatarDataPersisted) {
      console.log('❌ Avatar data was lost');
    }
    if (!authDataCleared) {
      console.log('❌ Auth data was not properly cleared');
    }
    return false;
  }
};

// Auto-run test if in development
if (import.meta.env.DEV) {
  // console.log('🔧 Development mode detected - storage test available');
  // console.log('Run testStoragePersistence() in console to test avatar persistence');

  // Make it available globally for console testing
  window.testStoragePersistence = testStoragePersistence;
}
