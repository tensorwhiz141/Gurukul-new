/**
 * Pin Mode Test Utilities
 * Helper functions to test and debug pin mode functionality
 */

/**
 * Create a test jupiter avatar for pin mode testing
 */
export function createTestJupiterAvatar() {
  return {
    id: "test-jupiter-" + Date.now(),
    name: "Brihaspati",
    isDefault: true,
    previewUrl: "/avatar/jupiter.glb",
    timestamp: new Date().toISOString(),
    gridPosition: { x: 0, y: 0, z: 0 },
    gridRotation: { x: 0, y: 180, z: 0 },
    gridScale: 1,
    pinPosition: { x: 0, y: 0.5, z: -4.0 },
    pinRotation: { x: 0, y: 180, z: 0 },
    pinScale: 0.6,
    isPinModeEnabled: true,
    pinnedAvatarPosition: { x: 100, y: 100 }
  };
}

/**
 * Test pin mode state consistency
 */
export function testPinModeState(store) {
  const state = store.getState();
  const avatar = state.avatar;
  
  console.log("🧪 Pin Mode State Test:", {
    isPinModeEnabled: avatar.isPinModeEnabled,
    selectedAvatar: avatar.selectedAvatar?.name || "none",
    favoritesCount: avatar.favorites?.length || 0,
    pinPosition: avatar.pinPosition,
    pinRotation: avatar.pinRotation,
    pinScale: avatar.pinScale,
    pinnedAvatarPosition: avatar.pinnedAvatarPosition,
    activeSettingsTab: avatar.activeSettingsTab,
    activeMainTab: avatar.activeMainTab
  });
  
  // Check for common issues
  const issues = [];
  
  if (avatar.isPinModeEnabled && !avatar.selectedAvatar && (!avatar.favorites || avatar.favorites.length === 0)) {
    issues.push("Pin mode enabled but no avatar selected and no favorites");
  }
  
  if (avatar.isPinModeEnabled && avatar.pinScale === 0) {
    issues.push("Pin scale is 0 - avatar may be invisible");
  }
  
  if (avatar.isPinModeEnabled && !avatar.pinnedAvatarPosition) {
    issues.push("Pin mode enabled but no screen position set");
  }
  
  if (issues.length > 0) {
    console.warn("🚨 Pin Mode Issues Found:", issues);
  } else {
    console.log("✅ Pin Mode State looks good!");
  }
  
  return { state: avatar, issues };
}

/**
 * Force enable pin mode with jupiter avatar
 */
export function forceEnablePinMode(dispatch) {
  console.log("🔧 Force enabling pin mode with jupiter avatar");

  const jupiterAvatar = createTestJupiterAvatar();
  
  // Import actions dynamically to avoid circular dependencies
  import('../store/avatarSlice').then(({ 
    setSelectedAvatar, 
    addFavorite, 
    setIsPinModeEnabled, 
    setPinnedAvatarPosition,
    setPinPosition,
    setPinRotation,
    setPinScale,
    loadAvatarSettings
  }) => {
    // Set the jupiter avatar
    dispatch(setSelectedAvatar(jupiterAvatar));
    dispatch(addFavorite(jupiterAvatar));

    // Enable pin mode
    dispatch(setIsPinModeEnabled(true));

    // Set pin settings with better defaults for jupiter.glb
    dispatch(setPinnedAvatarPosition({ x: 100, y: 100 }));
    dispatch(setPinPosition({ x: 0, y: 0.5, z: -4.0 })); // Perfect position for jupiter.glb
    dispatch(setPinRotation({ x: 0, y: 180, z: 0 }));
    dispatch(setPinScale(0.6));

    // Load avatar settings
    dispatch(loadAvatarSettings(jupiterAvatar));

    console.log("✅ Pin mode force enabled with jupiter avatar");
  });
}

/**
 * Check if jupiter.glb file exists and is accessible
 */
export async function checkJupiterAvatarFile() {
  try {
    console.log("🧪 Testing jupiter.glb accessibility...");

    // Test 1: HEAD request to check if file exists
    const headResponse = await fetch('/avatar/jupiter.glb', { method: 'HEAD' });
    console.log("🧪 HEAD response:", {
      status: headResponse.status,
      ok: headResponse.ok,
      contentType: headResponse.headers.get('content-type'),
      contentLength: headResponse.headers.get('content-length')
    });

    if (!headResponse.ok) {
      console.error("❌ Jupiter avatar file not found:", headResponse.status);
      return false;
    }

    // Test 2: GET request to check if file can be downloaded
    const getResponse = await fetch('/avatar/jupiter.glb');
    console.log("🧪 GET response:", {
      status: getResponse.status,
      ok: getResponse.ok,
      contentType: getResponse.headers.get('content-type'),
      contentLength: getResponse.headers.get('content-length')
    });

    if (getResponse.ok) {
      const blob = await getResponse.blob();
      console.log("🧪 File blob:", {
        size: blob.size,
        type: blob.type
      });

      // Test 3: Try to create object URL
      const objectUrl = URL.createObjectURL(blob);
      console.log("🧪 Object URL created:", objectUrl);
      URL.revokeObjectURL(objectUrl);

      console.log("✅ Fallback avatar file is fully accessible");
      return true;
    } else {
      console.error("❌ Failed to download fallback avatar file:", getResponse.status);
      return false;
    }
  } catch (error) {
    console.error("❌ Error checking fallback avatar file:", error);
    return false;
  }
}

/**
 * Debug pin mode visibility conditions
 */
export function debugPinModeVisibility(location, avatar, auth) {
  const isOnAvatarSelectionPage = location.pathname === "/avatar-selection";
  const isOnAvatarSelectionPinTab = isOnAvatarSelectionPage && 
    avatar.activeMainTab === "favorites" && 
    avatar.activeSettingsTab === "pin";
  
  const shouldShowPinnedAvatar = auth.isAuthenticated && 
    avatar.isPinModeEnabled && 
    !isOnAvatarSelectionPinTab;
    
  const shouldShowContainedAvatar = auth.isAuthenticated && 
    avatar.isPinModeEnabled && 
    isOnAvatarSelectionPinTab;
  
  console.log("🔍 Pin Mode Visibility Debug:", {
    location: location.pathname,
    isAuthenticated: auth.isAuthenticated,
    isPinModeEnabled: avatar.isPinModeEnabled,
    activeMainTab: avatar.activeMainTab,
    activeSettingsTab: avatar.activeSettingsTab,
    isOnAvatarSelectionPage,
    isOnAvatarSelectionPinTab,
    shouldShowPinnedAvatar,
    shouldShowContainedAvatar,
    selectedAvatar: avatar.selectedAvatar?.name || "none",
    favoritesCount: avatar.favorites?.length || 0
  });
  
  return {
    shouldShowPinnedAvatar,
    shouldShowContainedAvatar,
    conditions: {
      isOnAvatarSelectionPage,
      isOnAvatarSelectionPinTab
    }
  };
}

/**
 * Test 3D model loading with Three.js GLTFLoader
 */
export async function test3DModelLoading() {
  console.log("🧪 Testing 3D model loading with Three.js...");

  try {
    // Dynamic import to avoid bundling issues
    const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');
    const loader = new GLTFLoader();

    console.log("🧪 GLTFLoader imported successfully");

    // Test loading jupiter.glb
    return new Promise((resolve, reject) => {
      const startTime = Date.now();

      loader.load(
        '/avatar/jupiter.glb',
        (gltf) => {
          const loadTime = Date.now() - startTime;
          console.log("✅ 3D model loaded successfully!", {
            loadTime: `${loadTime}ms`,
            scene: gltf.scene,
            animations: gltf.animations?.length || 0,
            cameras: gltf.cameras?.length || 0,
            scenes: gltf.scenes?.length || 0
          });

          // Test scene properties
          if (gltf.scene) {
            console.log("🧪 Scene details:", {
              children: gltf.scene.children.length,
              type: gltf.scene.type,
              uuid: gltf.scene.uuid
            });
          }

          resolve(true);
        },
        (progress) => {
          console.log("🧪 Loading progress:", {
            loaded: progress.loaded,
            total: progress.total,
            percentage: progress.total ? Math.round((progress.loaded / progress.total) * 100) + '%' : 'unknown'
          });
        },
        (error) => {
          console.error("❌ 3D model loading failed:", error);
          reject(error);
        }
      );
    });
  } catch (error) {
    console.error("❌ Failed to import GLTFLoader:", error);
    return false;
  }
}

// Make functions available globally in development
if (import.meta.env.DEV) {
  window.pinModeTestUtils = {
    createTestFallbackAvatar,
    testPinModeState,
    forceEnablePinMode,
    checkFallbackAvatarFile,
    debugPinModeVisibility,
    test3DModelLoading
  };
}
