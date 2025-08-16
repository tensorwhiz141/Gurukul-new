/**
 * Avatar Debug Utilities
 * Helper functions to debug avatar loading issues
 */

export const testAvatarPath = async (path) => {
  console.log(`🧪 Testing avatar path: ${path}`);
  
  try {
    // Test 1: Basic fetch
    const response = await fetch(path);
    console.log(`📡 Fetch response:`, {
      status: response.status,
      statusText: response.statusText,
      ok: response.ok,
      headers: Object.fromEntries(response.headers.entries())
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    // Test 2: Check content type
    const contentType = response.headers.get('content-type');
    console.log(`📄 Content type: ${contentType}`);
    
    // Test 3: Check file size
    const contentLength = response.headers.get('content-length');
    if (contentLength) {
      const sizeKB = (parseInt(contentLength) / 1024).toFixed(2);
      console.log(`📏 File size: ${sizeKB} KB`);
    }
    
    // Test 4: Try to read as blob
    const blob = await response.blob();
    console.log(`💾 Blob created:`, {
      size: blob.size,
      type: blob.type
    });
    
    return {
      success: true,
      status: response.status,
      contentType,
      size: blob.size,
      blob
    };
    
  } catch (error) {
    console.error(`❌ Avatar path test failed for ${path}:`, error);
    return {
      success: false,
      error: error.message
    };
  }
};

export const testGLTFLoading = async (path) => {
  console.log(`🎭 Testing GLTF loading for: ${path}`);
  
  try {
    // Import GLTFLoader dynamically
    const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');
    const loader = new GLTFLoader();
    
    return new Promise((resolve, reject) => {
      loader.load(
        path,
        (gltf) => {
          console.log(`✅ GLTF loaded successfully:`, {
            scene: !!gltf.scene,
            animations: gltf.animations?.length || 0,
            children: gltf.scene?.children?.length || 0
          });
          resolve({
            success: true,
            gltf,
            hasScene: !!gltf.scene,
            animationCount: gltf.animations?.length || 0,
            childrenCount: gltf.scene?.children?.length || 0
          });
        },
        (progress) => {
          console.log(`📊 GLTF loading progress:`, {
            loaded: progress.loaded,
            total: progress.total,
            percentage: progress.total ? (progress.loaded / progress.total * 100).toFixed(2) + '%' : 'unknown'
          });
        },
        (error) => {
          console.error(`❌ GLTF loading failed:`, error);
          reject({
            success: false,
            error: error.message || 'Unknown GLTF loading error'
          });
        }
      );
    });
    
  } catch (error) {
    console.error(`❌ GLTF test setup failed:`, error);
    return {
      success: false,
      error: error.message
    };
  }
};

export const runFullAvatarTest = async (path = '/avatar/jupiter.glb') => {
  console.group(`🔍 Full Avatar Test: ${path}`);
  
  const results = {
    path,
    timestamp: new Date().toISOString(),
    tests: {}
  };
  
  // Test 1: Basic path access
  console.log('🧪 Test 1: Basic path access');
  results.tests.pathAccess = await testAvatarPath(path);
  
  // Test 2: GLTF loading (only if path access succeeded)
  if (results.tests.pathAccess.success) {
    console.log('🧪 Test 2: GLTF loading');
    try {
      results.tests.gltfLoading = await testGLTFLoading(path);
    } catch (error) {
      results.tests.gltfLoading = {
        success: false,
        error: error.message || error
      };
    }
  } else {
    console.log('⏭️ Skipping GLTF test due to path access failure');
    results.tests.gltfLoading = {
      success: false,
      error: 'Skipped due to path access failure'
    };
  }
  
  // Test 3: Alternative paths
  console.log('🧪 Test 3: Alternative paths');
  const alternativePaths = [
    './avatar/jupiter.glb',
    'avatar/jupiter.glb',
    '/public/avatar/jupiter.glb',
    `${window.location.origin}/avatar/jupiter.glb`
  ];
  
  results.tests.alternativePaths = {};
  for (const altPath of alternativePaths) {
    if (altPath !== path) {
      results.tests.alternativePaths[altPath] = await testAvatarPath(altPath);
    }
  }
  
  // Summary
  console.log('📋 Test Summary:');
  console.table({
    'Path Access': results.tests.pathAccess.success ? '✅ Pass' : '❌ Fail',
    'GLTF Loading': results.tests.gltfLoading.success ? '✅ Pass' : '❌ Fail',
    'Alternative Paths': Object.values(results.tests.alternativePaths).some(r => r.success) ? '✅ Some Pass' : '❌ All Fail'
  });
  
  if (!results.tests.pathAccess.success) {
    console.error('🚨 Primary issue: Cannot access avatar file at specified path');
    console.error('💡 Suggestions:');
    console.error('   - Check if file exists in public/avatar/jupiter.glb');
    console.error('   - Verify file permissions');
    console.error('   - Check for typos in filename');
  } else if (!results.tests.gltfLoading.success) {
    console.error('🚨 Primary issue: File accessible but GLTF loading failed');
    console.error('💡 Suggestions:');
    console.error('   - Check if file is a valid GLTF/GLB format');
    console.error('   - Try opening file in a GLTF viewer');
    console.error('   - Check for file corruption');
  } else {
    console.log('🎉 All tests passed! Avatar should load correctly.');
  }
  
  console.groupEnd();
  return results;
};

// Auto-run test in development
if (import.meta.env.DEV) {
  // Add global function for manual testing
  window.testAvatar = runFullAvatarTest;
  console.log('🔧 Avatar debug utilities loaded. Run window.testAvatar() to test avatar loading.');
}
