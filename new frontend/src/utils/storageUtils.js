/**
 * Storage utilities for managing localStorage with selective clearing
 * Preserves avatar data during authentication state changes
 */

// Keys that should be preserved during logout
const PERSISTENT_KEYS = [
  'gurukul_favorite_avatars',
  'gurukul_last_selected_avatar',
  'gurukul_avatar_global_state',
  'gurukul_custom_models', // Custom uploaded .glb models
  'gurukul_settings', // Keep user settings too
  'gurukul_chat_history', // Chat history storage
  'gurukul_chat_sessions', // Chat sessions storage
  'gurukul_chat_settings', // Chat settings
  'selectedAIModel', // Selected AI model preference
];

// Keys that should be cleared during logout (auth-related)
const AUTH_KEYS = [
  'supabase.auth.token',
  'sb-auth-token',
  'lastVisitedPath',
];

/**
 * Safe localStorage operations with error handling
 */
export const storage = {
  /**
   * Get item from localStorage with error handling
   */
  getItem: (key) => {
    try {
      return localStorage.getItem(key);
    } catch (error) {
      console.error(`Error reading from localStorage (${key}):`, error);
      return null;
    }
  },

  /**
   * Set item in localStorage with error handling
   */
  setItem: (key, value) => {
    try {
      localStorage.setItem(key, value);
      return true;
    } catch (error) {
      console.error(`Error writing to localStorage (${key}):`, error);
      return false;
    }
  },

  /**
   * Remove item from localStorage with error handling
   */
  removeItem: (key) => {
    try {
      localStorage.removeItem(key);
      return true;
    } catch (error) {
      console.error(`Error removing from localStorage (${key}):`, error);
      return false;
    }
  },

  /**
   * Clear all localStorage except persistent keys
   */
  clearExceptPersistent: () => {
    try {
      // Get all persistent data before clearing
      const persistentData = {};
      PERSISTENT_KEYS.forEach(key => {
        const value = localStorage.getItem(key);
        if (value !== null) {
          persistentData[key] = value;
        }
      });

      // Clear all localStorage
      localStorage.clear();

      // Restore persistent data
      Object.entries(persistentData).forEach(([key, value]) => {
        localStorage.setItem(key, value);
      });

      console.log('localStorage cleared except persistent keys:', PERSISTENT_KEYS);
      return true;
    } catch (error) {
      console.error('Error clearing localStorage selectively:', error);
      return false;
    }
  },

  /**
   * Clear only authentication-related keys
   */
  clearAuthKeys: () => {
    try {
      AUTH_KEYS.forEach(key => {
        localStorage.removeItem(key);
        sessionStorage.removeItem(key);
      });

      // Clear any keys that start with 'sb-' (Supabase auth tokens)
      Object.keys(localStorage).forEach(key => {
        if (key.startsWith('sb-')) {
          localStorage.removeItem(key);
        }
      });

      Object.keys(sessionStorage).forEach(key => {
        if (key.startsWith('sb-')) {
          sessionStorage.removeItem(key);
        }
      });

      console.log('Authentication keys cleared');
      return true;
    } catch (error) {
      console.error('Error clearing auth keys:', error);
      return false;
    }
  },

  /**
   * Get all avatar-related data
   */
  getAvatarData: () => {
    try {
      const favorites = storage.getItem('gurukul_favorite_avatars');
      const lastSelected = storage.getItem('gurukul_last_selected_avatar');
      const globalState = storage.getItem('gurukul_avatar_global_state');

      return {
        favorites: favorites ? JSON.parse(favorites) : null,
        lastSelectedId: lastSelected,
        globalState: globalState ? JSON.parse(globalState) : null,
      };
    } catch (error) {
      console.error('Error getting avatar data:', error);
      return {
        favorites: null,
        lastSelectedId: null,
        globalState: null,
      };
    }
  },

  /**
   * Save avatar global state (pin mode, settings tab, etc.)
   * Note: Explicitly excludes chat history to prevent duplication issues
   */
  saveAvatarGlobalState: (state) => {
    try {
      const globalState = {
        isPinModeEnabled: state.isPinModeEnabled,
        pinnedAvatarPosition: state.pinnedAvatarPosition,
        activeSettingsTab: state.activeSettingsTab,
        activeMainTab: state.activeMainTab,
        gridPosition: state.gridPosition,
        gridRotation: state.gridRotation,
        gridScale: state.gridScale,
        pinPosition: state.pinPosition,
        pinRotation: state.pinRotation,
        pinScale: state.pinScale,
        lastUpdated: new Date().toISOString(),
        // Explicitly exclude chat-related data to prevent duplication
        // chatHistory: intentionally omitted
        // isChatOpen: intentionally omitted (should not persist)
      };

      return storage.setItem('gurukul_avatar_global_state', JSON.stringify(globalState));
    } catch (error) {
      console.error('Error saving avatar global state:', error);
      return false;
    }
  },

  /**
   * Save custom models to localStorage
   */
  saveCustomModels: (models) => {
    try {
      return storage.setItem('gurukul_custom_models', JSON.stringify(models));
    } catch (error) {
      console.error('Error saving custom models:', error);
      return false;
    }
  },

  /**
   * Load custom models from localStorage
   */
  loadCustomModels: () => {
    try {
      const models = storage.getItem('gurukul_custom_models');
      return models ? JSON.parse(models) : [];
    } catch (error) {
      console.error('Error loading custom models:', error);
      return [];
    }
  },

  /**
   * Save .glb file data to localStorage (Base64 encoded)
   */
  saveGlbFile: (fileId, fileData, metadata) => {
    try {
      const fileKey = `gurukul_glb_${fileId}`;
      const fileInfo = {
        data: fileData, // Base64 encoded file data
        metadata: metadata,
        savedAt: new Date().toISOString(),
      };
      return storage.setItem(fileKey, JSON.stringify(fileInfo));
    } catch (error) {
      console.error('Error saving GLB file:', error);
      return false;
    }
  },

  /**
   * Load .glb file data from localStorage
   */
  loadGlbFile: (fileId) => {
    try {
      const fileKey = `gurukul_glb_${fileId}`;
      const fileInfo = storage.getItem(fileKey);
      return fileInfo ? JSON.parse(fileInfo) : null;
    } catch (error) {
      console.error('Error loading GLB file:', error);
      return null;
    }
  },

  /**
   * Delete .glb file from localStorage
   */
  deleteGlbFile: (fileId) => {
    try {
      const fileKey = `gurukul_glb_${fileId}`;
      storage.removeItem(fileKey);
      return true;
    } catch (error) {
      console.error('Error deleting GLB file:', error);
      return false;
    }
  },

  /**
   * Get storage usage information
   */
  getStorageInfo: () => {
    try {
      let totalSize = 0;
      let customModelsSize = 0;

      for (let key in localStorage) {
        if (localStorage.hasOwnProperty(key)) {
          const size = localStorage[key].length;
          totalSize += size;

          if (key.startsWith('gurukul_glb_') || key === 'gurukul_custom_models') {
            customModelsSize += size;
          }
        }
      }

      // Convert to MB (approximate)
      return {
        totalSizeMB: (totalSize / 1024 / 1024).toFixed(2),
        customModelsSizeMB: (customModelsSize / 1024 / 1024).toFixed(2),
        totalItems: Object.keys(localStorage).length,
      };
    } catch (error) {
      console.error('Error getting storage info:', error);
      return {
        totalSizeMB: '0',
        customModelsSizeMB: '0',
        totalItems: 0,
      };
    }
  },
};

export default storage;
