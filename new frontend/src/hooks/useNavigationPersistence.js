/**
 * Navigation Persistence Hook
 * Ensures chat history and other data persists across page navigation
 */

import { useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import chatHistoryStorage from '../utils/chatHistoryStorage';

/**
 * Hook to handle chat history persistence across navigation
 */
export const useNavigationPersistence = () => {
  const location = useLocation();
  const navigate = useNavigate();

  /**
   * Save current state before navigation
   */
  const saveStateBeforeNavigation = useCallback(async () => {
    try {
      // Force save current chat state
      if (chatHistoryStorage.isInitialized) {
        const currentMessages = chatHistoryStorage.loadChatHistory();
        await chatHistoryStorage.saveChatHistory(currentMessages);
        
        // Save current location for potential return
        localStorage.setItem('gurukul_last_chat_location', location.pathname);
        
        console.log('💾 Chat state saved before navigation');
      }
    } catch (error) {
      console.error('Failed to save state before navigation:', error);
    }
  }, [location.pathname]);

  /**
   * Restore state after navigation
   */
  const restoreStateAfterNavigation = useCallback(async () => {
    try {
      // Check if we're returning to chatbot from another page
      const lastLocation = localStorage.getItem('gurukul_last_chat_location');
      
      if (location.pathname === '/chatbot' && lastLocation && lastLocation !== '/chatbot') {
        console.log('🔄 Returning to chatbot, ensuring history is loaded');
        
        // Small delay to ensure component is mounted
        setTimeout(async () => {
          if (chatHistoryStorage.isInitialized) {
            // Force reload chat history to ensure it's fresh
            const history = chatHistoryStorage.loadChatHistory();
            console.log('📚 Restored chat history with', history.length, 'messages');
          }
        }, 100);
      }
    } catch (error) {
      console.error('Failed to restore state after navigation:', error);
    }
  }, [location.pathname]);

  // Handle page visibility changes (when user switches tabs/windows)
  const handleVisibilityChange = useCallback(async () => {
    if (document.visibilityState === 'hidden') {
      // Page is being hidden, save current state
      await saveStateBeforeNavigation();
    } else if (document.visibilityState === 'visible') {
      // Page is becoming visible, restore state if needed
      await restoreStateAfterNavigation();
    }
  }, [saveStateBeforeNavigation, restoreStateAfterNavigation]);

  // Handle beforeunload (when user closes tab/refreshes)
  const handleBeforeUnload = useCallback(async () => {
    await saveStateBeforeNavigation();
  }, [saveStateBeforeNavigation]);

  // Set up event listeners
  useEffect(() => {
    // Save state when navigating away from chatbot
    if (location.pathname !== '/chatbot') {
      saveStateBeforeNavigation();
    }

    // Restore state when navigating to chatbot
    if (location.pathname === '/chatbot') {
      restoreStateAfterNavigation();
    }
  }, [location.pathname, saveStateBeforeNavigation, restoreStateAfterNavigation]);

  // Set up visibility and unload listeners
  useEffect(() => {
    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [handleVisibilityChange, handleBeforeUnload]);

  return {
    saveStateBeforeNavigation,
    restoreStateAfterNavigation,
  };
};

/**
 * Hook to handle chat state persistence during auth changes
 */
export const useAuthPersistence = () => {
  /**
   * Handle user login - migrate guest chat to user account
   */
  const handleUserLogin = useCallback(async (newUserId) => {
    try {
      console.log('👤 User logged in, migrating chat history...');
      
      // Get current guest session data
      const guestSessions = chatHistoryStorage.loadSessions();
      const guestSessionKeys = Object.keys(guestSessions).filter(key => 
        key.startsWith('guest-user_')
      );
      
      if (guestSessionKeys.length > 0) {
        // Migrate guest sessions to user account
        const userSessions = {};
        
        guestSessionKeys.forEach(guestKey => {
          const sessionData = guestSessions[guestKey];
          const datePart = guestKey.split('_').slice(1).join('_'); // Everything after 'guest-user_'
          const newKey = `${newUserId}_${datePart}`;
          userSessions[newKey] = sessionData;
        });
        
        // Save migrated sessions
        const allSessions = { ...guestSessions, ...userSessions };
        await chatHistoryStorage.saveSessions(allSessions);
        
        // Switch to the most recent user session
        const userSessionKeys = Object.keys(userSessions);
        if (userSessionKeys.length > 0) {
          const mostRecentSession = userSessionKeys.sort().pop();
          await chatHistoryStorage.switchToSession(mostRecentSession);
        }
        
        console.log(`✅ Migrated ${guestSessionKeys.length} guest sessions to user account`);
      }
      
      // Re-initialize with new user ID
      await chatHistoryStorage.init(newUserId);
      
    } catch (error) {
      console.error('Failed to handle user login:', error);
    }
  }, []);

  /**
   * Handle user logout - keep chat history but switch to guest mode
   */
  const handleUserLogout = useCallback(async () => {
    try {
      console.log('👋 User logged out, switching to guest mode...');
      
      // Save current state before switching
      const currentMessages = chatHistoryStorage.loadChatHistory();
      await chatHistoryStorage.saveChatHistory(currentMessages);
      
      // Re-initialize as guest user
      await chatHistoryStorage.init('guest-user');
      
      console.log('✅ Switched to guest mode, chat history preserved');
      
    } catch (error) {
      console.error('Failed to handle user logout:', error);
    }
  }, []);

  return {
    handleUserLogin,
    handleUserLogout,
  };
};

/**
 * Hook to handle storage quota management for chat history
 */
export const useChatStorageQuota = () => {
  /**
   * Check storage quota and clean up if needed
   */
  const checkAndCleanupStorage = useCallback(async () => {
    try {
      const stats = chatHistoryStorage.getChatStats();
      const storageSize = stats.storageSize;
      
      // If storage is getting large (>5MB), trigger cleanup
      if (storageSize > 5 * 1024 * 1024) {
        console.log('🧹 Storage quota getting high, triggering cleanup...');
        
        // Get all sessions and remove oldest ones
        const sessions = chatHistoryStorage.loadSessions();
        const sessionKeys = Object.keys(sessions);
        
        if (sessionKeys.length > 20) {
          // Sort by last updated and keep only the 15 most recent
          const sortedSessions = sessionKeys
            .map(key => ({ key, lastUpdated: sessions[key].lastUpdated }))
            .sort((a, b) => new Date(b.lastUpdated) - new Date(a.lastUpdated))
            .slice(0, 15);
          
          const cleanedSessions = {};
          sortedSessions.forEach(({ key }) => {
            cleanedSessions[key] = sessions[key];
          });
          
          await chatHistoryStorage.saveSessions(cleanedSessions);
          
          const removedCount = sessionKeys.length - 15;
          console.log(`🧹 Cleaned up ${removedCount} old chat sessions`);
          
          return removedCount;
        }
      }
      
      return 0;
    } catch (error) {
      console.error('Failed to check and cleanup storage:', error);
      return 0;
    }
  }, []);

  /**
   * Get storage usage information
   */
  const getStorageInfo = useCallback(() => {
    try {
      const stats = chatHistoryStorage.getChatStats();
      const totalStorage = 10 * 1024 * 1024; // Assume 10MB limit for localStorage
      const usedPercentage = (stats.storageSize / totalStorage) * 100;
      
      return {
        used: stats.storageSize,
        total: totalStorage,
        percentage: Math.min(usedPercentage, 100),
        isNearLimit: usedPercentage > 80,
      };
    } catch (error) {
      console.error('Failed to get storage info:', error);
      return {
        used: 0,
        total: 10 * 1024 * 1024,
        percentage: 0,
        isNearLimit: false,
      };
    }
  }, []);

  return {
    checkAndCleanupStorage,
    getStorageInfo,
  };
};
