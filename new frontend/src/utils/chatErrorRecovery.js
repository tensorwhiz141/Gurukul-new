/**
 * Chat Error Recovery Utility
 * Handles storage quota issues, corruption recovery, and fallback mechanisms
 */

import { toast } from 'react-hot-toast';
import { safeSetItem } from './storageManager';

// Error types
export const ERROR_TYPES = {
  QUOTA_EXCEEDED: 'QUOTA_EXCEEDED',
  STORAGE_CORRUPTED: 'STORAGE_CORRUPTED',
  SESSION_INVALID: 'SESSION_INVALID',
  NETWORK_ERROR: 'NETWORK_ERROR',
  UNKNOWN_ERROR: 'UNKNOWN_ERROR',
};

// Recovery strategies
export const RECOVERY_STRATEGIES = {
  CLEANUP_OLD_SESSIONS: 'CLEANUP_OLD_SESSIONS',
  COMPRESS_DATA: 'COMPRESS_DATA',
  RESET_STORAGE: 'RESET_STORAGE',
  FALLBACK_MODE: 'FALLBACK_MODE',
};

class ChatErrorRecovery {
  constructor() {
    this.recoveryAttempts = new Map();
    this.maxRecoveryAttempts = 3;
    this.fallbackMode = false;
    this.lastError = null;
  }

  /**
   * Handle storage quota exceeded error
   */
  async handleQuotaExceeded() {
    try {
      console.log('🚨 Storage quota exceeded, attempting recovery...');
      
      // Strategy 1: Clean up old sessions
      const cleanedSessions = await this.cleanupOldSessions();
      if (cleanedSessions > 0) {
        toast.success(`Cleaned up ${cleanedSessions} old sessions to free space`, {
          duration: 4000,
          icon: '🧹',
        });
        return true;
      }

      // Strategy 2: Compress existing data
      const compressionResult = await this.compressStorageData();
      if (compressionResult.success) {
        toast.success(`Compressed data, saved ${compressionResult.savedBytes} bytes`, {
          duration: 4000,
          icon: '📦',
        });
        return true;
      }

      // Strategy 3: Emergency cleanup - keep only current session
      const emergencyCleanup = await this.emergencyCleanup();
      if (emergencyCleanup) {
        toast.warning('Emergency cleanup performed. Some old chat history was removed.', {
          duration: 5000,
          icon: '⚠️',
        });
        return true;
      }

      // Strategy 4: Enable fallback mode
      this.enableFallbackMode();
      return false;

    } catch (error) {
      console.error('Failed to recover from quota exceeded:', error);
      this.enableFallbackMode();
      return false;
    }
  }

  /**
   * Clean up old sessions to free space
   */
  async cleanupOldSessions() {
    try {
      // Load sessions directly from localStorage to avoid circular dependency
      const stored = localStorage.getItem('gurukul_chat_sessions');
      if (!stored) return 0;

      const sessions = JSON.parse(stored);
      const sessionKeys = Object.keys(sessions);

      if (sessionKeys.length <= 5) {
        return 0; // Don't clean if we have 5 or fewer sessions
      }

      // Sort by last updated and keep only the 5 most recent
      const sortedSessions = sessionKeys
        .map(key => ({ key, lastUpdated: sessions[key].lastUpdated }))
        .sort((a, b) => new Date(b.lastUpdated) - new Date(a.lastUpdated))
        .slice(0, 5);

      const cleanedSessions = {};
      sortedSessions.forEach(({ key }) => {
        cleanedSessions[key] = sessions[key];
      });

      // Save directly to localStorage
      const result = await safeSetItem('gurukul_chat_sessions', JSON.stringify(cleanedSessions));

      if (result.success) {
        const removedCount = sessionKeys.length - 5;
        console.log(`🧹 Cleaned up ${removedCount} old sessions`);
        return removedCount;
      }

      return 0;
    } catch (error) {
      console.error('Failed to cleanup old sessions:', error);
      return 0;
    }
  }

  /**
   * Compress storage data
   */
  async compressStorageData() {
    try {
      // Load sessions directly from localStorage
      const stored = localStorage.getItem('gurukul_chat_sessions');
      if (!stored) return { success: false, savedBytes: 0 };

      const sessions = JSON.parse(stored);
      let savedBytes = 0;
      let compressedSessions = {};

      // Compress each session
      for (const [sessionId, sessionData] of Object.entries(sessions)) {
        const originalSize = JSON.stringify(sessionData).length;

        // Compress messages by removing unnecessary data
        const compressedMessages = sessionData.messages?.map(msg => ({
          id: msg.id,
          role: msg.role,
          content: typeof msg.content === 'string' ? msg.content.trim() : msg.content,
          model: msg.model,
          timestamp: msg.timestamp,
          ...(msg.isWelcome && { isWelcome: true }),
          ...(msg.isError && { isError: true }),
        })) || [];

        const compressedSession = {
          messages: compressedMessages,
          lastUpdated: sessionData.lastUpdated,
          messageCount: compressedMessages.length,
        };

        const compressedSize = JSON.stringify(compressedSession).length;
        savedBytes += originalSize - compressedSize;

        compressedSessions[sessionId] = compressedSession;
      }

      if (savedBytes > 0) {
        const result = await safeSetItem('gurukul_chat_sessions', JSON.stringify(compressedSessions));
        return { success: result.success, savedBytes };
      }

      return { success: false, savedBytes: 0 };
    } catch (error) {
      console.error('Failed to compress storage data:', error);
      return { success: false, savedBytes: 0 };
    }
  }

  /**
   * Emergency cleanup - keep only current session
   */
  async emergencyCleanup() {
    try {
      // Load sessions directly from localStorage
      const stored = localStorage.getItem('gurukul_chat_sessions');
      if (!stored) return false;

      const sessions = JSON.parse(stored);
      const sessionKeys = Object.keys(sessions);

      if (sessionKeys.length === 0) return false;

      // Keep only the most recent session
      const sortedSessions = sessionKeys
        .map(key => ({ key, lastUpdated: sessions[key].lastUpdated }))
        .sort((a, b) => new Date(b.lastUpdated) - new Date(a.lastUpdated));

      if (sortedSessions.length > 0) {
        const mostRecentKey = sortedSessions[0].key;
        const emergencySessions = {
          [mostRecentKey]: sessions[mostRecentKey],
        };

        const result = await safeSetItem('gurukul_chat_sessions', JSON.stringify(emergencySessions));

        if (result.success) {
          console.log('🚨 Emergency cleanup completed, kept only most recent session');
          return true;
        }
      }

      return false;
    } catch (error) {
      console.error('Failed to perform emergency cleanup:', error);
      return false;
    }
  }

  /**
   * Enable fallback mode (in-memory only)
   */
  enableFallbackMode() {
    this.fallbackMode = true;
    console.log('🔄 Enabled fallback mode - chat will work in memory only');
    
    toast.error('Storage limit reached. Chat will work in memory only for this session.', {
      duration: 6000,
      icon: '⚠️',
    });
  }

  /**
   * Handle corrupted storage
   */
  async handleCorruptedStorage() {
    try {
      console.log('🔧 Attempting to recover corrupted storage...');

      // Try to backup what we can
      const backupData = this.createEmergencyBackup();

      // Clear corrupted storage
      localStorage.removeItem('gurukul_chat_sessions');
      localStorage.removeItem('gurukul_chat_settings');

      // Try to restore from backup
      if (backupData) {
        await this.restoreFromEmergencyBackup(backupData);
      }

      toast.warning('Storage was corrupted and has been reset. Some chat history may be lost.', {
        duration: 5000,
        icon: '🔧',
      });

      return true;
    } catch (error) {
      console.error('Failed to recover from corrupted storage:', error);
      this.enableFallbackMode();
      return false;
    }
  }

  /**
   * Create emergency backup of current session
   */
  createEmergencyBackup() {
    try {
      // Load sessions directly from localStorage
      const stored = localStorage.getItem('gurukul_chat_sessions');
      if (!stored) return null;

      const sessions = JSON.parse(stored);
      const sessionKeys = Object.keys(sessions);

      if (sessionKeys.length > 0) {
        // Get the most recent session as backup
        const sortedSessions = sessionKeys
          .map(key => ({ key, lastUpdated: sessions[key].lastUpdated }))
          .sort((a, b) => new Date(b.lastUpdated) - new Date(a.lastUpdated));

        const mostRecentKey = sortedSessions[0].key;

        return {
          currentSession: sessions[mostRecentKey],
          sessionId: mostRecentKey,
          timestamp: new Date().toISOString(),
        };
      }
    } catch (error) {
      console.error('Failed to create emergency backup:', error);
    }
    return null;
  }

  /**
   * Restore from emergency backup
   */
  async restoreFromEmergencyBackup(backupData) {
    try {
      if (backupData && backupData.currentSession && backupData.sessionId) {
        const sessions = { [backupData.sessionId]: backupData.currentSession };
        const result = await safeSetItem('gurukul_chat_sessions', JSON.stringify(sessions));

        if (result.success) {
          console.log('✅ Restored from emergency backup');
          return true;
        }
      }
    } catch (error) {
      console.error('Failed to restore from emergency backup:', error);
    }
    return false;
  }

  /**
   * Handle general errors with recovery strategies
   */
  async handleError(error, context = 'unknown') {
    const errorKey = `${context}_${error.name || 'unknown'}`;
    const attempts = this.recoveryAttempts.get(errorKey) || 0;

    if (attempts >= this.maxRecoveryAttempts) {
      console.log(`Max recovery attempts reached for ${errorKey}, enabling fallback mode`);
      this.enableFallbackMode();
      return false;
    }

    this.recoveryAttempts.set(errorKey, attempts + 1);
    this.lastError = { error, context, attempts: attempts + 1 };

    // Determine error type and recovery strategy
    if (error.name === 'QuotaExceededError' || error.message?.includes('quota')) {
      return await this.handleQuotaExceeded();
    }

    if (error.message?.includes('corrupt') || error.name === 'SyntaxError') {
      return await this.handleCorruptedStorage();
    }

    // Generic error handling
    console.warn(`Unhandled error in ${context}:`, error);
    return false;
  }

  /**
   * Reset recovery state
   */
  resetRecoveryState() {
    this.recoveryAttempts.clear();
    this.fallbackMode = false;
    this.lastError = null;
  }

  /**
   * Get recovery status
   */
  getRecoveryStatus() {
    return {
      fallbackMode: this.fallbackMode,
      lastError: this.lastError,
      recoveryAttempts: Object.fromEntries(this.recoveryAttempts),
    };
  }
}

// Create singleton instance
const chatErrorRecovery = new ChatErrorRecovery();

export default chatErrorRecovery;
