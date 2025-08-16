/**
 * Chat Session Manager Component
 * Provides session management functionality for the chatbot
 */

import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { 
  Plus, 
  MessageSquare, 
  Calendar, 
  Trash2, 
  MoreVertical,
  Clock,
  ChevronRight 
} from 'lucide-react';

const ChatSessionManager = ({ 
  getAllSessions, 
  switchToSession, 
  createNewSession, 
  deleteSession, 
  getCurrentSessionInfo,
  onSessionChange 
}) => {
  const { t } = useTranslation();
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [showDropdown, setShowDropdown] = useState({});

  // Load sessions on mount and when dependencies change
  useEffect(() => {
    if (getAllSessions) {
      const allSessions = getAllSessions();
      setSessions(allSessions);
      
      const current = getCurrentSessionInfo();
      setCurrentSession(current);
    }
  }, [getAllSessions, getCurrentSessionInfo]);

  const handleCreateNewSession = async () => {
    const newSessionId = await createNewSession();
    if (newSessionId) {
      // Refresh sessions list
      const allSessions = getAllSessions();
      setSessions(allSessions);
      
      const current = getCurrentSessionInfo();
      setCurrentSession(current);
      
      if (onSessionChange) {
        onSessionChange(newSessionId);
      }
    }
  };

  const handleSwitchSession = async (sessionId) => {
    const success = await switchToSession(sessionId);
    if (success) {
      // Refresh sessions list and current session
      const allSessions = getAllSessions();
      setSessions(allSessions);
      
      const current = getCurrentSessionInfo();
      setCurrentSession(current);
      
      if (onSessionChange) {
        onSessionChange(sessionId);
      }
    }
  };

  const handleDeleteSession = async (sessionId, event) => {
    event.stopPropagation();
    
    const success = await deleteSession(sessionId);
    if (success) {
      // Refresh sessions list
      const allSessions = getAllSessions();
      setSessions(allSessions);
      
      // Close dropdown
      setShowDropdown({});
    }
  };

  const toggleDropdown = (sessionId, event) => {
    event.stopPropagation();
    setShowDropdown(prev => ({
      ...prev,
      [sessionId]: !prev[sessionId]
    }));
  };

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString(undefined, { 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return dateString;
    }
  };

  const formatSessionDate = (sessionId) => {
    try {
      const datePart = sessionId.split('_')[1];
      if (datePart) {
        const date = new Date(datePart);
        return date.toLocaleDateString(undefined, { 
          month: 'short', 
          day: 'numeric' 
        });
      }
    } catch {
      // Fallback
    }
    return 'Unknown';
  };

  return (
    <div className="w-full max-w-md">
      {/* Header with New Session Button */}
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-lg font-semibold text-white flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-[#FF9933]" />
          {t('Chat Sessions')}
        </h4>
        <button
          onClick={handleCreateNewSession}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#FF9933]/20 hover:bg-[#FF9933]/30 text-white transition-colors border border-[#FF9933]/30"
        >
          <Plus className="w-4 h-4" />
          {t('New')}
        </button>
      </div>

      {/* Current Session Info */}
      {currentSession && (
        <div className="mb-4 p-3 rounded-lg bg-[#FF9933]/10 border border-[#FF9933]/20">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 bg-[#FF9933] rounded-full"></div>
            <span className="text-white/80 text-sm font-medium">{t('Current Session')}</span>
          </div>
          <p className="text-white text-sm">{currentSession.preview}</p>
          <div className="flex items-center gap-4 mt-2 text-xs text-white/60">
            <span className="flex items-center gap-1">
              <MessageSquare className="w-3 h-3" />
              {currentSession.messageCount}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatDate(currentSession.lastUpdated)}
            </span>
          </div>
        </div>
      )}

      {/* Sessions List */}
      <div className="space-y-2 max-h-64 overflow-y-auto chat-scrollbar">
        {sessions.length === 0 ? (
          <div className="text-center py-8 text-white/60">
            <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">{t('No previous sessions')}</p>
          </div>
        ) : (
          sessions.map((session) => {
            const isCurrentSession = currentSession?.id === session.id;
            
            return (
              <div
                key={session.id}
                className={`relative p-3 rounded-lg border transition-all cursor-pointer group ${
                  isCurrentSession
                    ? 'bg-[#FF9933]/20 border-[#FF9933]/30'
                    : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20'
                }`}
                onClick={() => !isCurrentSession && handleSwitchSession(session.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <Calendar className="w-3 h-3 text-[#FF9933] flex-shrink-0" />
                      <span className="text-white/80 text-xs font-medium">
                        {formatSessionDate(session.id)}
                      </span>
                      {isCurrentSession && (
                        <div className="w-1.5 h-1.5 bg-[#FF9933] rounded-full"></div>
                      )}
                    </div>
                    <p className="text-white text-sm truncate mb-2">
                      {session.preview}
                    </p>
                    <div className="flex items-center gap-3 text-xs text-white/60">
                      <span className="flex items-center gap-1">
                        <MessageSquare className="w-3 h-3" />
                        {session.messageCount}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatDate(session.lastUpdated)}
                      </span>
                    </div>
                  </div>
                  
                  {/* Session Actions */}
                  {!isCurrentSession && (
                    <div className="relative">
                      <button
                        onClick={(e) => toggleDropdown(session.id, e)}
                        className="p-1 rounded hover:bg-white/10 text-white/60 hover:text-white transition-colors opacity-0 group-hover:opacity-100"
                      >
                        <MoreVertical className="w-4 h-4" />
                      </button>
                      
                      {showDropdown[session.id] && (
                        <div className="absolute right-0 top-8 bg-white/10 backdrop-blur-lg rounded-lg border border-white/20 py-1 z-10 min-w-[120px]">
                          <button
                            onClick={(e) => handleDeleteSession(session.id, e)}
                            className="w-full px-3 py-2 text-left text-red-400 hover:bg-red-500/20 transition-colors flex items-center gap-2 text-sm"
                          >
                            <Trash2 className="w-3 h-3" />
                            {t('Delete')}
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {!isCurrentSession && (
                    <ChevronRight className="w-4 h-4 text-white/40 group-hover:text-white/60 transition-colors ml-2" />
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Session Stats */}
      {sessions.length > 0 && (
        <div className="mt-4 pt-3 border-t border-white/10">
          <p className="text-xs text-white/60 text-center">
            {t('{{count}} sessions total', { count: sessions.length })}
          </p>
        </div>
      )}
    </div>
  );
};

export default ChatSessionManager;
