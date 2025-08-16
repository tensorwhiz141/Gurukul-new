/**
 * Chat History Controls Component
 * Provides simple history and clear buttons for chat management
 */

import React from 'react';
import { useTranslation } from 'react-i18next';
import {
  History,
  Trash2,
  X,
  MessageSquare,
  AlertTriangle,
  Plus,
  Calendar,
  Clock
} from 'lucide-react';
import { useChatHistoryControls } from '../hooks/useChatHistory';

const ChatHistoryControls = ({
  chatStats,
  onClearSession,
  onClearAll,
  getAllSessions,
  switchToSession,
  createNewSession,
  deleteSession,
  getCurrentSessionInfo,
  className = ''
}) => {
  const { t } = useTranslation();
  const {
    showHistoryModal,
    showClearConfirm,
    toggleHistoryModal,
    toggleClearConfirm,
    closeModals,
  } = useChatHistoryControls();

  return (
    <>
      {/* Control Buttons */}
      <div className={`flex items-center gap-2 ${className}`}>
        {/* History Button */}
        <button
          onClick={toggleHistoryModal}
          className="p-2 rounded-lg transition-all duration-200 hover:bg-[#FF9933]/20 hover:text-[#FF9933] text-white/70 group"
          title={t('View Chat History')}
        >
          <History className="w-5 h-5 transition-transform duration-200 group-hover:scale-110" />
        </button>

        {/* Clear Button */}
        <button
          onClick={toggleClearConfirm}
          className="p-2 rounded-lg transition-all duration-200 hover:bg-[#FF9933]/20 hover:text-[#FF9933] text-white/70 group"
          title={t('Clear Chat History')}
        >
          <Trash2 className="w-5 h-5 transition-transform duration-200 group-hover:scale-110" />
        </button>
      </div>

      {/* History Modal */}
      {showHistoryModal && (
        <ChatHistoryModal
          chatStats={chatStats}
          getAllSessions={getAllSessions}
          switchToSession={switchToSession}
          createNewSession={createNewSession}
          deleteSession={deleteSession}
          getCurrentSessionInfo={getCurrentSessionInfo}
          onClose={closeModals}
          t={t}
        />
      )}

      {/* Clear Confirmation Modal */}
      {showClearConfirm && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div 
            className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6 max-w-sm w-full"
            style={{
              background: 'rgba(255, 255, 255, 0.1)',
              backdropFilter: 'blur(20px)',
            }}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                {t('Clear History')}
              </h3>
              <button
                onClick={closeModals}
                className="p-1 rounded hover:bg-white/10 text-white/70 hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Content */}
            <p className="text-white/80 mb-6 text-sm">
              {t('What would you like to clear?')}
            </p>

            {/* Actions */}
            <div className="space-y-3">
              <button
                onClick={() => {
                  onClearSession();
                  closeModals();
                }}
                className="w-full flex items-center justify-center gap-2 p-3 rounded-lg bg-[#FF9933]/20 hover:bg-[#FF9933]/30 text-white transition-colors border border-[#FF9933]/30"
              >
                <Trash2 className="w-4 h-4" />
                {t('Clear Current Session')}
              </button>

              <button
                onClick={() => {
                  onClearAll();
                  closeModals();
                }}
                className="w-full flex items-center justify-center gap-2 p-3 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-white transition-colors border border-red-500/30"
              >
                <Trash2 className="w-4 h-4" />
                {t('Clear All History')}
              </button>

              <button
                onClick={closeModals}
                className="w-full p-3 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors border border-white/20"
              >
                {t('Cancel')}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

// ChatGPT-like History Modal Component
const ChatHistoryModal = ({
  chatStats,
  getAllSessions,
  switchToSession,
  createNewSession,
  deleteSession,
  getCurrentSessionInfo,
  onClose,
  t
}) => {
  const [sessions, setSessions] = React.useState([]);
  const [currentSession, setCurrentSession] = React.useState(null);

  React.useEffect(() => {
    if (getAllSessions) {
      const allSessions = getAllSessions();
      setSessions(allSessions);

      const current = getCurrentSessionInfo();
      setCurrentSession(current);
    }
  }, [getAllSessions, getCurrentSessionInfo]);

  const handleCreateNew = async () => {
    const newSessionId = await createNewSession();
    if (newSessionId) {
      // Refresh sessions
      const allSessions = getAllSessions();
      setSessions(allSessions);
      const current = getCurrentSessionInfo();
      setCurrentSession(current);
      onClose();
    }
  };

  const handleSwitchSession = async (sessionId) => {
    const success = await switchToSession(sessionId);
    if (success) {
      onClose();
    }
  };

  const handleDeleteSession = async (sessionId, event) => {
    event.stopPropagation();
    const success = await deleteSession(sessionId);
    if (success) {
      // Refresh sessions
      const allSessions = getAllSessions();
      setSessions(allSessions);
    }
  };

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      const now = new Date();
      const diffTime = Math.abs(now - date);
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

      if (diffDays === 1) return 'Today';
      if (diffDays === 2) return 'Yesterday';
      if (diffDays <= 7) return `${diffDays - 1} days ago`;

      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
      });
    } catch {
      return 'Unknown';
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div
        className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6 max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col"
        style={{
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(20px)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            <History className="w-6 h-6 text-[#FF9933]" />
            {t('Chat History')}
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCreateNew}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#FF9933]/20 hover:bg-[#FF9933]/30 text-white transition-colors border border-[#FF9933]/30"
            >
              <Plus className="w-4 h-4" />
              {t('New Chat')}
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 text-white/70 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-white/5 rounded-lg p-3 border border-white/10 text-center">
            <MessageSquare className="w-5 h-5 text-[#FF9933] mx-auto mb-1" />
            <p className="text-white font-semibold text-lg">{chatStats.totalMessages}</p>
            <p className="text-white/60 text-xs">{t('Messages')}</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/10 text-center">
            <Calendar className="w-5 h-5 text-[#FF9933] mx-auto mb-1" />
            <p className="text-white font-semibold text-lg">{chatStats.sessionCount}</p>
            <p className="text-white/60 text-xs">{t('Conversations')}</p>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/10 text-center">
            <Clock className="w-5 h-5 text-[#FF9933] mx-auto mb-1" />
            <p className="text-white font-semibold text-lg">{sessions.length > 0 ? formatDate(sessions[0]?.lastUpdated) : 'None'}</p>
            <p className="text-white/60 text-xs">{t('Last Chat')}</p>
          </div>
        </div>

        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto chat-scrollbar">
          {sessions.length === 0 ? (
            <div className="text-center py-12 text-white/60">
              <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg mb-2">{t('No conversations yet')}</p>
              <p className="text-sm">{t('Start chatting to see your history here')}</p>
            </div>
          ) : (
            <div className="space-y-2">
              {sessions.map((session) => {
                const isCurrentSession = currentSession?.id === session.id;

                return (
                  <div
                    key={session.id}
                    className={`group p-4 rounded-lg border transition-all cursor-pointer ${
                      isCurrentSession
                        ? 'bg-[#FF9933]/20 border-[#FF9933]/30'
                        : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20'
                    }`}
                    onClick={() => !isCurrentSession && handleSwitchSession(session.id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          {isCurrentSession && (
                            <div className="w-2 h-2 bg-[#FF9933] rounded-full"></div>
                          )}
                          <span className="text-white/80 text-sm font-medium">
                            {formatDate(session.lastUpdated)}
                          </span>
                        </div>
                        <p className="text-white text-sm mb-2 line-clamp-2">
                          {session.preview}
                        </p>
                        <div className="flex items-center gap-3 text-xs text-white/60">
                          <span className="flex items-center gap-1">
                            <MessageSquare className="w-3 h-3" />
                            {session.messageCount} {t('messages')}
                          </span>
                        </div>
                      </div>

                      {/* Delete button for non-current sessions */}
                      {!isCurrentSession && (
                        <button
                          onClick={(e) => handleDeleteSession(session.id, e)}
                          className="p-2 rounded hover:bg-red-500/20 text-white/60 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                          title={t('Delete conversation')}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-4 pt-4 border-t border-white/10 text-center">
          <p className="text-white/60 text-xs">
            {t('Your conversations are saved locally and persist across sessions')}
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatHistoryControls;
