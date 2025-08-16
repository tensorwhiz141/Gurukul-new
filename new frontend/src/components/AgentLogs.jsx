import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { selectUser } from '../store/authSlice';
import agentLogsService from '../services/agentLogsService';
import { Clock, User, Bot, Calendar, ArrowRight } from 'lucide-react';

/**
 * Component to display agent logs for the current user
 */
const AgentLogs = () => {
  const user = useSelector(selectUser);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        setLoading(true);
        const userId = user?.id || 'guest-user';
        const result = await agentLogsService.getAgentLogs(userId);
        
        if (result.success) {
          setLogs(result.data || []);
        } else {
          setError('Failed to load agent logs');
        }
      } catch (err) {
        console.error('Error fetching agent logs:', err);
        setError('An unexpected error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchLogs();
  }, [user]);

  // Format duration from seconds to readable format
  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${remainingSeconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`;
    } else {
      return `${remainingSeconds}s`;
    }
  };

  // Format date to readable format
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
        return 'text-green-500';
      case 'completed':
        return 'text-blue-500';
      case 'interrupted':
        return 'text-orange-500';
      default:
        return 'text-gray-500';
    }
  };

  // Get agent type color
  const getAgentTypeColor = (type) => {
    switch (type) {
      case 'financial':
        return 'bg-blue-500/20 text-blue-300';
      case 'education':
        return 'bg-green-500/20 text-green-300';
      case 'wellness':
        return 'bg-orange-500/20 text-orange-300';
      default:
        return 'bg-gray-500/20 text-gray-300';
    }
  };

  return (
    <div className="bg-gray-900/50 rounded-lg p-4 backdrop-blur-sm border border-gray-800">
      <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
        <Clock size={20} className="mr-2 text-blue-400" />
        Agent Session Logs
      </h2>
      
      {loading ? (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      ) : error ? (
        <div className="text-red-400 p-4 text-center">{error}</div>
      ) : logs.length === 0 ? (
        <div className="text-gray-400 p-4 text-center">No agent logs found</div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
          {logs.map((log) => (
            <div 
              key={log.id} 
              className="bg-gray-800/50 rounded-lg p-3 border border-gray-700 hover:border-gray-600 transition-colors"
            >
              <div className="flex justify-between items-start mb-2">
                <div>
                  <span className={`font-medium text-lg ${log.agent_name ? 'text-white' : 'text-gray-400'}`}>
                    {log.agent_name || 'Unknown Agent'}
                  </span>
                  {log.agent_type && (
                    <span className={`ml-2 text-xs px-2 py-0.5 rounded ${getAgentTypeColor(log.agent_type)}`}>
                      {log.agent_type}
                    </span>
                  )}
                </div>
                <span className={`text-sm font-medium ${getStatusColor(log.status)}`}>
                  {log.status}
                </span>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                <div className="flex items-center text-gray-400">
                  <User size={14} className="mr-1.5" />
                  <span>User: {log.user_id === 'guest-user' ? 'Guest' : 'Authenticated'}</span>
                </div>
                
                <div className="flex items-center text-gray-400">
                  <Bot size={14} className="mr-1.5" />
                  <span>Agent ID: {log.agent_id}</span>
                </div>
                
                <div className="flex items-center text-gray-400">
                  <Calendar size={14} className="mr-1.5" />
                  <span>Started: {formatDate(log.start_time)}</span>
                </div>
                
                {log.end_time && (
                  <div className="flex items-center text-gray-400">
                    <Calendar size={14} className="mr-1.5" />
                    <span>Ended: {formatDate(log.end_time)}</span>
                  </div>
                )}
                
                {log.duration && (
                  <div className="flex items-center text-gray-400 col-span-full">
                    <Clock size={14} className="mr-1.5" />
                    <span>Duration: {formatDuration(log.duration)}</span>
                  </div>
                )}
                
                <div className="flex items-center text-gray-400 col-span-full">
                  <ArrowRight size={14} className="mr-1.5" />
                  <span>Action: {log.action_type}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AgentLogs;
