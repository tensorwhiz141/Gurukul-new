import { supabase } from '../supabaseClient';

/**
 * Service for handling agent logging operations
 */
const agentLogsService = {
  /**
   * Log agent start event
   * @param {Object} params - Parameters for logging
   * @param {string} params.userId - User ID (or 'guest-user')
   * @param {number} params.agentId - Agent ID
   * @param {string} params.agentName - Agent name
   * @param {string} params.agentType - Agent type (e.g., 'financial', 'education')
   * @returns {Promise<Object>} - The created log entry
   */
  async logAgentStart({ userId, agentId, agentName, agentType }) {
    try {
      // Ensure we have a valid user ID
      const safeUserId = userId || 'guest-user';
      
      // Create a new log entry
      const { data, error } = await supabase
        .from('agent_logs')
        .insert({
          user_id: safeUserId,
          agent_id: agentId,
          agent_name: agentName,
          agent_type: agentType,
          action_type: 'start',
          start_time: new Date().toISOString(),
          status: 'active'
        })
        .select()
        .single();
      
      if (error) {
        console.error('Error logging agent start:', error);
        return { success: false, error };
      }
      
      return { success: true, data };
    } catch (error) {
      console.error('Unexpected error logging agent start:', error);
      return { success: false, error };
    }
  },
  
  /**
   * Log agent stop event
   * @param {Object} params - Parameters for logging
   * @param {string} params.userId - User ID (or 'guest-user')
   * @param {number} params.agentId - Agent ID
   * @returns {Promise<Object>} - The updated log entry
   */
  async logAgentStop({ userId, agentId }) {
    try {
      // Ensure we have a valid user ID
      const safeUserId = userId || 'guest-user';
      
      // Find the active log entry for this agent
      const { data: activeLog, error: findError } = await supabase
        .from('agent_logs')
        .select('*')
        .eq('user_id', safeUserId)
        .eq('agent_id', agentId)
        .eq('status', 'active')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();
      
      if (findError) {
        console.error('Error finding active agent log:', findError);
        return { success: false, error: findError };
      }
      
      if (!activeLog) {
        console.warn('No active log found for agent', agentId);
        return { success: false, error: { message: 'No active log found' } };
      }
      
      // Calculate duration
      const endTime = new Date();
      const startTime = new Date(activeLog.start_time);
      const durationSeconds = Math.floor((endTime - startTime) / 1000);
      
      // Update the log entry
      const { data, error } = await supabase
        .from('agent_logs')
        .update({
          end_time: endTime.toISOString(),
          duration: durationSeconds,
          status: 'completed',
          action_type: 'stop'
        })
        .eq('id', activeLog.id)
        .select()
        .single();
      
      if (error) {
        console.error('Error updating agent log:', error);
        return { success: false, error };
      }
      
      return { success: true, data };
    } catch (error) {
      console.error('Unexpected error logging agent stop:', error);
      return { success: false, error };
    }
  },
  
  /**
   * Log agent reset event
   * @param {Object} params - Parameters for logging
   * @param {string} params.userId - User ID (or 'guest-user')
   * @returns {Promise<Object>} - Result of the operation
   */
  async logAgentReset({ userId }) {
    try {
      // Ensure we have a valid user ID
      const safeUserId = userId || 'guest-user';
      
      // Find all active log entries for this user
      const { data: activeLogs, error: findError } = await supabase
        .from('agent_logs')
        .select('*')
        .eq('user_id', safeUserId)
        .eq('status', 'active');
      
      if (findError) {
        console.error('Error finding active agent logs:', findError);
        return { success: false, error: findError };
      }
      
      if (!activeLogs || activeLogs.length === 0) {
        // No active logs to update
        return { success: true, data: { message: 'No active logs found' } };
      }
      
      // Update all active logs
      const endTime = new Date();
      const updatePromises = activeLogs.map(log => {
        const startTime = new Date(log.start_time);
        const durationSeconds = Math.floor((endTime - startTime) / 1000);
        
        return supabase
          .from('agent_logs')
          .update({
            end_time: endTime.toISOString(),
            duration: durationSeconds,
            status: 'interrupted',
            action_type: 'reset'
          })
          .eq('id', log.id);
      });
      
      // Wait for all updates to complete
      await Promise.all(updatePromises);
      
      return { success: true, data: { message: `Updated ${activeLogs.length} logs` } };
    } catch (error) {
      console.error('Unexpected error logging agent reset:', error);
      return { success: false, error };
    }
  },
  
  /**
   * Get agent logs for a user
   * @param {string} userId - User ID (or 'guest-user')
   * @returns {Promise<Object>} - The user's agent logs
   */
  async getAgentLogs(userId) {
    try {
      // Ensure we have a valid user ID
      const safeUserId = userId || 'guest-user';
      
      const { data, error } = await supabase
        .from('agent_logs')
        .select('*')
        .eq('user_id', safeUserId)
        .order('created_at', { ascending: false });
      
      if (error) {
        console.error('Error fetching agent logs:', error);
        return { success: false, error };
      }
      
      return { success: true, data };
    } catch (error) {
      console.error('Unexpected error fetching agent logs:', error);
      return { success: false, error };
    }
  }
};

export default agentLogsService;
