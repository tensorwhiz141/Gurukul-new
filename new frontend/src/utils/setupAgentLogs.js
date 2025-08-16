import { supabase } from '../supabaseClient';
import fs from 'fs';
import path from 'path';

/**
 * Setup agent logs table in Supabase
 * This function should be run once to set up the necessary tables and policies
 * It can be called from a server-side script or admin panel
 */
export const setupAgentLogs = async () => {
  try {
    // Read the SQL file
    const sqlPath = path.resolve(process.cwd(), 'agent_logs_setup.sql');
    const sql = fs.readFileSync(sqlPath, 'utf8');
    
    // Execute the SQL
    const { error } = await supabase.rpc('exec_sql', { sql });
    
    if (error) {
      console.error('Error setting up agent logs:', error);
      return { success: false, error };
    }
    
    return { success: true };
  } catch (error) {
    console.error('Unexpected error setting up agent logs:', error);
    return { success: false, error };
  }
};

/**
 * Check if agent_logs table exists
 * @returns {Promise<boolean>} - Whether the table exists
 */
export const checkAgentLogsTableExists = async () => {
  try {
    // Try to select from the table
    const { data, error } = await supabase
      .from('agent_logs')
      .select('id')
      .limit(1);
    
    // If there's no error, the table exists
    return !error;
  } catch (error) {
    console.error('Error checking if agent_logs table exists:', error);
    return false;
  }
};

/**
 * Initialize agent logs
 * This function checks if the agent_logs table exists and creates it if it doesn't
 * It can be called during application startup
 */
export const initializeAgentLogs = async () => {
  try {
    const tableExists = await checkAgentLogsTableExists();
    
    if (!tableExists) {
      console.log('Agent logs table does not exist, setting up...');
      return await setupAgentLogs();
    }
    
    console.log('Agent logs table already exists');
    return { success: true };
  } catch (error) {
    console.error('Error initializing agent logs:', error);
    return { success: false, error };
  }
};

export default {
  setupAgentLogs,
  checkAgentLogsTableExists,
  initializeAgentLogs
};
