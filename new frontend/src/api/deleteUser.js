// This file would typically be deployed as a serverless function
// It handles the secure deletion of a user account from Supabase

import { createClient } from '@supabase/supabase-js';

// Create a Supabase admin client with service role key
// IMPORTANT: This should NEVER be exposed in client-side code
// This should only be used in a secure server environment
const createAdminClient = () => {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  
  if (!supabaseUrl || !supabaseServiceKey) {
    throw new Error('Missing Supabase environment variables');
  }
  
  return createClient(supabaseUrl, supabaseServiceKey, {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    }
  });
};

// Handler function for the delete-user endpoint
export const handler = async (req, res) => {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const { userId, authToken } = req.body;
    
    if (!userId || !authToken) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }
    
    // Verify the auth token
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_ANON_KEY
    );
    
    const { data: { user }, error: verifyError } = await supabase.auth.getUser(authToken);
    
    if (verifyError || !user) {
      return res.status(401).json({ error: 'Invalid authentication' });
    }
    
    // Ensure the user is only deleting their own account
    if (user.id !== userId) {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    
    // Create admin client to delete the user
    const adminClient = createAdminClient();
    
    // Delete user data from related tables first
    // This would depend on your database schema
    
    // Finally delete the user
    const { error: deleteError } = await adminClient.auth.admin.deleteUser(userId);
    
    if (deleteError) {
      throw deleteError;
    }
    
    return res.status(200).json({ success: true, message: 'User deleted successfully' });
  } catch (error) {
    console.error('Error deleting user:', error);
    return res.status(500).json({ error: 'Failed to delete user', message: error.message });
  }
};
