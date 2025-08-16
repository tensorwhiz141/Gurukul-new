import { supabase } from "../supabaseClient";

/**
 * Service for handling chatbot and summarizer logging operations
 */
const chatLogsService = {
  /**
   * Log chatbot message
   * @param {Object} params - Parameters for logging
   * @param {string} params.userId - User ID (or 'guest-user')
   * @param {string} params.message - User's message
   * @param {string} params.model - AI model used (e.g., 'grok', 'llama', 'chatgpt')
   * @param {string} params.responseLength - Length of AI response
   * @returns {Promise<Object>} - The created log entry
   */
  async logChatMessage({ userId, message, model, responseLength }) {
    try {
      // Ensure we have a valid user ID
      const safeUserId = userId || "guest-user";

      // Create a new log entry
      const { data, error } = await supabase
        .from("chatbot_logs")
        .insert({
          user_id: safeUserId,
          message: message,
          model: model,
          response_length: responseLength,
          start_time: new Date().toISOString(),
          status: "completed",
        })
        .select()
        .single();

      if (error) {
        console.error("Error logging chat message:", error);
        return { success: false, error };
      }

      return { success: true, data };
    } catch (error) {
      console.error("Unexpected error logging chat message:", error);
      return { success: false, error };
    }
  },

  /**
   * Log document summary
   * @param {Object} params - Parameters for logging
   * @param {string} params.userId - User ID (or 'guest-user')
   * @param {string} params.fileName - Name of the file
   * @param {string} params.fileType - Type of the file
   * @param {string} params.fileSize - Size of the file in bytes
   * @param {string} params.model - AI model used
   * @param {boolean} params.hasAudio - Whether audio was generated
   * @returns {Promise<Object>} - The created log entry
   */
  async logDocumentSummary({
    userId,
    fileName,
    fileType,
    fileSize,
    model,
    hasAudio,
  }) {
    try {
      // Ensure we have a valid user ID
      const safeUserId = userId || "guest-user";

      // Create a new log entry
      const { data, error } = await supabase
        .from("summarizer_logs")
        .insert({
          user_id: safeUserId,
          file_name: fileName,
          file_type: fileType,
          file_size: fileSize,
          model: model,
          has_audio: hasAudio,
          start_time: new Date().toISOString(),
          status: "completed",
        })
        .select()
        .single();

      if (error) {
        console.error("Error logging document summary:", error);
        return { success: false, error };
      }

      return { success: true, data };
    } catch (error) {
      console.error("Unexpected error logging document summary:", error);
      return { success: false, error };
    }
  },

  /**
   * Get chatbot logs for a user
   * @param {string} userId - User ID (or 'guest-user')
   * @returns {Promise<Object>} - The user's chatbot logs
   */
  async getChatbotLogs(userId) {
    try {
      // Ensure we have a valid user ID
      const safeUserId = userId || "guest-user";

      const { data, error } = await supabase
        .from("chatbot_logs")
        .select("*")
        .eq("user_id", safeUserId)
        .order("created_at", { ascending: false });

      if (error) {
        console.error("Error fetching chatbot logs:", error);
        return { success: false, error };
      }

      return { success: true, data };
    } catch (error) {
      console.error("Unexpected error fetching chatbot logs:", error);
      return { success: false, error };
    }
  },

  /**
   * Get summarizer logs for a user
   * @param {string} userId - User ID (or 'guest-user')
   * @returns {Promise<Object>} - The user's summarizer logs
   */
  async getSummarizerLogs(userId) {
    try {
      // Ensure we have a valid user ID
      const safeUserId = userId || "guest-user";

      const { data, error } = await supabase
        .from("summarizer_logs")
        .select("*")
        .eq("user_id", safeUserId)
        .order("created_at", { ascending: false });

      if (error) {
        console.error("Error fetching summarizer logs:", error);
        return { success: false, error };
      }

      return { success: true, data };
    } catch (error) {
      console.error("Unexpected error fetching summarizer logs:", error);
      return { success: false, error };
    }
  },

  /**
   * Get chat statistics for a user
   * @param {string} userId - User ID (or 'guest-user')
   * @returns {Promise<Object>} - The user's chat statistics
   */
  async getChatStats(userId) {
    try {
      // Ensure we have a valid user ID
      const safeUserId = userId || "guest-user";

      // Get total chat messages
      const { data: chatData, error: chatError } = await supabase
        .from("chatbot_logs")
        .select("id")
        .eq("user_id", safeUserId);

      // Get total document summaries
      const { data: summaryData, error: summaryError } = await supabase
        .from("summarizer_logs")
        .select("id")
        .eq("user_id", safeUserId);

      if (chatError || summaryError) {
        console.error("Error fetching chat stats:", chatError || summaryError);
        return { success: false, error: chatError || summaryError };
      }

      // Get chatbot model usage statistics
      const { data: chatModelData, error: chatModelError } = await supabase
        .from("chatbot_logs")
        .select("model, id")
        .eq("user_id", safeUserId);

      // Get summarizer model usage statistics
      const { data: summaryModelData, error: summaryModelError } =
        await supabase
          .from("summarizer_logs")
          .select("model, id")
          .eq("user_id", safeUserId);

      if (chatModelError || summaryModelError) {
        console.error(
          "Error fetching model stats:",
          chatModelError || summaryModelError
        );
        return { success: false, error: chatModelError || summaryModelError };
      }

      // Calculate model usage
      const modelStats = {};

      // Process chatbot models
      chatModelData.forEach((log) => {
        const model = log.model || "unknown";
        modelStats[model] = (modelStats[model] || 0) + 1;
      });

      // Process summarizer models
      summaryModelData.forEach((log) => {
        const model = log.model || "unknown";
        modelStats[model] = (modelStats[model] || 0) + 1;
      });

      return {
        success: true,
        data: {
          totalChatMessages: chatData.length,
          totalDocumentSummaries: summaryData.length,
          modelUsage: modelStats,
        },
      };
    } catch (error) {
      console.error("Unexpected error fetching chat stats:", error);
      return { success: false, error };
    }
  },
};

export default chatLogsService;
