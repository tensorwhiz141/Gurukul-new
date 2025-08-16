/**
 * BHIV Core Schema v0.9 - TypeScript Types
 * Defines the complete schema for BHIV Core communication flow:
 * USER → NLP INPUT → AGENTIC LOGIC → CONTEXT RETRIEVAL → RESPONSE → VOICE → VIDEO
 */

// Enums
export enum UserPersona {
  STUDENT = "student",
  TEACHER = "teacher",
  ADMIN = "admin",
  GUEST = "guest"
}

export enum InputMode {
  CHAT = "chat",
  VOICE = "voice",
  VIDEO = "video",
  TEXT = "text"
}

export enum ClassificationType {
  LEARNING_QUERY = "learning-query",
  WELLNESS_QUERY = "wellness-query",
  SPIRITUAL_QUERY = "spiritual-query",
  GENERAL_QUERY = "general-query",
  EMERGENCY = "emergency"
}

export enum UrgencyLevel {
  LOW = "low",
  NORMAL = "normal",
  HIGH = "high",
  CRITICAL = "critical"
}

export enum TriggerModule {
  KNOWLEDGEBASE = "knowledgebase",
  TUTORBOT = "tutorbot",
  WELLNESS_BOT = "wellness_bot",
  SPIRITUAL_BOT = "spiritual_bot",
  QUIZ_BOT = "quiz_bot"
}

export enum VectorType {
  QDRANT = "qdrant",
  FAISS = "faiss",
  CHROMA = "chroma"
}

export enum LLMModel {
  LLAMA3 = "llama3",
  GEMINI = "gemini",
  GPT4 = "gpt4",
  INTERNAL = "internal"
}

export enum PromptStyle {
  INSTRUCTION = "instruction",
  CONVERSATION = "conversation",
  SYSTEM = "system"
}

export enum OutputFormat {
  RICH_TEXT_WITH_TTS = "rich_text_with_tts",
  PLAIN_TEXT = "plain_text",
  JSON = "json",
  MARKDOWN = "markdown"
}

// Core Interfaces
export interface UserInfo {
  user_id: string;
  auth_token: string;
  lang?: string;
  persona?: UserPersona;
  permissions?: string[];
}

export interface InputContext {
  course?: string;
  previous_message_id?: string;
  session_context?: Record<string, any>;
}

export interface UserInput {
  raw_text: string;
  voice_enabled?: boolean;
  mode?: InputMode;
  context?: InputContext;
}

export interface AgentDirectives {
  classification: ClassificationType;
  urgency?: UrgencyLevel;
  trigger_module: TriggerModule;
  agent_name: string;
  expected_output_format?: OutputFormat;
  confidence_score?: number;
}

export interface KnowledgebaseFilters {
  lang?: string;
  curriculum_tag?: string[];
  difficulty_level?: string;
  subject?: string;
}

export interface KnowledgebaseQuery {
  use_vector?: boolean;
  vector_type?: VectorType;
  top_k?: number;
  include_sources?: boolean;
  query_embedding?: string;
  filters?: KnowledgebaseFilters;
  similarity_threshold?: number;
}

export interface LLMRequest {
  model?: LLMModel;
  prompt_style?: PromptStyle;
  max_tokens?: number;
  temperature?: number;
  response_language?: string;
  system_prompt?: string;
}

export interface OutputResponse {
  response_text: string;
  confidence_score: number;
  sources?: string[];
  voice_response_url?: string;
  followup_suggestions?: string[];
  metadata?: Record<string, any>;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  component: string;
  data?: Record<string, any>;
}

export interface SystemLogs {
  agent_chain_log?: string[];
  reward_score?: number;
  error_flag?: boolean;
  processing_time_ms?: number;
  detailed_logs?: LogEntry[];
}

// Main BHIV Core Schema Interfaces
export interface BHIVCoreRequest {
  session_id?: string;
  timestamp?: string;
  user_info: UserInfo;
  input: UserInput;
  agent_directives?: AgentDirectives;
  knowledgebase_query?: KnowledgebaseQuery;
  llm_request?: LLMRequest;
}

export interface BHIVCoreResponse {
  session_id: string;
  timestamp: string;
  user_info: UserInfo;
  input: UserInput;
  agent_directives: AgentDirectives;
  knowledgebase_query?: KnowledgebaseQuery;
  llm_request?: LLMRequest;
  output: OutputResponse;
  logs: SystemLogs;
}

// Helper Interfaces
export interface AgentClassificationResult {
  classification: ClassificationType;
  confidence: number;
  urgency: UrgencyLevel;
  recommended_agent: string;
  reasoning: string;
}

export interface KnowledgebaseResult {
  documents: Record<string, any>[];
  total_results: number;
  query_time_ms: number;
  embedding_used?: string;
  filters_applied?: Record<string, any>;
}

export interface VoiceProcessingResult {
  audio_url?: string;
  duration_seconds?: number;
  voice_model?: string;
  processing_time_ms?: number;
  error?: string;
}

// Utility Functions
export const createBHIVRequest = (
  userId: string,
  rawText: string,
  options: {
    authToken?: string;
    voiceEnabled?: boolean;
    mode?: InputMode;
    persona?: UserPersona;
    lang?: string;
  } = {}
): BHIVCoreRequest => {
  const {
    authToken = "default_token",
    voiceEnabled = false,
    mode = InputMode.CHAT,
    persona = UserPersona.STUDENT,
    lang = "en-IN"
  } = options;

  return {
    session_id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    user_info: {
      user_id: userId,
      auth_token: authToken,
      lang,
      persona,
      permissions: ["read", "write"]
    },
    input: {
      raw_text: rawText,
      voice_enabled: voiceEnabled,
      mode,
      context: {}
    }
  };
};

export const createAgentDirectives = (
  classification: ClassificationType,
  agentName: string,
  options: {
    urgency?: UrgencyLevel;
    triggerModule?: TriggerModule;
    outputFormat?: OutputFormat;
    confidence?: number;
  } = {}
): AgentDirectives => {
  const {
    urgency = UrgencyLevel.NORMAL,
    triggerModule = TriggerModule.KNOWLEDGEBASE,
    outputFormat = OutputFormat.RICH_TEXT_WITH_TTS,
    confidence
  } = options;

  return {
    classification,
    urgency,
    trigger_module: triggerModule,
    agent_name: agentName,
    expected_output_format: outputFormat,
    confidence_score: confidence
  };
};

// Type Guards
export const isBHIVCoreResponse = (obj: any): obj is BHIVCoreResponse => {
  return obj && 
    typeof obj.session_id === 'string' &&
    typeof obj.timestamp === 'string' &&
    obj.user_info &&
    obj.input &&
    obj.agent_directives &&
    obj.output &&
    obj.logs;
};

export const isValidClassification = (classification: string): classification is ClassificationType => {
  return Object.values(ClassificationType).includes(classification as ClassificationType);
};
