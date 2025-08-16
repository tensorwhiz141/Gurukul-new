"""
BHIV Core Schema v0.9
Defines the complete schema for BHIV Core communication flow:
USER → NLP INPUT → AGENTIC LOGIC → CONTEXT RETRIEVAL → RESPONSE → VOICE → VIDEO

This schema ensures coherent communication between:
- BHIV Core (orchestrator)
- BHIV Alpha (agent router + planner)
- BHIV Knowledgebase (contextual vector retriever)
- Uniguru and Gurukul Interface (frontend: TTS/STT/NLP input/output)
- External APIs (e.g., LLama/Fallback)
- API Gateway (auth, permissions, session logs)
"""

from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import uuid


class UserPersona(str, Enum):
    """User persona types"""
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"
    GUEST = "guest"


class InputMode(str, Enum):
    """Input mode types"""
    CHAT = "chat"
    VOICE = "voice"
    VIDEO = "video"
    TEXT = "text"


class ClassificationType(str, Enum):
    """Agent classification types"""
    LEARNING_QUERY = "learning-query"
    WELLNESS_QUERY = "wellness-query"
    SPIRITUAL_QUERY = "spiritual-query"
    GENERAL_QUERY = "general-query"
    EMERGENCY = "emergency"


class UrgencyLevel(str, Enum):
    """Urgency levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TriggerModule(str, Enum):
    """Available trigger modules"""
    KNOWLEDGEBASE = "knowledgebase"
    TUTORBOT = "tutorbot"
    WELLNESS_BOT = "wellness_bot"
    SPIRITUAL_BOT = "spiritual_bot"
    QUIZ_BOT = "quiz_bot"


class VectorType(str, Enum):
    """Vector database types"""
    QDRANT = "qdrant"
    FAISS = "faiss"
    CHROMA = "chroma"


class LLMModel(str, Enum):
    """Available LLM models"""
    LLAMA3 = "llama3"
    GEMINI = "gemini"
    GPT4 = "gpt4"
    INTERNAL = "internal"


class PromptStyle(str, Enum):
    """Prompt styles"""
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    SYSTEM = "system"


class OutputFormat(str, Enum):
    """Expected output formats"""
    RICH_TEXT_WITH_TTS = "rich_text_with_tts"
    PLAIN_TEXT = "plain_text"
    JSON = "json"
    MARKDOWN = "markdown"


# Core Schema Models

class UserInfo(BaseModel):
    """User information and authentication"""
    user_id: str = Field(..., description="Unique user identifier")
    auth_token: str = Field(..., description="JWT or API key for authentication")
    lang: str = Field(default="en-IN", description="User's preferred language")
    persona: UserPersona = Field(default=UserPersona.STUDENT, description="User's role/persona")
    permissions: List[str] = Field(default=["read"], description="User permissions")

    class Config:
        use_enum_values = True


class InputContext(BaseModel):
    """Context information for the input"""
    course: Optional[str] = Field(None, description="Current course context")
    previous_message_id: Optional[str] = Field(None, description="Previous message ID for conversation continuity")
    session_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional session context")


class UserInput(BaseModel):
    """User input processing"""
    raw_text: str = Field(..., description="Raw user input text")
    voice_enabled: bool = Field(default=False, description="Whether voice input/output is enabled")
    mode: InputMode = Field(default=InputMode.CHAT, description="Input mode")
    context: InputContext = Field(default_factory=InputContext, description="Input context")

    class Config:
        use_enum_values = True


class AgentDirectives(BaseModel):
    """Agent routing and planning directives"""
    classification: ClassificationType = Field(..., description="Query classification")
    urgency: UrgencyLevel = Field(default=UrgencyLevel.NORMAL, description="Urgency level")
    trigger_module: TriggerModule = Field(..., description="Module to trigger")
    agent_name: str = Field(..., description="Specific agent name")
    expected_output_format: OutputFormat = Field(default=OutputFormat.RICH_TEXT_WITH_TTS, description="Expected output format")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classification confidence")

    class Config:
        use_enum_values = True


class KnowledgebaseFilters(BaseModel):
    """Filters for knowledgebase queries"""
    lang: Optional[str] = Field(None, description="Language filter")
    curriculum_tag: Optional[List[str]] = Field(None, description="Curriculum tags")
    difficulty_level: Optional[str] = Field(None, description="Difficulty level")
    subject: Optional[str] = Field(None, description="Subject filter")


class KnowledgebaseQuery(BaseModel):
    """Knowledgebase query configuration"""
    use_vector: bool = Field(default=True, description="Whether to use vector search")
    vector_type: VectorType = Field(default=VectorType.QDRANT, description="Vector database type")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to retrieve")
    include_sources: bool = Field(default=True, description="Whether to include source information")
    query_embedding: Optional[str] = Field(None, description="Pre-computed query embedding")
    filters: KnowledgebaseFilters = Field(default_factory=KnowledgebaseFilters, description="Query filters")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity threshold")

    class Config:
        use_enum_values = True


class LLMRequest(BaseModel):
    """LLM request configuration"""
    model: LLMModel = Field(default=LLMModel.LLAMA3, description="LLM model to use")
    prompt_style: PromptStyle = Field(default=PromptStyle.INSTRUCTION, description="Prompt style")
    max_tokens: int = Field(default=1024, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Temperature for generation")
    response_language: str = Field(default="en", description="Response language")
    system_prompt: Optional[str] = Field(None, description="System prompt override")

    class Config:
        use_enum_values = True


class OutputResponse(BaseModel):
    """Response output"""
    response_text: str = Field(..., description="Generated response text")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Response confidence score")
    sources: List[str] = Field(default_factory=list, description="Source documents/references")
    voice_response_url: Optional[str] = Field(None, description="URL to voice response audio")
    followup_suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional response metadata")


class LogEntry(BaseModel):
    """Individual log entry"""
    timestamp: datetime = Field(default_factory=datetime.now, description="Log entry timestamp")
    level: str = Field(..., description="Log level (info, warning, error)")
    message: str = Field(..., description="Log message")
    component: str = Field(..., description="Component that generated the log")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional log data")


class SystemLogs(BaseModel):
    """System logging information"""
    agent_chain_log: List[str] = Field(default_factory=list, description="Agent processing chain log")
    reward_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Reward score for the interaction")
    error_flag: bool = Field(default=False, description="Whether an error occurred")
    processing_time_ms: Optional[float] = Field(None, description="Total processing time in milliseconds")
    detailed_logs: List[LogEntry] = Field(default_factory=list, description="Detailed log entries")


# Main BHIV Core Schema

class BHIVCoreRequest(BaseModel):
    """Complete BHIV Core request schema"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Request timestamp")
    user_info: UserInfo = Field(..., description="User information and authentication")
    input: UserInput = Field(..., description="User input processing")
    agent_directives: Optional[AgentDirectives] = Field(None, description="Agent routing and planning directives")
    knowledgebase_query: Optional[KnowledgebaseQuery] = Field(None, description="Knowledgebase query configuration")
    llm_request: Optional[LLMRequest] = Field(None, description="LLM request configuration")

    @validator('session_id')
    def validate_session_id(cls, v):
        """Validate session ID format"""
        if not v or len(v) < 10:
            return str(uuid.uuid4())
        return v

    class Config:
        use_enum_values = True


class BHIVCoreResponse(BaseModel):
    """Complete BHIV Core response schema"""
    session_id: str = Field(..., description="Session identifier from request")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    user_info: UserInfo = Field(..., description="User information from request")
    input: UserInput = Field(..., description="Original user input")
    agent_directives: AgentDirectives = Field(..., description="Applied agent directives")
    knowledgebase_query: Optional[KnowledgebaseQuery] = Field(None, description="Executed knowledgebase query")
    llm_request: Optional[LLMRequest] = Field(None, description="Executed LLM request")
    output: OutputResponse = Field(..., description="Generated response output")
    logs: SystemLogs = Field(default_factory=SystemLogs, description="System logs and metrics")

    class Config:
        use_enum_values = True


# Helper Models for Agent Communication

class AgentClassificationResult(BaseModel):
    """Result of agent classification"""
    classification: ClassificationType = Field(..., description="Classified query type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    urgency: UrgencyLevel = Field(..., description="Determined urgency level")
    recommended_agent: str = Field(..., description="Recommended agent name")
    reasoning: str = Field(..., description="Classification reasoning")

    class Config:
        use_enum_values = True


class KnowledgebaseResult(BaseModel):
    """Result from knowledgebase query"""
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    total_results: int = Field(default=0, description="Total number of results")
    query_time_ms: float = Field(..., description="Query execution time")
    embedding_used: Optional[str] = Field(None, description="Embedding model used")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Filters that were applied")


class VoiceProcessingResult(BaseModel):
    """Result from voice processing"""
    audio_url: Optional[str] = Field(None, description="Generated audio URL")
    duration_seconds: Optional[float] = Field(None, description="Audio duration")
    voice_model: Optional[str] = Field(None, description="Voice model used")
    processing_time_ms: Optional[float] = Field(None, description="Voice processing time")
    error: Optional[str] = Field(None, description="Voice processing error if any")


# Utility Functions

def create_bhiv_request(
    user_id: str,
    raw_text: str,
    auth_token: str = "default_token",
    voice_enabled: bool = False,
    mode: InputMode = InputMode.CHAT,
    persona: UserPersona = UserPersona.STUDENT,
    lang: str = "en-IN"
) -> BHIVCoreRequest:
    """Create a BHIV Core request with sensible defaults"""
    return BHIVCoreRequest(
        user_info=UserInfo(
            user_id=user_id,
            auth_token=auth_token,
            lang=lang,
            persona=persona,
            permissions=["read", "write"]
        ),
        input=UserInput(
            raw_text=raw_text,
            voice_enabled=voice_enabled,
            mode=mode,
            context=InputContext()
        )
    )


def create_agent_directives(
    classification: ClassificationType,
    agent_name: str,
    urgency: UrgencyLevel = UrgencyLevel.NORMAL,
    trigger_module: TriggerModule = TriggerModule.KNOWLEDGEBASE,
    output_format: OutputFormat = OutputFormat.RICH_TEXT_WITH_TTS,
    confidence: Optional[float] = None
) -> AgentDirectives:
    """Create agent directives with specified parameters"""
    return AgentDirectives(
        classification=classification,
        urgency=urgency,
        trigger_module=trigger_module,
        agent_name=agent_name,
        expected_output_format=output_format,
        confidence_score=confidence
    )
