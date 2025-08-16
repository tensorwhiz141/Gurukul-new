"""
Comprehensive Tests for BHIV Core Implementation
Tests the complete BHIV Core flow and all components

Test Coverage:
- BHIV Core Schema validation
- BHIV Core Orchestrator functionality
- BHIV Alpha Agent Router
- BHIV Knowledgebase integration
- Voice and Video Pipeline
- Session Management
- API Gateway functionality
- End-to-end flow validation
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

# Import BHIV components
from bhiv_core_schema import (
    BHIVCoreRequest, BHIVCoreResponse, UserInfo, UserInput, AgentDirectives,
    KnowledgebaseQuery, LLMRequest, OutputResponse, SystemLogs,
    ClassificationType, UrgencyLevel, TriggerModule, OutputFormat,
    InputMode, UserPersona, VectorType, LLMModel, PromptStyle,
    create_bhiv_request, create_agent_directives
)
from bhiv_core_orchestrator import BHIVCoreOrchestrator, BHIVAlphaRouter, BHIVKnowledgebase
from bhiv_alpha_router import BHIVAlphaEnhanced, AdvancedClassificationEngine
from bhiv_knowledgebase import EnhancedKnowledgebaseManager
from bhiv_voice_video_pipeline import BHIVVoiceVideoOrchestrator
from bhiv_session_manager import BHIVSessionManager, SessionMetrics, InteractionRecord
from bhiv_api_gateway import BHIVAPIGateway, AuthenticationManager, RateLimiter


class TestBHIVCoreSchema:
    """Test BHIV Core Schema components"""
    
    def test_user_info_creation(self):
        """Test UserInfo model creation and validation"""
        user_info = UserInfo(
            user_id="test_user_123",
            auth_token="test_token",
            lang="en-IN",
            persona=UserPersona.STUDENT,
            permissions=["read", "write"]
        )
        
        assert user_info.user_id == "test_user_123"
        assert user_info.auth_token == "test_token"
        assert user_info.lang == "en-IN"
        assert user_info.persona == UserPersona.STUDENT
        assert "read" in user_info.permissions
        assert "write" in user_info.permissions
    
    def test_user_input_creation(self):
        """Test UserInput model creation"""
        user_input = UserInput(
            raw_text="How do I learn machine learning?",
            voice_enabled=True,
            mode=InputMode.CHAT
        )
        
        assert user_input.raw_text == "How do I learn machine learning?"
        assert user_input.voice_enabled is True
        assert user_input.mode == InputMode.CHAT
    
    def test_agent_directives_creation(self):
        """Test AgentDirectives model creation"""
        directives = create_agent_directives(
            ClassificationType.LEARNING_QUERY,
            "EduMentor",
            urgency=UrgencyLevel.NORMAL,
            trigger_module=TriggerModule.KNOWLEDGEBASE,
            output_format=OutputFormat.RICH_TEXT_WITH_TTS,
            confidence=0.95
        )
        
        assert directives.classification == ClassificationType.LEARNING_QUERY
        assert directives.agent_name == "EduMentor"
        assert directives.urgency == UrgencyLevel.NORMAL
        assert directives.confidence_score == 0.95
    
    def test_bhiv_request_creation(self):
        """Test BHIV Core request creation"""
        request = create_bhiv_request(
            user_id="test_user",
            raw_text="What is artificial intelligence?",
            voice_enabled=True,
            mode=InputMode.CHAT,
            persona=UserPersona.STUDENT,
            lang="en-IN"
        )
        
        assert isinstance(request, BHIVCoreRequest)
        assert request.user_info.user_id == "test_user"
        assert request.input.raw_text == "What is artificial intelligence?"
        assert request.input.voice_enabled is True
        assert request.user_info.persona == UserPersona.STUDENT
    
    def test_knowledgebase_query_creation(self):
        """Test KnowledgebaseQuery model"""
        kb_query = KnowledgebaseQuery(
            use_vector=True,
            vector_type=VectorType.QDRANT,
            top_k=5,
            include_sources=True,
            similarity_threshold=0.8
        )
        
        assert kb_query.use_vector is True
        assert kb_query.vector_type == VectorType.QDRANT
        assert kb_query.top_k == 5
        assert kb_query.similarity_threshold == 0.8
    
    def test_llm_request_creation(self):
        """Test LLMRequest model"""
        llm_request = LLMRequest(
            model=LLMModel.GEMINI,
            prompt_style=PromptStyle.INSTRUCTION,
            max_tokens=1024,
            temperature=0.3,
            response_language="en"
        )
        
        assert llm_request.model == LLMModel.GEMINI
        assert llm_request.prompt_style == PromptStyle.INSTRUCTION
        assert llm_request.max_tokens == 1024
        assert llm_request.temperature == 0.3


class TestBHIVAlphaRouter:
    """Test BHIV Alpha Agent Router"""
    
    @pytest.fixture
    def mock_gemini_manager(self):
        """Mock Gemini API manager"""
        mock_manager = Mock()
        mock_manager.is_available.return_value = True
        mock_manager.generate_content.return_value = json.dumps({
            "classification": "learning-query",
            "confidence": 0.95,
            "urgency": "normal",
            "recommended_agent": "EduMentor",
            "reasoning": "Educational query about machine learning"
        })
        return mock_manager
    
    @pytest.fixture
    def alpha_router(self, mock_gemini_manager):
        """Create BHIV Alpha router with mocked dependencies"""
        return BHIVAlphaEnhanced(mock_gemini_manager)
    
    @pytest.mark.asyncio
    async def test_query_classification(self, alpha_router):
        """Test query classification functionality"""
        user_input = UserInput(
            raw_text="How do I learn machine learning?",
            voice_enabled=False,
            mode=InputMode.CHAT
        )
        
        user_info = UserInfo(
            user_id="test_user",
            auth_token="test_token",
            lang="en-IN",
            persona=UserPersona.STUDENT
        )
        
        classification, plan = await alpha_router.route_and_plan(user_input, user_info)
        
        assert classification.classification == ClassificationType.LEARNING_QUERY
        assert classification.recommended_agent == "EduMentor"
        assert classification.confidence > 0.5
        assert isinstance(plan, dict)
        assert "agent" in plan
        assert "priority" in plan
    
    def test_fallback_classification(self, alpha_router):
        """Test fallback classification when AI fails"""
        classification_engine = alpha_router.classification_engine
        
        user_input = UserInput(
            raw_text="I'm feeling very stressed and anxious",
            voice_enabled=False,
            mode=InputMode.CHAT
        )
        
        result = classification_engine._fallback_classification(user_input)
        
        assert result.classification == ClassificationType.WELLNESS_QUERY
        assert result.recommended_agent == "WellnessBot"
        assert "wellness" in result.reasoning.lower()
    
    def test_emergency_classification(self, alpha_router):
        """Test emergency query classification"""
        classification_engine = alpha_router.classification_engine
        
        user_input = UserInput(
            raw_text="I want to hurt myself and can't go on",
            voice_enabled=False,
            mode=InputMode.CHAT
        )
        
        result = classification_engine._fallback_classification(user_input)
        
        assert result.classification == ClassificationType.EMERGENCY
        assert result.urgency == UrgencyLevel.CRITICAL
        assert result.recommended_agent == "EmergencyBot"


class TestBHIVKnowledgebase:
    """Test BHIV Knowledgebase functionality"""
    
    @pytest.fixture
    def mock_data_ingestion(self):
        """Mock data ingestion"""
        mock_ingestion = Mock()
        mock_ingestion.initialize_embedding_model.return_value = Mock()
        mock_ingestion.load_existing_vector_stores.return_value = {
            "educational": Mock(),
            "wellness": Mock(),
            "vedas": Mock(),
            "unified": Mock()
        }
        return mock_ingestion
    
    @pytest.fixture
    def knowledgebase_manager(self, mock_data_ingestion):
        """Create knowledgebase manager with mocked dependencies"""
        return EnhancedKnowledgebaseManager(mock_data_ingestion)
    
    @pytest.mark.asyncio
    async def test_knowledgebase_initialization(self, knowledgebase_manager):
        """Test knowledgebase initialization"""
        await knowledgebase_manager.initialize()
        
        assert knowledgebase_manager.embedding_model is not None
        assert len(knowledgebase_manager.vector_stores) > 0
    
    def test_store_selection_for_classification(self, knowledgebase_manager):
        """Test vector store selection based on classification"""
        # Mock vector stores
        knowledgebase_manager.vector_stores = {
            "educational": Mock(),
            "wellness": Mock(),
            "vedas": Mock(),
            "unified": Mock()
        }
        
        # Test educational query
        store = knowledgebase_manager._select_store_for_classification(
            ClassificationType.LEARNING_QUERY
        )
        assert store == "educational"
        
        # Test wellness query
        store = knowledgebase_manager._select_store_for_classification(
            ClassificationType.WELLNESS_QUERY
        )
        assert store == "wellness"
        
        # Test spiritual query
        store = knowledgebase_manager._select_store_for_classification(
            ClassificationType.SPIRITUAL_QUERY
        )
        assert store == "vedas"
        
        # Test general query
        store = knowledgebase_manager._select_store_for_classification(
            ClassificationType.GENERAL_QUERY
        )
        assert store == "unified"
    
    def test_cache_key_generation(self, knowledgebase_manager):
        """Test cache key generation"""
        kb_query = KnowledgebaseQuery(top_k=5, vector_type=VectorType.FAISS)
        
        cache_key = knowledgebase_manager._generate_cache_key(
            "test query",
            kb_query,
            ClassificationType.LEARNING_QUERY
        )
        
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length


class TestBHIVVoiceVideoPipeline:
    """Test BHIV Voice and Video Pipeline"""
    
    @pytest.fixture
    def voice_orchestrator(self):
        """Create voice orchestrator"""
        return BHIVVoiceVideoOrchestrator()
    
    @pytest.mark.asyncio
    async def test_voice_orchestrator_initialization(self, voice_orchestrator):
        """Test voice orchestrator initialization"""
        # Mock provider initialization
        with patch.object(voice_orchestrator, '_configure_provider_hierarchy'):
            await voice_orchestrator.initialize()
        
        assert isinstance(voice_orchestrator.tts_providers, dict)
        assert isinstance(voice_orchestrator.performance_metrics, dict)
    
    def test_voice_parameters_preparation(self, voice_orchestrator):
        """Test voice parameter preparation"""
        user_info = UserInfo(
            user_id="test_user",
            auth_token="test_token",
            lang="hi-IN",
            persona=UserPersona.STUDENT
        )
        
        voice_preferences = {"speed": 1.2, "pitch": 1.1}
        
        params = voice_orchestrator._prepare_voice_parameters(user_info, voice_preferences)
        
        assert params["language"] == "hi-IN"
        assert params["speed"] == 1.2
        assert params["pitch"] == 1.1
    
    def test_cache_key_generation(self, voice_orchestrator):
        """Test voice cache key generation"""
        user_info = UserInfo(
            user_id="test_user",
            auth_token="test_token",
            lang="en-IN"
        )
        
        cache_key = voice_orchestrator._generate_cache_key(
            "Hello world",
            user_info,
            {"speed": 1.0}
        )
        
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length


class TestBHIVSessionManager:
    """Test BHIV Session Manager"""
    
    @pytest.fixture
    def session_manager(self):
        """Create session manager with test database"""
        return BHIVSessionManager(":memory:")  # Use in-memory SQLite for testing
    
    @pytest.mark.asyncio
    async def test_session_lifecycle(self, session_manager):
        """Test complete session lifecycle"""
        await session_manager.initialize()
        
        user_info = UserInfo(
            user_id="test_user",
            auth_token="test_token",
            lang="en-IN",
            persona=UserPersona.STUDENT
        )
        
        # Start session
        session_id = await session_manager.start_session(user_info)
        assert session_id is not None
        assert session_id in session_manager.active_sessions
        
        # Get session info
        session_info = await session_manager.get_session_info(session_id)
        assert session_info is not None
        assert session_info["user_info"]["user_id"] == "test_user"
        
        # End session
        await session_manager.end_session(session_id)
        assert session_manager.active_sessions[session_id]["status"] == "ended"
    
    def test_session_metrics_creation(self):
        """Test session metrics creation"""
        metrics = SessionMetrics(
            session_id="test_session",
            user_id="test_user",
            start_time=datetime.now()
        )
        
        assert metrics.session_id == "test_session"
        assert metrics.user_id == "test_user"
        assert metrics.total_interactions == 0
        assert isinstance(metrics.classifications_used, dict)
        assert isinstance(metrics.agents_used, dict)
        assert isinstance(metrics.reward_scores, list)
    
    def test_interaction_record_creation(self):
        """Test interaction record creation"""
        interaction = InteractionRecord(
            interaction_id="test_interaction",
            session_id="test_session",
            user_id="test_user",
            timestamp=datetime.now(),
            request_data={"query": "test"},
            response_data={"response": "test response"},
            classification="learning-query",
            agent_used="EduMentor",
            processing_time_ms=1500.0,
            reward_score=0.85,
            error_flag=False
        )
        
        assert interaction.interaction_id == "test_interaction"
        assert interaction.classification == "learning-query"
        assert interaction.agent_used == "EduMentor"
        assert interaction.reward_score == 0.85
        assert interaction.error_flag is False


class TestBHIVAPIGateway:
    """Test BHIV API Gateway"""
    
    @pytest.fixture
    def api_gateway(self):
        """Create API gateway"""
        return BHIVAPIGateway()
    
    def test_authentication_manager(self, api_gateway):
        """Test authentication manager"""
        auth_manager = api_gateway.auth_manager
        
        # Test API key verification
        api_key_info = auth_manager.verify_api_key("bhiv-core-key")
        assert api_key_info is not None
        assert api_key_info["name"] == "BHIV Core Service"
        assert "admin" in api_key_info["permissions"]
        
        # Test invalid API key
        invalid_info = auth_manager.verify_api_key("invalid-key")
        assert invalid_info is None
    
    def test_permission_checking(self, api_gateway):
        """Test permission checking"""
        auth_manager = api_gateway.auth_manager
        
        # Test admin permission
        assert auth_manager.check_permissions(["admin"], "read") is True
        assert auth_manager.check_permissions(["admin"], "write") is True
        
        # Test specific permission
        assert auth_manager.check_permissions(["read", "write"], "read") is True
        assert auth_manager.check_permissions(["read"], "write") is False
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, api_gateway):
        """Test rate limiting functionality"""
        rate_limiter = api_gateway.rate_limiter
        
        # Test rate limiting
        is_limited, rate_info = await rate_limiter.is_rate_limited("test_user", 5, 3600)
        
        assert is_limited is False
        assert isinstance(rate_info, dict)
        assert "limit" in rate_info
        assert "remaining" in rate_info
        assert "current_count" in rate_info


class TestBHIVEndToEndFlow:
    """Test complete BHIV Core end-to-end flow"""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator with all dependencies"""
        orchestrator = Mock(spec=BHIVCoreOrchestrator)
        
        # Mock the process_bhiv_request method
        async def mock_process_request(request):
            return BHIVCoreResponse(
                session_id=request.session_id,
                timestamp=datetime.now().isoformat(),
                user_info=request.user_info,
                input=request.input,
                agent_directives=AgentDirectives(
                    classification=ClassificationType.LEARNING_QUERY,
                    urgency=UrgencyLevel.NORMAL,
                    trigger_module=TriggerModule.KNOWLEDGEBASE,
                    agent_name="EduMentor",
                    expected_output_format=OutputFormat.RICH_TEXT_WITH_TTS
                ),
                output=OutputResponse(
                    response_text="This is a test response about machine learning.",
                    confidence_score=0.9,
                    sources=["Educational Database"],
                    followup_suggestions=["Would you like to learn more?"]
                ),
                logs=SystemLogs(
                    agent_chain_log=["classification: learning-query", "agent: EduMentor"],
                    reward_score=0.85,
                    error_flag=False,
                    processing_time_ms=1500.0
                )
            )
        
        orchestrator.process_bhiv_request = AsyncMock(side_effect=mock_process_request)
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_complete_bhiv_flow(self, mock_orchestrator):
        """Test complete BHIV Core flow from request to response"""
        
        # Create BHIV request
        request = create_bhiv_request(
            user_id="test_user",
            raw_text="How do I learn machine learning?",
            voice_enabled=True,
            mode=InputMode.CHAT,
            persona=UserPersona.STUDENT,
            lang="en-IN"
        )
        
        # Process request
        response = await mock_orchestrator.process_bhiv_request(request)
        
        # Validate response
        assert isinstance(response, BHIVCoreResponse)
        assert response.session_id == request.session_id
        assert response.user_info.user_id == "test_user"
        assert response.input.raw_text == "How do I learn machine learning?"
        assert response.agent_directives.classification == ClassificationType.LEARNING_QUERY
        assert response.agent_directives.agent_name == "EduMentor"
        assert response.output.response_text is not None
        assert response.output.confidence_score > 0.5
        assert response.logs.reward_score > 0.0
        assert response.logs.error_flag is False
        assert response.logs.processing_time_ms > 0
    
    def test_schema_serialization(self):
        """Test BHIV schema serialization and deserialization"""
        
        # Create request
        request = create_bhiv_request(
            user_id="test_user",
            raw_text="Test query",
            voice_enabled=False
        )
        
        # Serialize to JSON
        request_json = request.model_dump_json()
        assert isinstance(request_json, str)
        
        # Deserialize from JSON
        request_dict = json.loads(request_json)
        reconstructed_request = BHIVCoreRequest(**request_dict)
        
        assert reconstructed_request.user_info.user_id == "test_user"
        assert reconstructed_request.input.raw_text == "Test query"
        assert reconstructed_request.input.voice_enabled is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
