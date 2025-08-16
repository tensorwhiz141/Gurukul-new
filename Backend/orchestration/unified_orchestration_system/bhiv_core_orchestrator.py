"""
BHIV Core Orchestrator Service
Main orchestrator that manages the complete BHIV Core flow:
USER → NLP INPUT → AGENTIC LOGIC → CONTEXT RETRIEVAL → RESPONSE → VOICE → VIDEO

This service coordinates between all BHIV components:
- BHIV Core (this orchestrator)
- BHIV Alpha (agent router + planner)
- BHIV Knowledgebase (contextual vector retriever)
- Uniguru/Gurukul Interface (frontend)
- External APIs and services
"""

import os
import sys
import json
import uuid
import logging
import asyncio
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Local imports
from bhiv_core_schema import (
    BHIVCoreRequest, BHIVCoreResponse, UserInfo, UserInput, AgentDirectives,
    KnowledgebaseQuery, LLMRequest, OutputResponse, SystemLogs, LogEntry,
    ClassificationType, UrgencyLevel, TriggerModule, OutputFormat,
    AgentClassificationResult, KnowledgebaseResult, VoiceProcessingResult,
    create_bhiv_request, create_agent_directives
)
from orchestration_api import UnifiedOrchestrationEngine, GeminiAPIManager
from data_ingestion import UnifiedDataIngestion

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BHIVAlphaRouter:
    """
    BHIV Alpha - Agent Router and Planner
    Handles intelligent agent routing and planning based on user input
    """
    
    def __init__(self, gemini_manager: GeminiAPIManager):
        self.gemini_manager = gemini_manager
        self.classification_cache = {}
        
    async def classify_query(self, user_input: UserInput, user_info: UserInfo) -> AgentClassificationResult:
        """Classify user query and determine appropriate agent routing"""
        try:
            # Create classification prompt
            classification_prompt = f"""
            You are BHIV Alpha, an intelligent agent router and planner for an educational platform.
            
            Analyze this user query and classify it appropriately:
            
            USER QUERY: "{user_input.raw_text}"
            USER CONTEXT:
            - Language: {user_info.lang}
            - Persona: {user_info.persona}
            - Voice Enabled: {user_input.voice_enabled}
            - Mode: {user_input.mode}
            - Course Context: {user_input.context.course if user_input.context else 'None'}
            
            Classify this query into one of these categories:
            1. learning-query: Educational content, lessons, explanations, academic help
            2. wellness-query: Mental health, stress, emotional support, physical wellness
            3. spiritual-query: Vedic wisdom, spiritual guidance, philosophical questions
            4. general-query: General conversation, greetings, basic information
            5. emergency: Urgent mental health crisis, immediate help needed
            
            Determine urgency level:
            - low: General information, casual learning
            - normal: Standard educational or wellness queries
            - high: Struggling with learning, moderate stress/anxiety
            - critical: Emergency situations, severe distress
            
            Recommend the best agent:
            - GuruAgent: For spiritual/vedic wisdom queries
            - EduMentor: For educational content and learning
            - WellnessBot: For health and wellness support
            - GeneralBot: For general conversation
            - EmergencyBot: For crisis situations
            
            Respond in JSON format:
            {{
                "classification": "learning-query",
                "confidence": 0.95,
                "urgency": "normal",
                "recommended_agent": "EduMentor",
                "reasoning": "User is asking about a specific educational topic and needs learning support"
            }}
            """
            
            # Get classification from Gemini
            response = self.gemini_manager.generate_content(classification_prompt)
            
            if response:
                try:
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        classification_data = json.loads(json_match.group())
                        
                        return AgentClassificationResult(
                            classification=ClassificationType(classification_data.get('classification', 'general-query')),
                            confidence=float(classification_data.get('confidence', 0.5)),
                            urgency=UrgencyLevel(classification_data.get('urgency', 'normal')),
                            recommended_agent=classification_data.get('recommended_agent', 'GeneralBot'),
                            reasoning=classification_data.get('reasoning', 'Automatic classification')
                        )
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse classification response: {e}")
            
            # Fallback classification
            return self._fallback_classification(user_input)
            
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            return self._fallback_classification(user_input)
    
    def _fallback_classification(self, user_input: UserInput) -> AgentClassificationResult:
        """Fallback classification using simple keyword matching"""
        query_lower = user_input.raw_text.lower()
        
        # Educational keywords
        edu_keywords = ['learn', 'study', 'explain', 'teach', 'lesson', 'homework', 'quiz', 'test', 'subject']
        # Wellness keywords
        wellness_keywords = ['stress', 'anxiety', 'mood', 'health', 'wellness', 'tired', 'sad', 'worried']
        # Spiritual keywords
        spiritual_keywords = ['vedas', 'spiritual', 'meditation', 'wisdom', 'philosophy', 'meaning', 'purpose']
        # Emergency keywords
        emergency_keywords = ['help', 'crisis', 'emergency', 'urgent', 'suicide', 'harm', 'danger']
        
        if any(keyword in query_lower for keyword in emergency_keywords):
            return AgentClassificationResult(
                classification=ClassificationType.EMERGENCY,
                confidence=0.9,
                urgency=UrgencyLevel.CRITICAL,
                recommended_agent="EmergencyBot",
                reasoning="Emergency keywords detected"
            )
        elif any(keyword in query_lower for keyword in edu_keywords):
            return AgentClassificationResult(
                classification=ClassificationType.LEARNING_QUERY,
                confidence=0.7,
                urgency=UrgencyLevel.NORMAL,
                recommended_agent="EduMentor",
                reasoning="Educational keywords detected"
            )
        elif any(keyword in query_lower for keyword in wellness_keywords):
            return AgentClassificationResult(
                classification=ClassificationType.WELLNESS_QUERY,
                confidence=0.7,
                urgency=UrgencyLevel.NORMAL,
                recommended_agent="WellnessBot",
                reasoning="Wellness keywords detected"
            )
        elif any(keyword in query_lower for keyword in spiritual_keywords):
            return AgentClassificationResult(
                classification=ClassificationType.SPIRITUAL_QUERY,
                confidence=0.7,
                urgency=UrgencyLevel.NORMAL,
                recommended_agent="GuruAgent",
                reasoning="Spiritual keywords detected"
            )
        else:
            return AgentClassificationResult(
                classification=ClassificationType.GENERAL_QUERY,
                confidence=0.5,
                urgency=UrgencyLevel.LOW,
                recommended_agent="GeneralBot",
                reasoning="No specific keywords detected, defaulting to general"
            )


class BHIVKnowledgebase:
    """
    BHIV Knowledgebase - Contextual Vector Retriever
    Handles vector search and knowledge retrieval
    """
    
    def __init__(self, orchestration_engine: UnifiedOrchestrationEngine):
        self.orchestration_engine = orchestration_engine
        self.vector_stores = {}
        
    async def initialize(self):
        """Initialize the knowledgebase"""
        await self.orchestration_engine.initialize()
        self.vector_stores = self.orchestration_engine.vector_stores
        
    async def query_knowledgebase(
        self, 
        query: str, 
        kb_query: KnowledgebaseQuery,
        agent_classification: AgentClassificationResult
    ) -> KnowledgebaseResult:
        """Query the knowledgebase based on classification and parameters"""
        try:
            start_time = datetime.now()
            
            # Select appropriate vector store based on classification
            store_name = self._select_vector_store(agent_classification.classification)
            
            if store_name not in self.vector_stores:
                logger.warning(f"Vector store '{store_name}' not found, using unified store")
                store_name = 'unified' if 'unified' in self.vector_stores else list(self.vector_stores.keys())[0]
            
            # Perform vector search
            if store_name in self.vector_stores:
                retriever = self.vector_stores[store_name].as_retriever(
                    search_kwargs={"k": kb_query.top_k}
                )
                relevant_docs = retriever.get_relevant_documents(query)
                
                # Format results
                documents = []
                for doc in relevant_docs:
                    documents.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get('source', 'unknown')
                    })
                
                query_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return KnowledgebaseResult(
                    documents=documents,
                    total_results=len(documents),
                    query_time_ms=query_time,
                    embedding_used=self.orchestration_engine.embedding_model.__class__.__name__ if self.orchestration_engine.embedding_model else None,
                    filters_applied=kb_query.filters.dict() if kb_query.filters else None
                )
            else:
                return KnowledgebaseResult(
                    documents=[],
                    total_results=0,
                    query_time_ms=0,
                    embedding_used=None,
                    filters_applied=None
                )
                
        except Exception as e:
            logger.error(f"Error querying knowledgebase: {e}")
            return KnowledgebaseResult(
                documents=[],
                total_results=0,
                query_time_ms=0,
                embedding_used=None,
                filters_applied=None
            )
    
    def _select_vector_store(self, classification: ClassificationType) -> str:
        """Select appropriate vector store based on classification"""
        if classification == ClassificationType.SPIRITUAL_QUERY:
            return 'vedas'
        elif classification == ClassificationType.WELLNESS_QUERY:
            return 'wellness'
        elif classification == ClassificationType.LEARNING_QUERY:
            return 'educational'
        else:
            return 'unified'


class BHIVVoiceProcessor:
    """
    BHIV Voice Processor - Handles voice response generation
    """

    def __init__(self):
        self.tts_service_url = os.getenv("TTS_SERVICE_URL", "http://localhost:8005")

    async def generate_voice_response(
        self,
        text: str,
        user_info: UserInfo,
        voice_enabled: bool = True
    ) -> VoiceProcessingResult:
        """Generate voice response from text"""
        if not voice_enabled:
            return VoiceProcessingResult()

        try:
            start_time = datetime.now()

            # Call TTS service
            response = requests.post(
                f"{self.tts_service_url}/api/v1/generate-speech",
                json={
                    "text": text,
                    "language": user_info.lang,
                    "voice_model": "default",
                    "speed": 1.0,
                    "pitch": 1.0
                },
                timeout=30
            )

            if response.status_code == 200:
                tts_data = response.json()
                processing_time = (datetime.now() - start_time).total_seconds() * 1000

                return VoiceProcessingResult(
                    audio_url=tts_data.get('audio_url'),
                    duration_seconds=tts_data.get('duration'),
                    voice_model=tts_data.get('voice_model', 'default'),
                    processing_time_ms=processing_time
                )
            else:
                return VoiceProcessingResult(
                    error=f"TTS service returned {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error generating voice response: {e}")
            return VoiceProcessingResult(
                error=str(e)
            )


class BHIVCoreOrchestrator:
    """
    Main BHIV Core Orchestrator
    Manages the complete flow: USER → NLP INPUT → AGENTIC LOGIC → CONTEXT RETRIEVAL → RESPONSE → VOICE → VIDEO
    """

    def __init__(self):
        self.orchestration_engine = UnifiedOrchestrationEngine()
        self.gemini_manager = GeminiAPIManager()
        self.alpha_router = BHIVAlphaRouter(self.gemini_manager)
        self.knowledgebase = BHIVKnowledgebase(self.orchestration_engine)
        self.voice_processor = BHIVVoiceProcessor()
        self.session_store = {}

    async def initialize(self):
        """Initialize all BHIV components"""
        logger.info("Initializing BHIV Core Orchestrator...")

        # Initialize orchestration engine
        await self.orchestration_engine.initialize()

        # Initialize knowledgebase
        await self.knowledgebase.initialize()

        logger.info("BHIV Core Orchestrator initialized successfully")

    async def process_bhiv_request(self, request: BHIVCoreRequest) -> BHIVCoreResponse:
        """
        Process a complete BHIV Core request through the full pipeline
        USER → NLP INPUT → AGENTIC LOGIC → CONTEXT RETRIEVAL → RESPONSE → VOICE → VIDEO
        """
        start_time = datetime.now()
        logs = SystemLogs()

        try:
            # Store session
            self.session_store[request.session_id] = {
                'start_time': start_time,
                'user_info': request.user_info,
                'status': 'processing'
            }

            # Log start
            logs.agent_chain_log.append("bhiv-core: request received")
            logs.detailed_logs.append(LogEntry(
                level="info",
                message="BHIV Core request processing started",
                component="bhiv-core",
                data={"session_id": request.session_id}
            ))

            # Step 1: NLP INPUT - Process and validate user input
            logs.agent_chain_log.append("nlp-input: processing user input")
            processed_input = await self._process_nlp_input(request.input, request.user_info)

            # Step 2: AGENTIC LOGIC - Route and plan with BHIV Alpha
            logs.agent_chain_log.append("agentic-logic: routing with bhiv-alpha")
            classification_result = await self.alpha_router.classify_query(processed_input, request.user_info)

            # Create agent directives
            agent_directives = create_agent_directives(
                classification=classification_result.classification,
                agent_name=classification_result.recommended_agent,
                urgency=classification_result.urgency,
                confidence=classification_result.confidence
            )

            # Step 3: CONTEXT RETRIEVAL - Query knowledgebase
            logs.agent_chain_log.append("context-retrieval: querying knowledgebase")
            kb_query = request.knowledgebase_query or KnowledgebaseQuery()
            kb_result = await self.knowledgebase.query_knowledgebase(
                processed_input.raw_text, kb_query, classification_result
            )

            # Step 4: RESPONSE - Generate response using appropriate agent
            logs.agent_chain_log.append(f"response: generating with {agent_directives.agent_name}")
            llm_request = request.llm_request or LLMRequest()
            response_output = await self._generate_response(
                processed_input, agent_directives, kb_result, llm_request, request.user_info
            )

            # Step 5: VOICE - Generate voice response if enabled
            if processed_input.voice_enabled:
                logs.agent_chain_log.append("voice: generating audio response")
                voice_result = await self.voice_processor.generate_voice_response(
                    response_output.response_text, request.user_info, processed_input.voice_enabled
                )
                if voice_result.audio_url:
                    response_output.voice_response_url = voice_result.audio_url

            # Step 6: VIDEO - Future implementation for video responses
            logs.agent_chain_log.append("video: placeholder for future video integration")

            # Calculate processing time and reward score
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logs.processing_time_ms = processing_time
            logs.reward_score = self._calculate_reward_score(classification_result, kb_result, response_output)

            # Update session
            self.session_store[request.session_id]['status'] = 'completed'
            self.session_store[request.session_id]['end_time'] = datetime.now()

            # Create response
            return BHIVCoreResponse(
                session_id=request.session_id,
                timestamp=datetime.now().isoformat(),
                user_info=request.user_info,
                input=processed_input,
                agent_directives=agent_directives,
                knowledgebase_query=kb_query,
                llm_request=llm_request,
                output=response_output,
                logs=logs
            )

        except Exception as e:
            logger.error(f"Error processing BHIV request: {e}")
            logs.error_flag = True
            logs.detailed_logs.append(LogEntry(
                level="error",
                message=f"BHIV Core processing failed: {str(e)}",
                component="bhiv-core",
                data={"session_id": request.session_id, "error": str(e)}
            ))

            # Update session with error
            self.session_store[request.session_id]['status'] = 'error'
            self.session_store[request.session_id]['error'] = str(e)

            raise HTTPException(status_code=500, detail=f"BHIV Core processing failed: {str(e)}")

    async def _process_nlp_input(self, user_input: UserInput, user_info: UserInfo) -> UserInput:
        """Process and enhance user input"""
        # For now, return as-is, but this could include:
        # - Text preprocessing
        # - Language detection
        # - Intent extraction
        # - Context enhancement
        return user_input

    async def _generate_response(
        self,
        user_input: UserInput,
        agent_directives: AgentDirectives,
        kb_result: KnowledgebaseResult,
        llm_request: LLMRequest,
        user_info: UserInfo
    ) -> OutputResponse:
        """Generate response using the appropriate agent"""
        try:
            # Route to appropriate agent based on classification
            if agent_directives.classification == ClassificationType.SPIRITUAL_QUERY:
                return await self._call_spiritual_agent(user_input, kb_result, llm_request, user_info)
            elif agent_directives.classification == ClassificationType.WELLNESS_QUERY:
                return await self._call_wellness_agent(user_input, kb_result, llm_request, user_info)
            elif agent_directives.classification == ClassificationType.LEARNING_QUERY:
                return await self._call_educational_agent(user_input, kb_result, llm_request, user_info)
            else:
                return await self._call_general_agent(user_input, kb_result, llm_request, user_info)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return OutputResponse(
                response_text="I apologize, but I encountered an error while processing your request. Please try again.",
                confidence_score=0.1,
                sources=[],
                followup_suggestions=["Please try rephrasing your question", "Contact support if the issue persists"]
            )

    async def _call_spiritual_agent(
        self, user_input: UserInput, kb_result: KnowledgebaseResult,
        llm_request: LLMRequest, user_info: UserInfo
    ) -> OutputResponse:
        """Call spiritual/vedas agent"""
        try:
            # Use existing orchestration engine's ask_vedas method
            vedas_response = await self.orchestration_engine.ask_vedas(
                user_input.raw_text, user_info.user_id
            )

            wisdom = vedas_response.get('wisdom', {})
            sources = [doc.get('text', '')[:100] + '...' for doc in vedas_response.get('source_documents', [])]

            return OutputResponse(
                response_text=wisdom.get('core_teaching', 'Ancient wisdom guides us to seek truth within ourselves.'),
                confidence_score=0.85,
                sources=sources,
                followup_suggestions=[
                    "Would you like to explore this teaching further?",
                    "How can you apply this wisdom in your daily life?",
                    "Are there related spiritual concepts you'd like to understand?"
                ],
                metadata={
                    'agent_type': 'spiritual',
                    'wisdom_type': 'vedic',
                    'practical_application': wisdom.get('practical_application'),
                    'philosophical_insight': wisdom.get('philosophical_insight')
                }
            )

        except Exception as e:
            logger.error(f"Error calling spiritual agent: {e}")
            return self._fallback_response("spiritual guidance")

    async def _call_wellness_agent(
        self, user_input: UserInput, kb_result: KnowledgebaseResult,
        llm_request: LLMRequest, user_info: UserInfo
    ) -> OutputResponse:
        """Call wellness agent"""
        try:
            # Use existing orchestration engine's ask_wellness method
            wellness_response = await self.orchestration_engine.ask_wellness(
                user_input.raw_text, user_info.user_id
            )

            advice = wellness_response.get('advice', {})
            sources = [doc.get('text', '')[:100] + '...' for doc in wellness_response.get('source_documents', [])]

            return OutputResponse(
                response_text=advice.get('main_advice', 'Taking care of your wellbeing is important. Consider speaking with a healthcare professional.'),
                confidence_score=0.8,
                sources=sources,
                followup_suggestions=advice.get('tips', [
                    "Practice mindfulness and self-care",
                    "Consider speaking with a counselor",
                    "Take time for activities you enjoy"
                ]),
                metadata={
                    'agent_type': 'wellness',
                    'practical_steps': advice.get('practical_steps', []),
                    'emotional_support': wellness_response.get('emotional_nudge', {})
                }
            )

        except Exception as e:
            logger.error(f"Error calling wellness agent: {e}")
            return self._fallback_response("wellness support")

    async def _call_educational_agent(
        self, user_input: UserInput, kb_result: KnowledgebaseResult,
        llm_request: LLMRequest, user_info: UserInfo
    ) -> OutputResponse:
        """Call educational agent"""
        try:
            # Use existing orchestration engine's ask_edumentor method
            edu_response = await self.orchestration_engine.ask_edumentor(
                user_input.raw_text, user_info.user_id
            )

            explanation = edu_response.get('explanation', {})
            sources = [doc.get('text', '')[:100] + '...' for doc in edu_response.get('source_documents', [])]

            return OutputResponse(
                response_text=explanation.get('main_explanation', 'Let me help you understand this topic better.'),
                confidence_score=0.9,
                sources=sources,
                followup_suggestions=explanation.get('study_tips', [
                    "Would you like more examples?",
                    "Should we explore related concepts?",
                    "Do you want to test your understanding with a quiz?"
                ]),
                metadata={
                    'agent_type': 'educational',
                    'learning_objectives': explanation.get('learning_objectives', []),
                    'examples': explanation.get('examples', []),
                    'study_tips': explanation.get('study_tips', [])
                }
            )

        except Exception as e:
            logger.error(f"Error calling educational agent: {e}")
            return self._fallback_response("educational content")

    async def _call_general_agent(
        self, user_input: UserInput, kb_result: KnowledgebaseResult,
        llm_request: LLMRequest, user_info: UserInfo
    ) -> OutputResponse:
        """Call general conversation agent"""
        try:
            # Create context from knowledgebase results
            context = "\n".join([doc.get('content', '')[:300] for doc in kb_result.documents[:3]])

            # Generate response using Gemini
            prompt = f"""
            You are a helpful AI assistant for an educational platform called Gurukul.

            Context from knowledge base:
            {context}

            User question: {user_input.raw_text}

            Provide a helpful, friendly response that:
            1. Directly addresses their question
            2. Uses the context if relevant
            3. Maintains a supportive, educational tone
            4. Offers to help further if needed

            Keep the response concise but informative.
            """

            response_text = self.gemini_manager.generate_content(prompt)

            if not response_text:
                response_text = "Hello! I'm here to help you with your questions. How can I assist you today?"

            return OutputResponse(
                response_text=response_text,
                confidence_score=0.7,
                sources=[doc.get('source', 'knowledge base') for doc in kb_result.documents[:3]],
                followup_suggestions=[
                    "Is there anything specific you'd like to learn about?",
                    "Would you like me to explain any concepts in more detail?",
                    "How else can I help you today?"
                ],
                metadata={
                    'agent_type': 'general',
                    'context_used': len(kb_result.documents) > 0
                }
            )

        except Exception as e:
            logger.error(f"Error calling general agent: {e}")
            return self._fallback_response("general assistance")

    def _fallback_response(self, agent_type: str) -> OutputResponse:
        """Generate fallback response when agents fail"""
        return OutputResponse(
            response_text=f"I apologize, but I'm having trouble providing {agent_type} right now. Please try again in a moment.",
            confidence_score=0.3,
            sources=[],
            followup_suggestions=[
                "Please try rephrasing your question",
                "Contact support if the issue persists",
                "Try again in a few moments"
            ],
            metadata={'agent_type': 'fallback', 'original_type': agent_type}
        )

    def _calculate_reward_score(
        self, classification: AgentClassificationResult,
        kb_result: KnowledgebaseResult,
        response: OutputResponse
    ) -> float:
        """Calculate reward score based on processing quality"""
        score = 0.5  # Base score

        # Add points for high classification confidence
        score += classification.confidence * 0.3

        # Add points for successful knowledge retrieval
        if kb_result.total_results > 0:
            score += 0.2

        # Add points for high response confidence
        score += response.confidence_score * 0.3

        # Ensure score is between 0 and 1
        return min(max(score, 0.0), 1.0)

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        return self.session_store.get(session_id)

    async def health_check(self) -> Dict[str, Any]:
        """Health check for BHIV Core"""
        return {
            "status": "healthy",
            "components": {
                "orchestration_engine": "initialized" if self.orchestration_engine else "not_initialized",
                "gemini_manager": "available" if self.gemini_manager.is_available() else "unavailable",
                "knowledgebase": "ready" if self.knowledgebase.vector_stores else "not_ready",
                "voice_processor": "ready"
            },
            "active_sessions": len(self.session_store),
            "timestamp": datetime.now().isoformat()
        }
