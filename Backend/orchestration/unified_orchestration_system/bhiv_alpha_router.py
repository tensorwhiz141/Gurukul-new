"""
BHIV Alpha - Enhanced Agent Router and Planner
Advanced intelligent agent routing and planning system for BHIV Core

This component handles:
- Advanced query classification using multiple techniques
- Intelligent agent selection and routing
- Urgency assessment and priority handling
- Context-aware planning and decision making
- Learning from user interactions for improved routing
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque

# Local imports
from bhiv_core_schema import (
    UserInfo, UserInput, ClassificationType, UrgencyLevel, TriggerModule,
    AgentClassificationResult, create_agent_directives
)
from orchestration_api import GeminiAPIManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedClassificationEngine:
    """
    Advanced classification engine using multiple techniques
    """
    
    def __init__(self, gemini_manager: GeminiAPIManager):
        self.gemini_manager = gemini_manager
        self.keyword_patterns = self._load_keyword_patterns()
        self.classification_history = defaultdict(list)
        self.user_patterns = defaultdict(dict)
        
    def _load_keyword_patterns(self) -> Dict[ClassificationType, Dict[str, List[str]]]:
        """Load keyword patterns for classification"""
        return {
            ClassificationType.LEARNING_QUERY: {
                "primary": ["learn", "study", "explain", "teach", "lesson", "understand", "concept", "theory"],
                "secondary": ["homework", "assignment", "quiz", "test", "exam", "subject", "topic", "chapter"],
                "academic": ["mathematics", "science", "history", "literature", "physics", "chemistry", "biology"],
                "skills": ["how to", "tutorial", "guide", "steps", "method", "process", "technique"]
            },
            ClassificationType.WELLNESS_QUERY: {
                "mental": ["stress", "anxiety", "depression", "mood", "mental health", "emotional", "feelings"],
                "physical": ["health", "wellness", "fitness", "exercise", "diet", "nutrition", "sleep"],
                "support": ["help", "support", "counseling", "therapy", "advice", "guidance", "coping"],
                "symptoms": ["tired", "exhausted", "overwhelmed", "sad", "worried", "scared", "angry"]
            },
            ClassificationType.SPIRITUAL_QUERY: {
                "vedic": ["vedas", "upanishads", "bhagavad gita", "sanskrit", "mantra", "yoga", "dharma"],
                "spiritual": ["spiritual", "meditation", "mindfulness", "consciousness", "enlightenment"],
                "philosophical": ["meaning", "purpose", "existence", "truth", "wisdom", "philosophy"],
                "practices": ["prayer", "chanting", "ritual", "ceremony", "pilgrimage", "devotion"]
            },
            ClassificationType.EMERGENCY: {
                "crisis": ["suicide", "kill myself", "end it all", "can't go on", "want to die"],
                "urgent": ["emergency", "crisis", "urgent", "immediate help", "right now"],
                "harm": ["hurt myself", "self harm", "cutting", "dangerous", "unsafe"],
                "severe": ["severe", "extreme", "unbearable", "can't cope", "breaking down"]
            },
            ClassificationType.GENERAL_QUERY: {
                "greetings": ["hello", "hi", "hey", "good morning", "good evening", "how are you"],
                "basic": ["what", "who", "when", "where", "why", "how", "tell me about"],
                "casual": ["chat", "talk", "conversation", "discuss", "opinion", "think"],
                "info": ["information", "details", "facts", "data", "statistics", "news"]
            }
        }
    
    async def classify_with_multiple_methods(
        self, 
        user_input: UserInput, 
        user_info: UserInfo
    ) -> AgentClassificationResult:
        """Classify using multiple methods and combine results"""
        
        # Method 1: Keyword-based classification
        keyword_result = self._classify_by_keywords(user_input.raw_text)
        
        # Method 2: AI-based classification
        ai_result = await self._classify_with_ai(user_input, user_info)
        
        # Method 3: User pattern-based classification
        pattern_result = self._classify_by_user_patterns(user_input, user_info)
        
        # Method 4: Context-based classification
        context_result = self._classify_by_context(user_input)
        
        # Combine results using weighted scoring
        final_result = self._combine_classification_results([
            (keyword_result, 0.3),
            (ai_result, 0.4),
            (pattern_result, 0.2),
            (context_result, 0.1)
        ])
        
        # Store classification for learning
        self._store_classification_result(user_info.user_id, user_input, final_result)
        
        return final_result
    
    def _classify_by_keywords(self, text: str) -> AgentClassificationResult:
        """Classify based on keyword patterns"""
        text_lower = text.lower()
        scores = defaultdict(float)
        
        for classification, categories in self.keyword_patterns.items():
            for category, keywords in categories.items():
                weight = 1.0 if category == "primary" else 0.7 if category == "secondary" else 0.5
                
                for keyword in keywords:
                    if keyword in text_lower:
                        scores[classification] += weight
        
        if not scores:
            return AgentClassificationResult(
                classification=ClassificationType.GENERAL_QUERY,
                confidence=0.3,
                urgency=UrgencyLevel.LOW,
                recommended_agent="GeneralBot",
                reasoning="No specific keywords detected"
            )
        
        # Get best classification
        best_classification = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_classification] / 3.0, 1.0)  # Normalize
        
        # Determine urgency and agent
        urgency = self._determine_urgency(best_classification, text_lower)
        agent = self._select_agent(best_classification)
        
        return AgentClassificationResult(
            classification=best_classification,
            confidence=confidence,
            urgency=urgency,
            recommended_agent=agent,
            reasoning=f"Keyword-based classification with score {scores[best_classification]:.2f}"
        )
    
    async def _classify_with_ai(self, user_input: UserInput, user_info: UserInfo) -> AgentClassificationResult:
        """Classify using AI/LLM"""
        try:
            prompt = f"""
            You are BHIV Alpha, an expert AI classifier for an educational platform.
            
            Analyze this user query and provide classification:
            
            USER QUERY: "{user_input.raw_text}"
            USER CONTEXT:
            - Language: {user_info.lang}
            - Persona: {user_info.persona}
            - Previous context: {user_input.context.course if user_input.context else 'None'}
            
            Classify into exactly one category:
            1. learning-query: Educational content, academic help, explanations
            2. wellness-query: Mental/physical health, emotional support, wellness
            3. spiritual-query: Vedic wisdom, spiritual guidance, philosophy
            4. general-query: Casual conversation, basic information
            5. emergency: Crisis situations, urgent help needed
            
            Consider:
            - Intent and purpose of the query
            - Emotional tone and urgency indicators
            - Subject matter and domain
            - User's likely needs and expectations
            
            Respond with JSON:
            {{
                "classification": "learning-query",
                "confidence": 0.95,
                "urgency": "normal",
                "recommended_agent": "EduMentor",
                "reasoning": "Detailed explanation of classification decision"
            }}
            """
            
            response = self.gemini_manager.generate_content(prompt)
            
            if response:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        return AgentClassificationResult(
                            classification=ClassificationType(data.get('classification', 'general-query')),
                            confidence=float(data.get('confidence', 0.5)),
                            urgency=UrgencyLevel(data.get('urgency', 'normal')),
                            recommended_agent=data.get('recommended_agent', 'GeneralBot'),
                            reasoning=f"AI-based: {data.get('reasoning', 'AI classification')}"
                        )
                    except (json.JSONDecodeError, ValueError, KeyError):
                        pass
            
            # Fallback
            return self._fallback_ai_classification(user_input)
            
        except Exception as e:
            logger.warning(f"AI classification failed: {e}")
            return self._fallback_ai_classification(user_input)
    
    def _classify_by_user_patterns(self, user_input: UserInput, user_info: UserInfo) -> AgentClassificationResult:
        """Classify based on user's historical patterns"""
        user_id = user_info.user_id
        
        if user_id not in self.user_patterns:
            return AgentClassificationResult(
                classification=ClassificationType.GENERAL_QUERY,
                confidence=0.2,
                urgency=UrgencyLevel.NORMAL,
                recommended_agent="GeneralBot",
                reasoning="No user pattern history available"
            )
        
        patterns = self.user_patterns[user_id]
        
        # Get most common classification for this user
        if 'common_classifications' in patterns:
            common_class = max(patterns['common_classifications'].items(), key=lambda x: x[1])[0]
            confidence = min(patterns['common_classifications'][common_class] / 10.0, 0.8)
            
            return AgentClassificationResult(
                classification=ClassificationType(common_class),
                confidence=confidence,
                urgency=UrgencyLevel.NORMAL,
                recommended_agent=self._select_agent(ClassificationType(common_class)),
                reasoning=f"Based on user pattern: {common_class} ({patterns['common_classifications'][common_class]} times)"
            )
        
        return AgentClassificationResult(
            classification=ClassificationType.GENERAL_QUERY,
            confidence=0.2,
            urgency=UrgencyLevel.NORMAL,
            recommended_agent="GeneralBot",
            reasoning="Insufficient user pattern data"
        )
    
    def _classify_by_context(self, user_input: UserInput) -> AgentClassificationResult:
        """Classify based on context information"""
        if not user_input.context:
            return AgentClassificationResult(
                classification=ClassificationType.GENERAL_QUERY,
                confidence=0.1,
                urgency=UrgencyLevel.NORMAL,
                recommended_agent="GeneralBot",
                reasoning="No context available"
            )
        
        course = user_input.context.course
        if course:
            # Map course to classification
            if any(subject in course.lower() for subject in ['math', 'science', 'history', 'literature']):
                return AgentClassificationResult(
                    classification=ClassificationType.LEARNING_QUERY,
                    confidence=0.6,
                    urgency=UrgencyLevel.NORMAL,
                    recommended_agent="EduMentor",
                    reasoning=f"Context indicates educational course: {course}"
                )
            elif any(term in course.lower() for term in ['wellness', 'health', 'mindfulness']):
                return AgentClassificationResult(
                    classification=ClassificationType.WELLNESS_QUERY,
                    confidence=0.6,
                    urgency=UrgencyLevel.NORMAL,
                    recommended_agent="WellnessBot",
                    reasoning=f"Context indicates wellness course: {course}"
                )
            elif any(term in course.lower() for term in ['spiritual', 'vedas', 'philosophy']):
                return AgentClassificationResult(
                    classification=ClassificationType.SPIRITUAL_QUERY,
                    confidence=0.6,
                    urgency=UrgencyLevel.NORMAL,
                    recommended_agent="GuruAgent",
                    reasoning=f"Context indicates spiritual course: {course}"
                )
        
        return AgentClassificationResult(
            classification=ClassificationType.GENERAL_QUERY,
            confidence=0.3,
            urgency=UrgencyLevel.NORMAL,
            recommended_agent="GeneralBot",
            reasoning="Context available but not specific enough"
        )

    def _combine_classification_results(
        self,
        results: List[Tuple[AgentClassificationResult, float]]
    ) -> AgentClassificationResult:
        """Combine multiple classification results using weighted scoring"""

        # Calculate weighted scores for each classification
        classification_scores = defaultdict(float)
        confidence_scores = defaultdict(list)
        urgency_scores = defaultdict(list)
        agents = defaultdict(list)
        reasonings = []

        for result, weight in results:
            classification_scores[result.classification] += result.confidence * weight
            confidence_scores[result.classification].append(result.confidence)
            urgency_scores[result.classification].append(result.urgency)
            agents[result.classification].append(result.recommended_agent)
            reasonings.append(f"{result.reasoning} (weight: {weight})")

        # Get best classification
        best_classification = max(classification_scores.keys(), key=lambda k: classification_scores[k])

        # Calculate combined confidence
        combined_confidence = min(classification_scores[best_classification], 1.0)

        # Determine urgency (take highest urgency for safety)
        urgencies = urgency_scores[best_classification]
        urgency_priority = {UrgencyLevel.LOW: 1, UrgencyLevel.NORMAL: 2, UrgencyLevel.HIGH: 3, UrgencyLevel.CRITICAL: 4}
        best_urgency = max(urgencies, key=lambda u: urgency_priority[u])

        # Select most common agent
        agent_counts = defaultdict(int)
        for agent in agents[best_classification]:
            agent_counts[agent] += 1
        best_agent = max(agent_counts.keys(), key=lambda a: agent_counts[a])

        return AgentClassificationResult(
            classification=best_classification,
            confidence=combined_confidence,
            urgency=best_urgency,
            recommended_agent=best_agent,
            reasoning=f"Combined classification: {'; '.join(reasonings)}"
        )

    def _determine_urgency(self, classification: ClassificationType, text: str) -> UrgencyLevel:
        """Determine urgency level based on classification and text content"""

        # Emergency classification always gets critical urgency
        if classification == ClassificationType.EMERGENCY:
            return UrgencyLevel.CRITICAL

        # Check for urgency indicators in text
        high_urgency_words = ["urgent", "emergency", "immediate", "asap", "quickly", "right now", "help me"]
        critical_words = ["crisis", "can't cope", "breaking down", "severe", "extreme"]

        if any(word in text for word in critical_words):
            return UrgencyLevel.CRITICAL
        elif any(word in text for word in high_urgency_words):
            return UrgencyLevel.HIGH
        elif classification == ClassificationType.WELLNESS_QUERY:
            # Wellness queries default to normal but can be elevated
            stress_words = ["stress", "anxiety", "depressed", "overwhelmed"]
            if any(word in text for word in stress_words):
                return UrgencyLevel.HIGH
            return UrgencyLevel.NORMAL
        elif classification == ClassificationType.LEARNING_QUERY:
            # Learning queries are usually normal unless deadline-related
            deadline_words = ["exam tomorrow", "due today", "test soon", "deadline"]
            if any(word in text for word in deadline_words):
                return UrgencyLevel.HIGH
            return UrgencyLevel.NORMAL
        else:
            return UrgencyLevel.NORMAL

    def _select_agent(self, classification: ClassificationType) -> str:
        """Select appropriate agent based on classification"""
        agent_mapping = {
            ClassificationType.LEARNING_QUERY: "EduMentor",
            ClassificationType.WELLNESS_QUERY: "WellnessBot",
            ClassificationType.SPIRITUAL_QUERY: "GuruAgent",
            ClassificationType.EMERGENCY: "EmergencyBot",
            ClassificationType.GENERAL_QUERY: "GeneralBot"
        }
        return agent_mapping.get(classification, "GeneralBot")

    def _fallback_ai_classification(self, user_input: UserInput) -> AgentClassificationResult:
        """Fallback when AI classification fails"""
        return AgentClassificationResult(
            classification=ClassificationType.GENERAL_QUERY,
            confidence=0.4,
            urgency=UrgencyLevel.NORMAL,
            recommended_agent="GeneralBot",
            reasoning="AI classification failed, using fallback"
        )

    def _store_classification_result(
        self,
        user_id: str,
        user_input: UserInput,
        result: AgentClassificationResult
    ):
        """Store classification result for learning and pattern recognition"""

        # Store in classification history
        self.classification_history[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'query': user_input.raw_text,
            'classification': result.classification.value,
            'confidence': result.confidence,
            'agent': result.recommended_agent
        })

        # Keep only last 100 entries per user
        if len(self.classification_history[user_id]) > 100:
            self.classification_history[user_id] = self.classification_history[user_id][-100:]

        # Update user patterns
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {
                'common_classifications': defaultdict(int),
                'common_agents': defaultdict(int),
                'avg_confidence': 0.0,
                'total_queries': 0
            }

        patterns = self.user_patterns[user_id]
        patterns['common_classifications'][result.classification.value] += 1
        patterns['common_agents'][result.recommended_agent] += 1
        patterns['total_queries'] += 1

        # Update average confidence
        patterns['avg_confidence'] = (
            (patterns['avg_confidence'] * (patterns['total_queries'] - 1) + result.confidence)
            / patterns['total_queries']
        )


class BHIVAlphaEnhanced:
    """
    Enhanced BHIV Alpha Router with advanced planning capabilities
    """

    def __init__(self, gemini_manager: GeminiAPIManager):
        self.gemini_manager = gemini_manager
        self.classification_engine = AdvancedClassificationEngine(gemini_manager)
        self.routing_cache = {}
        self.performance_metrics = defaultdict(list)

    async def route_and_plan(
        self,
        user_input: UserInput,
        user_info: UserInfo
    ) -> Tuple[AgentClassificationResult, Dict[str, Any]]:
        """
        Advanced routing and planning with context awareness
        Returns classification result and execution plan
        """

        # Step 1: Advanced classification
        classification = await self.classification_engine.classify_with_multiple_methods(
            user_input, user_info
        )

        # Step 2: Create execution plan
        execution_plan = await self._create_execution_plan(classification, user_input, user_info)

        # Step 3: Cache routing decision
        cache_key = f"{user_info.user_id}:{hash(user_input.raw_text)}"
        self.routing_cache[cache_key] = {
            'classification': classification,
            'plan': execution_plan,
            'timestamp': datetime.now()
        }

        return classification, execution_plan

    async def _create_execution_plan(
        self,
        classification: AgentClassificationResult,
        user_input: UserInput,
        user_info: UserInfo
    ) -> Dict[str, Any]:
        """Create detailed execution plan based on classification"""

        plan = {
            'agent': classification.recommended_agent,
            'priority': self._get_priority_level(classification.urgency),
            'estimated_processing_time': self._estimate_processing_time(classification),
            'required_resources': self._determine_required_resources(classification),
            'fallback_agents': self._get_fallback_agents(classification),
            'context_requirements': self._determine_context_requirements(classification, user_input),
            'output_format': self._determine_output_format(classification, user_input),
            'post_processing': self._determine_post_processing(classification, user_input)
        }

        return plan

    def _get_priority_level(self, urgency: UrgencyLevel) -> int:
        """Convert urgency to numeric priority (higher = more urgent)"""
        priority_map = {
            UrgencyLevel.LOW: 1,
            UrgencyLevel.NORMAL: 2,
            UrgencyLevel.HIGH: 3,
            UrgencyLevel.CRITICAL: 4
        }
        return priority_map.get(urgency, 2)

    def _estimate_processing_time(self, classification: AgentClassificationResult) -> float:
        """Estimate processing time in seconds"""
        base_times = {
            ClassificationType.GENERAL_QUERY: 2.0,
            ClassificationType.LEARNING_QUERY: 5.0,
            ClassificationType.WELLNESS_QUERY: 4.0,
            ClassificationType.SPIRITUAL_QUERY: 6.0,
            ClassificationType.EMERGENCY: 1.0  # Fastest response for emergencies
        }

        base_time = base_times.get(classification.classification, 3.0)

        # Adjust based on confidence (lower confidence = more processing time)
        confidence_factor = 1.0 + (1.0 - classification.confidence) * 0.5

        return base_time * confidence_factor

    def _determine_required_resources(self, classification: AgentClassificationResult) -> List[str]:
        """Determine what resources are needed for processing"""
        resources = ['llm']  # All need LLM

        if classification.classification in [ClassificationType.LEARNING_QUERY, ClassificationType.SPIRITUAL_QUERY]:
            resources.append('knowledgebase')

        if classification.classification == ClassificationType.WELLNESS_QUERY:
            resources.extend(['knowledgebase', 'wellness_tools'])

        if classification.classification == ClassificationType.EMERGENCY:
            resources.extend(['emergency_protocols', 'crisis_resources'])

        return resources

    def _get_fallback_agents(self, classification: AgentClassificationResult) -> List[str]:
        """Get fallback agents if primary agent fails"""
        fallback_map = {
            "EduMentor": ["GeneralBot", "TutorBot"],
            "WellnessBot": ["GeneralBot", "EmotionalWellnessBot"],
            "GuruAgent": ["GeneralBot", "SpiritualBot"],
            "EmergencyBot": ["WellnessBot", "GeneralBot"],
            "GeneralBot": ["EduMentor"]
        }
        return fallback_map.get(classification.recommended_agent, ["GeneralBot"])

    def _determine_context_requirements(
        self,
        classification: AgentClassificationResult,
        user_input: UserInput
    ) -> Dict[str, Any]:
        """Determine what context is needed for processing"""
        requirements = {
            'user_history': classification.classification != ClassificationType.EMERGENCY,
            'session_context': True,
            'knowledgebase_context': classification.classification in [
                ClassificationType.LEARNING_QUERY,
                ClassificationType.SPIRITUAL_QUERY,
                ClassificationType.WELLNESS_QUERY
            ],
            'real_time_data': classification.urgency == UrgencyLevel.CRITICAL
        }

        return requirements

    def _determine_output_format(
        self,
        classification: AgentClassificationResult,
        user_input: UserInput
    ) -> str:
        """Determine appropriate output format"""
        if user_input.voice_enabled:
            return "rich_text_with_tts"
        elif classification.classification == ClassificationType.LEARNING_QUERY:
            return "structured_educational"
        elif classification.classification == ClassificationType.EMERGENCY:
            return "crisis_response"
        else:
            return "conversational"

    def _determine_post_processing(
        self,
        classification: AgentClassificationResult,
        user_input: UserInput
    ) -> List[str]:
        """Determine post-processing steps needed"""
        steps = []

        if user_input.voice_enabled:
            steps.append("generate_tts")

        if classification.classification == ClassificationType.LEARNING_QUERY:
            steps.extend(["generate_followup_questions", "suggest_related_topics"])

        if classification.classification == ClassificationType.WELLNESS_QUERY:
            steps.extend(["wellness_check", "resource_suggestions"])

        if classification.urgency in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
            steps.append("priority_logging")

        return steps

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        total_routes = sum(len(metrics) for metrics in self.performance_metrics.values())

        if total_routes == 0:
            return {"total_routes": 0, "message": "No routing data available"}

        # Calculate average confidence by classification
        avg_confidence = {}
        for classification, metrics in self.performance_metrics.items():
            if metrics:
                avg_confidence[classification] = sum(m['confidence'] for m in metrics) / len(metrics)

        return {
            "total_routes": total_routes,
            "classifications_count": {k: len(v) for k, v in self.performance_metrics.items()},
            "average_confidence": avg_confidence,
            "cache_size": len(self.routing_cache),
            "timestamp": datetime.now().isoformat()
        }

    def _combine_classification_results(
        self,
        results: List[Tuple[AgentClassificationResult, float]]
    ) -> AgentClassificationResult:
        """Combine multiple classification results using weighted scoring"""

        # Calculate weighted scores for each classification
        classification_scores = defaultdict(float)
        confidence_scores = defaultdict(list)
        urgency_scores = defaultdict(list)
        agents = defaultdict(list)
        reasonings = []

        for result, weight in results:
            classification_scores[result.classification] += result.confidence * weight
            confidence_scores[result.classification].append(result.confidence)
            urgency_scores[result.classification].append(result.urgency)
            agents[result.classification].append(result.recommended_agent)
            reasonings.append(f"{result.reasoning} (weight: {weight})")

        # Get best classification
        best_classification = max(classification_scores.keys(), key=lambda k: classification_scores[k])

        # Calculate combined confidence
        combined_confidence = min(classification_scores[best_classification], 1.0)

        # Determine urgency (take highest urgency for safety)
        urgencies = urgency_scores[best_classification]
        urgency_priority = {UrgencyLevel.LOW: 1, UrgencyLevel.NORMAL: 2, UrgencyLevel.HIGH: 3, UrgencyLevel.CRITICAL: 4}
        best_urgency = max(urgencies, key=lambda u: urgency_priority[u])

        # Select most common agent
        agent_counts = defaultdict(int)
        for agent in agents[best_classification]:
            agent_counts[agent] += 1
        best_agent = max(agent_counts.keys(), key=lambda a: agent_counts[a])

        return AgentClassificationResult(
            classification=best_classification,
            confidence=combined_confidence,
            urgency=best_urgency,
            recommended_agent=best_agent,
            reasoning=f"Combined classification: {'; '.join(reasonings)}"
        )

    def _determine_urgency(self, classification: ClassificationType, text: str) -> UrgencyLevel:
        """Determine urgency level based on classification and text content"""

        # Emergency classification always gets critical urgency
        if classification == ClassificationType.EMERGENCY:
            return UrgencyLevel.CRITICAL

        # Check for urgency indicators in text
        high_urgency_words = ["urgent", "emergency", "immediate", "asap", "quickly", "right now", "help me"]
        critical_words = ["crisis", "can't cope", "breaking down", "severe", "extreme"]

        if any(word in text for word in critical_words):
            return UrgencyLevel.CRITICAL
        elif any(word in text for word in high_urgency_words):
            return UrgencyLevel.HIGH
        elif classification == ClassificationType.WELLNESS_QUERY:
            # Wellness queries default to normal but can be elevated
            stress_words = ["stress", "anxiety", "depressed", "overwhelmed"]
            if any(word in text for word in stress_words):
                return UrgencyLevel.HIGH
            return UrgencyLevel.NORMAL
        elif classification == ClassificationType.LEARNING_QUERY:
            # Learning queries are usually normal unless urgent
            exam_words = ["exam tomorrow", "test today", "due today", "deadline"]
            if any(word in text for word in exam_words):
                return UrgencyLevel.HIGH
            return UrgencyLevel.NORMAL
        else:
            return UrgencyLevel.NORMAL

    def _select_agent(self, classification: ClassificationType) -> str:
        """Select appropriate agent based on classification"""
        agent_mapping = {
            ClassificationType.LEARNING_QUERY: "EduMentor",
            ClassificationType.WELLNESS_QUERY: "WellnessBot",
            ClassificationType.SPIRITUAL_QUERY: "GuruAgent",
            ClassificationType.EMERGENCY: "EmergencyBot",
            ClassificationType.GENERAL_QUERY: "GeneralBot"
        }
        return agent_mapping.get(classification, "GeneralBot")

    def _fallback_ai_classification(self, user_input: UserInput) -> AgentClassificationResult:
        """Fallback when AI classification fails"""
        return AgentClassificationResult(
            classification=ClassificationType.GENERAL_QUERY,
            confidence=0.4,
            urgency=UrgencyLevel.NORMAL,
            recommended_agent="GeneralBot",
            reasoning="AI classification failed, using fallback"
        )

    def _store_classification_result(
        self,
        user_id: str,
        user_input: UserInput,
        result: AgentClassificationResult
    ):
        """Store classification result for learning and pattern recognition"""

        # Store in classification history
        self.classification_history[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'query': user_input.raw_text,
            'classification': result.classification.value,
            'confidence': result.confidence,
            'agent': result.recommended_agent
        })

        # Keep only last 100 entries per user
        if len(self.classification_history[user_id]) > 100:
            self.classification_history[user_id] = self.classification_history[user_id][-100:]

        # Update user patterns
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {
                'common_classifications': defaultdict(int),
                'common_agents': defaultdict(int),
                'avg_confidence': 0.0,
                'total_queries': 0
            }

        patterns = self.user_patterns[user_id]
        patterns['common_classifications'][result.classification.value] += 1
        patterns['common_agents'][result.recommended_agent] += 1
        patterns['total_queries'] += 1

        # Update average confidence
        patterns['avg_confidence'] = (
            (patterns['avg_confidence'] * (patterns['total_queries'] - 1) + result.confidence)
            / patterns['total_queries']
        )


class BHIVAlphaEnhanced:
    """
    Enhanced BHIV Alpha Router with advanced planning capabilities
    """

    def __init__(self, gemini_manager: GeminiAPIManager):
        self.gemini_manager = gemini_manager
        self.classification_engine = AdvancedClassificationEngine(gemini_manager)
        self.agent_performance = defaultdict(lambda: {'success_rate': 0.8, 'avg_response_time': 1000})
        self.load_balancer = AgentLoadBalancer()

    async def route_and_plan(
        self,
        user_input: UserInput,
        user_info: UserInfo
    ) -> Tuple[AgentClassificationResult, Dict[str, Any]]:
        """
        Enhanced routing and planning with load balancing and performance optimization
        """

        # Step 1: Advanced classification
        classification = await self.classification_engine.classify_with_multiple_methods(
            user_input, user_info
        )

        # Step 2: Agent selection with load balancing
        optimal_agent = await self._select_optimal_agent(classification, user_info)

        # Step 3: Create execution plan
        execution_plan = await self._create_execution_plan(classification, optimal_agent, user_input)

        # Update classification with optimal agent
        classification.recommended_agent = optimal_agent

        return classification, execution_plan

    async def _select_optimal_agent(
        self,
        classification: AgentClassificationResult,
        user_info: UserInfo
    ) -> str:
        """Select optimal agent considering performance and load"""

        base_agent = classification.recommended_agent

        # Check agent performance and load
        agent_load = self.load_balancer.get_agent_load(base_agent)
        agent_performance = self.agent_performance[base_agent]

        # If agent is overloaded or performing poorly, consider alternatives
        if agent_load > 0.8 or agent_performance['success_rate'] < 0.6:
            alternative_agents = self._get_alternative_agents(classification.classification)

            for alt_agent in alternative_agents:
                alt_load = self.load_balancer.get_agent_load(alt_agent)
                alt_performance = self.agent_performance[alt_agent]

                if alt_load < 0.6 and alt_performance['success_rate'] > 0.7:
                    logger.info(f"Switching from {base_agent} to {alt_agent} due to load/performance")
                    return alt_agent

        return base_agent

    def _get_alternative_agents(self, classification: ClassificationType) -> List[str]:
        """Get alternative agents for a classification"""
        alternatives = {
            ClassificationType.LEARNING_QUERY: ["EduMentor", "GeneralBot"],
            ClassificationType.WELLNESS_QUERY: ["WellnessBot", "GeneralBot"],
            ClassificationType.SPIRITUAL_QUERY: ["GuruAgent", "GeneralBot"],
            ClassificationType.GENERAL_QUERY: ["GeneralBot"],
            ClassificationType.EMERGENCY: ["EmergencyBot", "WellnessBot"]
        }
        return alternatives.get(classification, ["GeneralBot"])

    async def _create_execution_plan(
        self,
        classification: AgentClassificationResult,
        agent: str,
        user_input: UserInput
    ) -> Dict[str, Any]:
        """Create detailed execution plan for the request"""

        plan = {
            'agent': agent,
            'classification': classification.classification.value,
            'urgency': classification.urgency.value,
            'confidence': classification.confidence,
            'estimated_processing_time': self._estimate_processing_time(classification, agent),
            'required_resources': self._determine_required_resources(classification),
            'fallback_agents': self._get_alternative_agents(classification.classification),
            'optimization_hints': self._get_optimization_hints(classification, user_input)
        }

        return plan

    def _estimate_processing_time(self, classification: AgentClassificationResult, agent: str) -> float:
        """Estimate processing time based on classification and agent"""
        base_times = {
            ClassificationType.LEARNING_QUERY: 3000,  # 3 seconds
            ClassificationType.WELLNESS_QUERY: 2500,  # 2.5 seconds
            ClassificationType.SPIRITUAL_QUERY: 4000,  # 4 seconds
            ClassificationType.GENERAL_QUERY: 1500,   # 1.5 seconds
            ClassificationType.EMERGENCY: 1000        # 1 second (priority)
        }

        base_time = base_times.get(classification.classification, 2000)
        agent_modifier = self.agent_performance[agent]['avg_response_time'] / 1000

        return base_time * agent_modifier

    def _determine_required_resources(self, classification: AgentClassificationResult) -> List[str]:
        """Determine required resources for processing"""
        resources = ['knowledgebase']

        if classification.classification == ClassificationType.LEARNING_QUERY:
            resources.extend(['educational_db', 'quiz_engine'])
        elif classification.classification == ClassificationType.WELLNESS_QUERY:
            resources.extend(['wellness_db', 'emotional_support'])
        elif classification.classification == ClassificationType.SPIRITUAL_QUERY:
            resources.extend(['vedas_db', 'spiritual_guidance'])
        elif classification.classification == ClassificationType.EMERGENCY:
            resources.extend(['crisis_protocols', 'emergency_contacts'])

        return resources

    def _get_optimization_hints(
        self,
        classification: AgentClassificationResult,
        user_input: UserInput
    ) -> List[str]:
        """Get optimization hints for processing"""
        hints = []

        if classification.confidence > 0.9:
            hints.append('high_confidence_fast_track')

        if classification.urgency == UrgencyLevel.CRITICAL:
            hints.append('priority_processing')

        if user_input.voice_enabled:
            hints.append('prepare_tts_early')

        if len(user_input.raw_text) > 500:
            hints.append('long_query_optimization')

        return hints


class AgentLoadBalancer:
    """Simple agent load balancer"""

    def __init__(self):
        self.agent_loads = defaultdict(float)
        self.last_update = datetime.now()

    def get_agent_load(self, agent: str) -> float:
        """Get current load for an agent (0.0 to 1.0)"""
        # Simulate load decay over time
        time_diff = (datetime.now() - self.last_update).total_seconds()
        decay_factor = max(0.0, 1.0 - (time_diff / 60.0))  # Decay over 1 minute

        return self.agent_loads[agent] * decay_factor

    def update_agent_load(self, agent: str, load_delta: float):
        """Update agent load"""
        current_load = self.get_agent_load(agent)
        self.agent_loads[agent] = min(1.0, max(0.0, current_load + load_delta))
        self.last_update = datetime.now()
