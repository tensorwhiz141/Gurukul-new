"""
Nudge Engine Module for Karthikeya Multilingual Reporting Engine

This module provides advanced nudge generation logic for both Edumentor and Wellness Bot
with intelligent intervention suggestions based on risk thresholds and user context.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .sentiment_analyzer import SentimentAnalyzer, SentimentResult


class NudgeType(Enum):
    """Types of nudges that can be generated"""
    EDUCATIONAL = "educational"
    MOTIVATIONAL = "motivational"
    INTERVENTION = "intervention"
    REMINDER = "reminder"
    CELEBRATION = "celebration"


class UrgencyLevel(Enum):
    """Urgency levels for nudges"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NudgeContext:
    """Context information for nudge generation"""
    user_id: str
    context: str  # "edumentor" or "wellness"
    user_data: Dict[str, Any]
    historical_data: Dict[str, Any]
    preferences: Dict[str, Any]
    language: str = "en"


@dataclass
class GeneratedNudge:
    """Structure for a generated nudge"""
    nudge_id: str
    nudge_type: NudgeType
    urgency: UrgencyLevel
    title: str
    message: str
    action_text: str
    action_type: str
    metadata: Dict[str, Any]
    expires_at: str
    created_at: str


class NudgeEngine:
    """
    Advanced nudge generation engine with intelligent intervention logic
    """
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        """
        Initialize the nudge engine
        
        Args:
            sentiment_analyzer: Instance of SentimentAnalyzer
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.nudge_history = {}  # In production, this would be a database
        
    def generate_nudges(self, context: NudgeContext, sentiment_result: SentimentResult) -> List[GeneratedNudge]:
        """
        Generate contextual nudges based on user data and sentiment
        
        Args:
            context: Nudge context with user data
            sentiment_result: Result from sentiment analysis
            
        Returns:
            List of generated nudges
        """
        nudges = []
        
        if context.context == "edumentor":
            nudges.extend(self._generate_edumentor_nudges(context, sentiment_result))
        elif context.context == "wellness":
            nudges.extend(self._generate_wellness_nudges(context, sentiment_result))
        
        # Filter out recently sent nudges to avoid spam
        nudges = self._filter_recent_nudges(nudges, context.user_id)
        
        # Sort by urgency and relevance
        nudges.sort(key=lambda x: (x.urgency.value, x.nudge_type.value))
        
        return nudges
    
    def _generate_edumentor_nudges(self, context: NudgeContext, sentiment_result: SentimentResult) -> List[GeneratedNudge]:
        """Generate nudges for Edumentor context"""
        nudges = []
        data = context.user_data
        
        # Performance-based nudges
        if data.get("average_score", 0) < 40:
            nudges.append(self._create_intervention_nudge(
                context, "poor_performance", UrgencyLevel.HIGH,
                "Academic Support Needed",
                "Your recent performance suggests you might benefit from additional support. Let's get you back on track!",
                "Schedule Tutoring Session",
                "schedule_tutoring"
            ))
        
        # Engagement-based nudges
        engagement_score = data.get("engagement_score", 100)
        if engagement_score < 30:
            nudges.append(self._create_motivational_nudge(
                context, "low_engagement", UrgencyLevel.HIGH,
                "We Miss You!",
                "Your learning journey is important. Come back and continue your progress!",
                "Resume Learning",
                "resume_learning"
            ))
        
        # Streak maintenance nudges
        streak_days = data.get("streak_days", 0)
        days_since_activity = data.get("days_since_activity", 0)
        if streak_days > 7 and days_since_activity >= 1:
            nudges.append(self._create_reminder_nudge(
                context, "streak_maintenance", UrgencyLevel.MEDIUM,
                f"Maintain Your {streak_days}-Day Streak!",
                "You're doing great! Don't break your learning streak now.",
                "Continue Today's Lesson",
                "continue_lesson"
            ))
        
        # Quiz completion nudges
        missed_quizzes = data.get("missed_quizzes", 0)
        if missed_quizzes >= 2:
            nudges.append(self._create_educational_nudge(
                context, "quiz_completion", UrgencyLevel.MEDIUM,
                "Quiz Time!",
                f"You have {missed_quizzes} pending quizzes. Regular assessment helps track your progress.",
                "Take Quiz Now",
                "take_quiz"
            ))
        
        # Celebration nudges for achievements
        if data.get("recent_achievement"):
            nudges.append(self._create_celebration_nudge(
                context, "achievement", UrgencyLevel.LOW,
                "Congratulations!",
                f"Amazing work on {data.get('recent_achievement')}! Keep up the excellent progress.",
                "View Achievement",
                "view_achievement"
            ))
        
        return nudges
    
    def _generate_wellness_nudges(self, context: NudgeContext, sentiment_result: SentimentResult) -> List[GeneratedNudge]:
        """Generate nudges for Wellness context"""
        nudges = []
        data = context.user_data
        
        # Financial wellness nudges
        if "financial" in data:
            financial_data = data["financial"]
            
            # Overspending alerts
            spending_ratio = financial_data.get("spending_ratio", 1.0)
            if spending_ratio > 1.2:
                nudges.append(self._create_intervention_nudge(
                    context, "overspending", UrgencyLevel.HIGH,
                    "Budget Alert!",
                    f"You've exceeded your budget by {(spending_ratio - 1) * 100:.0f}%. Time to review your expenses.",
                    "Review Budget",
                    "review_budget"
                ))
            
            # Savings encouragement
            savings_rate = financial_data.get("savings_rate", 0)
            if savings_rate < 5:
                nudges.append(self._create_educational_nudge(
                    context, "low_savings", UrgencyLevel.MEDIUM,
                    "Boost Your Savings",
                    "Even small amounts saved regularly can make a big difference in your financial future.",
                    "Set Savings Goal",
                    "set_savings_goal"
                ))
            
            # Bill payment reminders
            pending_bills = financial_data.get("pending_bills", 0)
            if pending_bills > 0:
                nudges.append(self._create_reminder_nudge(
                    context, "bill_payment", UrgencyLevel.HIGH,
                    "Bills Due Soon",
                    f"You have {pending_bills} bills due in the next 3 days. Don't forget to pay them on time!",
                    "Pay Bills",
                    "pay_bills"
                ))
        
        # Emotional wellness nudges
        if "emotional" in data:
            emotional_data = data["emotional"]
            
            # Stress management
            stress_level = emotional_data.get("stress_level", 50)
            if stress_level > 75:
                nudges.append(self._create_intervention_nudge(
                    context, "stress_management", UrgencyLevel.HIGH,
                    "Take a Breather",
                    "Your stress levels seem elevated. Take a 10-minute break and try some deep breathing exercises.",
                    "Start Breathing Exercise",
                    "breathing_exercise"
                ))
            
            # Activity encouragement
            activity_score = emotional_data.get("activity_score", 50)
            if activity_score < 20:
                nudges.append(self._create_motivational_nudge(
                    context, "increase_activity", UrgencyLevel.MEDIUM,
                    "Get Moving!",
                    "A short walk or light exercise can significantly boost your mood and energy levels.",
                    "Start 10-Min Walk",
                    "start_walk"
                ))
            
            # Social connection
            days_since_social = emotional_data.get("days_since_social_interaction", 0)
            if days_since_social > 7:
                nudges.append(self._create_motivational_nudge(
                    context, "social_connection", UrgencyLevel.MEDIUM,
                    "Connect with Others",
                    "Reaching out to friends or family can improve your emotional well-being.",
                    "Call a Friend",
                    "call_friend"
                ))
        
        return nudges
    
    def _create_intervention_nudge(self, context: NudgeContext, nudge_subtype: str, 
                                 urgency: UrgencyLevel, title: str, message: str, 
                                 action_text: str, action_type: str) -> GeneratedNudge:
        """Create an intervention nudge"""
        return self._create_nudge(context, NudgeType.INTERVENTION, nudge_subtype, urgency, 
                                title, message, action_text, action_type)
    
    def _create_motivational_nudge(self, context: NudgeContext, nudge_subtype: str,
                                  urgency: UrgencyLevel, title: str, message: str,
                                  action_text: str, action_type: str) -> GeneratedNudge:
        """Create a motivational nudge"""
        return self._create_nudge(context, NudgeType.MOTIVATIONAL, nudge_subtype, urgency,
                                title, message, action_text, action_type)
    
    def _create_educational_nudge(self, context: NudgeContext, nudge_subtype: str,
                                 urgency: UrgencyLevel, title: str, message: str,
                                 action_text: str, action_type: str) -> GeneratedNudge:
        """Create an educational nudge"""
        return self._create_nudge(context, NudgeType.EDUCATIONAL, nudge_subtype, urgency,
                                title, message, action_text, action_type)
    
    def _create_reminder_nudge(self, context: NudgeContext, nudge_subtype: str,
                              urgency: UrgencyLevel, title: str, message: str,
                              action_text: str, action_type: str) -> GeneratedNudge:
        """Create a reminder nudge"""
        return self._create_nudge(context, NudgeType.REMINDER, nudge_subtype, urgency,
                                title, message, action_text, action_type)
    
    def _create_celebration_nudge(self, context: NudgeContext, nudge_subtype: str,
                                 urgency: UrgencyLevel, title: str, message: str,
                                 action_text: str, action_type: str) -> GeneratedNudge:
        """Create a celebration nudge"""
        return self._create_nudge(context, NudgeType.CELEBRATION, nudge_subtype, urgency,
                                title, message, action_text, action_type)
    
    def _create_nudge(self, context: NudgeContext, nudge_type: NudgeType, nudge_subtype: str,
                     urgency: UrgencyLevel, title: str, message: str,
                     action_text: str, action_type: str) -> GeneratedNudge:
        """Create a nudge with common metadata"""
        nudge_id = f"{context.user_id}_{nudge_type.value}_{nudge_subtype}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Apply tone formatting based on urgency and type
        tone_formatting = self._get_tone_for_nudge(nudge_type, urgency, context.language)
        formatted_message = f"{tone_formatting.get('prefix', '')}{message}"
        
        return GeneratedNudge(
            nudge_id=nudge_id,
            nudge_type=nudge_type,
            urgency=urgency,
            title=title,
            message=formatted_message,
            action_text=action_text,
            action_type=action_type,
            metadata={
                "context": context.context,
                "subtype": nudge_subtype,
                "language": context.language,
                "tone": tone_formatting.get('style', 'neutral'),
                "emoji": tone_formatting.get('emoji', '')
            },
            expires_at=(datetime.now() + timedelta(hours=24)).isoformat(),
            created_at=datetime.now().isoformat()
        )
    
    def _get_tone_for_nudge(self, nudge_type: NudgeType, urgency: UrgencyLevel, language: str) -> Dict[str, str]:
        """Get appropriate tone formatting for nudge type and urgency"""
        # Map nudge type and urgency to tone enum
        from .sentiment_analyzer import ToneType

        if nudge_type == NudgeType.CELEBRATION:
            tone_enum = ToneType.CONGRATULATORY
        elif urgency == UrgencyLevel.HIGH or urgency == UrgencyLevel.CRITICAL:
            tone_enum = ToneType.ALERT
        elif nudge_type == NudgeType.INTERVENTION:
            tone_enum = ToneType.SUPPORTIVE
        elif nudge_type == NudgeType.MOTIVATIONAL:
            tone_enum = ToneType.ENCOURAGING
        else:
            tone_enum = ToneType.GENTLE

        # Get formatting from sentiment analyzer
        return self.sentiment_analyzer.get_tone_formatting(tone_enum, language)
    
    def _filter_recent_nudges(self, nudges: List[GeneratedNudge], user_id: str) -> List[GeneratedNudge]:
        """Filter out nudges that were recently sent to avoid spam"""
        # In production, this would check against a database
        # For now, we'll implement a simple in-memory filter
        
        if user_id not in self.nudge_history:
            self.nudge_history[user_id] = []
        
        recent_nudges = self.nudge_history[user_id]
        cutoff_time = datetime.now() - timedelta(hours=6)  # Don't repeat nudges within 6 hours
        
        filtered_nudges = []
        for nudge in nudges:
            # Check if similar nudge was sent recently
            is_recent = any(
                recent_nudge.get('action_type') == nudge.action_type and
                datetime.fromisoformat(recent_nudge.get('created_at', '1970-01-01')) > cutoff_time
                for recent_nudge in recent_nudges
            )
            
            if not is_recent:
                filtered_nudges.append(nudge)
                # Add to history
                recent_nudges.append({
                    'action_type': nudge.action_type,
                    'created_at': nudge.created_at
                })
        
        # Keep only recent history (last 50 nudges)
        self.nudge_history[user_id] = recent_nudges[-50:]
        
        return filtered_nudges
