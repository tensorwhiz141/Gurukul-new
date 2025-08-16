"""
Report Generator Module for Karthikeya Multilingual Reporting Engine

This module accepts forecast/score JSON from Vedant's engine and produces
multilingual-ready JSON output with proper text and TTS placeholders.
"""

import json
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict

from .sentiment_analyzer import SentimentAnalyzer, SentimentResult
from .template_loader import TemplateLoader


@dataclass
class ReportData:
    """Structure for report input data"""
    user_id: str
    report_type: str  # "progress_report", "quiz_performance", "financial_health", "emotional_health"
    context: str  # "edumentor" or "wellness"
    language: str = "en"
    data: Dict[str, Any] = None
    timestamp: str = None
    override_thresholds: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.data is None:
            self.data = {}


@dataclass
class GeneratedReport:
    """Structure for generated report output"""
    report_id: str
    user_id: str
    report_type: str
    language: str
    title: str
    content: Dict[str, Any]
    sentiment: Dict[str, Any]
    nudges: List[Dict[str, Any]]
    tts_ready: bool
    metadata: Dict[str, Any]
    timestamp: str


class ReportGenerator:
    """
    Core report generation engine that transforms forecast/score data
    into human-friendly multilingual advice
    """

    def __init__(self, sentiment_analyzer: SentimentAnalyzer = None, template_loader: TemplateLoader = None):
        """
        Initialize the report generator

        Args:
            sentiment_analyzer: Optional sentiment analyzer instance
            template_loader: Optional template loader instance
        """
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.template_loader = template_loader or TemplateLoader()
        # For backward compatibility, keep templates attribute
        self.templates = {}

    def _get_template(self, context: str, language: str) -> Dict[str, Any]:
        """Get template for specific context and language using template loader"""
        return self.template_loader.get_template(context, language)

    def generate_report(self, report_data: ReportData) -> GeneratedReport:
        """
        Generate a complete multilingual report

        Args:
            report_data: Input data for report generation

        Returns:
            GeneratedReport object with complete report
        """
        # Validate input
        self._validate_report_data(report_data)

        # Get appropriate template
        template = self._get_template_for_report(report_data.context, report_data.report_type, report_data.language)

        # Analyze sentiment
        sentiment_result = self._analyze_sentiment(report_data)

        # Generate content
        content = self._generate_content(template, report_data, sentiment_result)

        # Generate nudges
        nudges = self._generate_nudges(report_data, sentiment_result)

        # Create report ID
        report_id = f"{report_data.user_id}_{report_data.report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return GeneratedReport(
            report_id=report_id,
            user_id=report_data.user_id,
            report_type=report_data.report_type,
            language=report_data.language,
            title=template.get("title", "Report"),
            content=content,
            sentiment=asdict(sentiment_result),
            nudges=nudges,
            tts_ready=True,
            metadata={
                "context": report_data.context,
                "generation_time": datetime.now().isoformat(),
                "template_version": "1.0",
                "sentiment_confidence": sentiment_result.confidence
            },
            timestamp=report_data.timestamp
        )

    def _validate_report_data(self, report_data: ReportData) -> None:
        """Validate input report data"""
        if not report_data.user_id:
            raise ValueError("User ID is required")

        if report_data.context not in ["edumentor", "wellness"]:
            raise ValueError("Context must be 'edumentor' or 'wellness'")

        # Get supported languages from sentiment analyzer
        supported_languages = self.sentiment_analyzer.supported_languages
        if report_data.language not in supported_languages:
            raise ValueError(f"Language must be one of: {', '.join(supported_languages)}")

        valid_types = {
            "edumentor": ["progress_report", "quiz_performance", "lesson_content"],
            "wellness": ["financial_health", "emotional_health"]
        }

        if report_data.report_type not in valid_types[report_data.context]:
            raise ValueError(f"Invalid report type '{report_data.report_type}' for context '{report_data.context}'")

    def _get_template_for_report(self, context: str, report_type: str, language: str) -> Dict[str, Any]:
        """Get the appropriate template for the report"""
        try:
            # Get templates for the context and language
            context_templates = self.template_loader.get_template(context, language)
            if report_type in context_templates:
                return context_templates[report_type]
            else:
                # Fallback to progress_report if specific type not found
                return context_templates.get("progress_report", {})
        except Exception:
            # Fallback to English
            try:
                context_templates = self.template_loader.get_template(context, "en")
                if report_type in context_templates:
                    return context_templates[report_type]
                else:
                    return context_templates.get("progress_report", {})
            except Exception:
                # Return a basic template if all else fails
                return {
                    "title": f"{context.title()} Report",
                    "summary": "Report generated successfully.",
                    "risk_assessment": {"low": "Good progress", "medium": "Steady progress", "high": "Needs attention"},
                    "recommendations": {"low": "Keep it up", "medium": "Stay focused", "high": "Seek help"},
                    "nudge_triggers": {}
                }

    def _analyze_sentiment(self, report_data: ReportData) -> SentimentResult:
        """Analyze sentiment based on report data"""
        data = report_data.data

        # Different sentiment analysis based on report type
        if report_data.report_type in ["progress_report", "quiz_performance", "lesson_content"]:
            # Use score-based sentiment for educational reports
            score = data.get("average_score", data.get("current_score", 85))  # Default to good score for lessons
            return self.sentiment_analyzer.analyze_score_sentiment(
                score, report_data.language, report_data.override_thresholds
            )

        elif report_data.report_type == "financial_health":
            # Use risk-based sentiment for financial reports
            risk_level = data.get("risk_level", "medium")
            return self.sentiment_analyzer.analyze_risk_sentiment(risk_level)

        elif report_data.report_type == "emotional_health":
            # Use engagement-based sentiment for emotional health
            wellness_score = data.get("wellness_score", data.get("stress_level", 50))
            # Invert stress level to wellness score if needed
            if "stress_level" in data:
                wellness_score = 100 - wellness_score
            return self.sentiment_analyzer.analyze_engagement_sentiment(
                wellness_score, report_data.language
            )

        else:
            # Default sentiment analysis
            return self.sentiment_analyzer.analyze_score_sentiment(
                50, report_data.language, report_data.override_thresholds
            )

    def _generate_content(self, template: Dict[str, Any],
                         report_data: ReportData,
                         sentiment_result: SentimentResult) -> Dict[str, Any]:
        """Generate report content by filling template placeholders"""
        content = {}
        data = report_data.data

        # Fill in basic template fields
        for key, value in template.items():
            if isinstance(value, str):
                content[key] = self._fill_placeholders(value, data, sentiment_result, report_data.language)
            elif isinstance(value, dict):
                content[key] = self._process_template_section(value, data, sentiment_result, report_data.language)
            else:
                content[key] = value

        return content

    def _process_template_section(self, section: Dict[str, Any],
                                 data: Dict[str, Any],
                                 sentiment_result: SentimentResult,
                                 language: str) -> Dict[str, Any]:
        """Process a template section (like risk_assessment or recommendations)"""
        processed_section = {}

        for key, value in section.items():
            if isinstance(value, str):
                processed_section[key] = self._fill_placeholders(value, data, sentiment_result, language)
            elif isinstance(value, dict):
                processed_section[key] = self._process_template_section(value, data, sentiment_result, language)
            else:
                processed_section[key] = value

        return processed_section

    def _fill_placeholders(self, template_text: str,
                          data: Dict[str, Any],
                          sentiment_result: SentimentResult,
                          language: str) -> str:
        """Fill placeholders in template text with actual data"""
        text = template_text

        # Replace sentiment-related placeholders
        text = text.replace("{sentiment_description}", sentiment_result.description)

        # Replace data placeholders
        placeholders = re.findall(r'\{([^}]+)\}', text)
        for placeholder in placeholders:
            value = self._get_placeholder_value(placeholder, data, sentiment_result, language)
            text = text.replace(f"{{{placeholder}}}", str(value))

        return text

    def _get_placeholder_value(self, placeholder: str,
                              data: Dict[str, Any],
                              sentiment_result: SentimentResult,
                              language: str) -> str:
        """Get the value for a specific placeholder"""
        # Direct data lookup
        if placeholder in data:
            return str(data[placeholder])

        # Computed values based on placeholder name
        if placeholder == "comparison":
            current = data.get("current_score", 0)
            average = data.get("average_score", 0)
            if current > average:
                return {"en": "above", "hi": "से ऊपर", "bn": "উপরে"}.get(language, "above")
            elif current < average:
                return {"en": "below", "hi": "से नीचे", "bn": "নিচে"}.get(language, "below")
            else:
                return {"en": "equal to", "hi": "के बराबर", "bn": "সমান"}.get(language, "equal to")

        elif placeholder == "spending_comparison":
            spending_ratio = data.get("spending_ratio", 1.0)
            if spending_ratio > 1.1:
                return {"en": "significantly above", "hi": "काफी ऊपर", "bn": "উল্লেখযোগ্যভাবে উপরে"}.get(language, "significantly above")
            elif spending_ratio > 1.0:
                return {"en": "slightly above", "hi": "थोड़ा ऊपर", "bn": "সামান্য উপরে"}.get(language, "slightly above")
            elif spending_ratio < 0.9:
                return {"en": "below", "hi": "नीचे", "bn": "নিচে"}.get(language, "below")
            else:
                return {"en": "within", "hi": "के भीतर", "bn": "মধ্যে"}.get(language, "within")

        # Default values for common placeholders
        defaults = {
            "completed_modules": data.get("completed_modules", 0),
            "total_modules": data.get("total_modules", 10),
            "average_score": data.get("average_score", 0),
            "current_score": data.get("current_score", 0),
            "savings_rate": data.get("savings_rate", 0),
            "missed_count": data.get("missed_quizzes", 0),
            "percentage": data.get("engagement_drop", 0),
            "streak_days": data.get("streak_days", 0),
            "subject_area": data.get("subject_area", "your studies"),
            "weak_areas": data.get("weak_areas", "identified areas"),
            "strong_subjects": data.get("strong_subjects", "your strengths"),
            "weak_subjects": data.get("weak_subjects", "areas for improvement"),
            "accuracy_threshold": data.get("accuracy_threshold", 80),
            "high_spending_categories": data.get("high_spending_categories", "discretionary spending"),
            "category": data.get("overspending_category", "budget"),
            "amount": data.get("amount", "₹0"),
            "rate": data.get("current_savings_rate", 0),
            "count": data.get("pending_bills", 0)
        }

        return str(defaults.get(placeholder, f"[{placeholder}]"))

    def _generate_nudges(self, report_data: ReportData, sentiment_result: SentimentResult) -> List[Dict[str, Any]]:
        """Generate contextual nudges based on data and sentiment"""
        nudges = []
        data = report_data.data

        # Get tone formatting
        tone_formatting = self.sentiment_analyzer.get_tone_formatting(sentiment_result.tone, report_data.language)

        # Check for nudge triggers based on context
        if report_data.context == "edumentor":
            nudges.extend(self._generate_edumentor_nudges(report_data, sentiment_result, tone_formatting))
        elif report_data.context == "wellness":
            nudges.extend(self._generate_wellness_nudges(report_data, sentiment_result, tone_formatting))

        return nudges

    def _generate_edumentor_nudges(self, report_data: ReportData,
                                  sentiment_result: SentimentResult,
                                  tone_formatting: Dict[str, str]) -> List[Dict[str, Any]]:
        """Generate nudges for Edumentor context"""
        nudges = []
        data = report_data.data

        # Check for missed quizzes
        missed_quizzes = data.get("missed_quizzes", 0)
        should_nudge, urgency = self.sentiment_analyzer.should_trigger_nudge("edumentor", "missed_quizzes", missed_quizzes)

        if should_nudge:
            template = self._get_template(report_data.context, report_data.language)
            nudge_text = template.get("nudge_triggers", {}).get("missed_quizzes", "")
            if nudge_text:
                nudge_text = self._fill_placeholders(nudge_text, data, sentiment_result, report_data.language)
                nudges.append({
                    "type": "missed_quizzes",
                    "urgency": urgency,
                    "message": f"{tone_formatting.get('prefix', '')}{nudge_text}",
                    "action": "take_quiz",
                    "emoji": tone_formatting.get('emoji', ''),
                    "style": tone_formatting.get('style', 'neutral')
                })

        # Check for low engagement
        engagement_score = data.get("engagement_score", 100)
        should_nudge, urgency = self.sentiment_analyzer.should_trigger_nudge("edumentor", "low_engagement", engagement_score)

        if should_nudge:
            template = self._get_template(report_data.context, report_data.language)
            nudge_text = template.get("nudge_triggers", {}).get("low_engagement", "")
            if nudge_text:
                nudge_text = self._fill_placeholders(nudge_text, data, sentiment_result, report_data.language)
                nudges.append({
                    "type": "low_engagement",
                    "urgency": urgency,
                    "message": f"{tone_formatting.get('prefix', '')}{nudge_text}",
                    "action": "increase_engagement",
                    "emoji": tone_formatting.get('emoji', ''),
                    "style": tone_formatting.get('style', 'neutral')
                })

        # Check for streak break risk
        days_since_activity = data.get("days_since_activity", 0)
        should_nudge, urgency = self.sentiment_analyzer.should_trigger_nudge("edumentor", "streak_break", days_since_activity)

        if should_nudge:
            template = self._get_template(report_data.context, report_data.language)
            nudge_text = template.get("nudge_triggers", {}).get("streak_break", "")
            if nudge_text:
                nudge_text = self._fill_placeholders(nudge_text, data, sentiment_result, report_data.language)
                nudges.append({
                    "type": "streak_break",
                    "urgency": urgency,
                    "message": f"{tone_formatting.get('prefix', '')}{nudge_text}",
                    "action": "continue_streak",
                    "emoji": tone_formatting.get('emoji', ''),
                    "style": tone_formatting.get('style', 'neutral')
                })

        return nudges

    def _generate_wellness_nudges(self, report_data: ReportData,
                                 sentiment_result: SentimentResult,
                                 tone_formatting: Dict[str, str]) -> List[Dict[str, Any]]:
        """Generate nudges for Wellness context"""
        nudges = []
        data = report_data.data

        if report_data.report_type == "financial_health":
            # Check for overspending
            spending_ratio = data.get("spending_ratio", 1.0) * 100
            should_nudge, urgency = self.sentiment_analyzer.should_trigger_nudge("wellness", "financial.overspending", spending_ratio)

            if should_nudge:
                template = self._get_template(report_data.context, report_data.language)
                nudge_text = template.get("nudge_triggers", {}).get("overspending", "")
                if nudge_text:
                    nudge_text = self._fill_placeholders(nudge_text, data, sentiment_result, report_data.language)
                    nudges.append({
                        "type": "overspending",
                        "urgency": urgency,
                        "message": f"{tone_formatting.get('prefix', '')}{nudge_text}",
                        "action": "review_budget",
                        "emoji": tone_formatting.get('emoji', ''),
                        "style": tone_formatting.get('style', 'neutral')
                    })

            # Check for low savings
            savings_rate = data.get("savings_rate", 10)
            should_nudge, urgency = self.sentiment_analyzer.should_trigger_nudge("wellness", "financial.low_savings", savings_rate)

            if should_nudge:
                template = self._get_template(report_data.context, report_data.language)
                nudge_text = template.get("nudge_triggers", {}).get("low_savings", "")
                if nudge_text:
                    nudge_text = self._fill_placeholders(nudge_text, data, sentiment_result, report_data.language)
                    nudges.append({
                        "type": "low_savings",
                        "urgency": urgency,
                        "message": f"{tone_formatting.get('prefix', '')}{nudge_text}",
                        "action": "increase_savings",
                        "emoji": tone_formatting.get('emoji', ''),
                        "style": tone_formatting.get('style', 'neutral')
                    })

        elif report_data.report_type == "emotional_health":
            # Check for stress spike
            stress_level = data.get("stress_level", 50)
            should_nudge, urgency = self.sentiment_analyzer.should_trigger_nudge("wellness", "emotional.stress_spike", stress_level)

            if should_nudge:
                template = self._get_template(report_data.context, report_data.language)
                nudge_text = template.get("nudge_triggers", {}).get("stress_spike", "")
                if nudge_text:
                    nudge_text = self._fill_placeholders(nudge_text, data, sentiment_result, report_data.language)
                    nudges.append({
                        "type": "stress_spike",
                        "urgency": urgency,
                        "message": f"{tone_formatting.get('prefix', '')}{nudge_text}",
                        "action": "stress_relief",
                        "emoji": tone_formatting.get('emoji', ''),
                        "style": tone_formatting.get('style', 'neutral')
                    })

        return nudges