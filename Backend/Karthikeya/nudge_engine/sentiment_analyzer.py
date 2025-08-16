"""
Sentiment Analysis Module for Karthikeya Multilingual Reporting Engine

This module provides rule-based sentiment detection that maps forecast/score inputs
to sentiment categories and appropriate nudge tones for multilingual reports.
"""

import json
import os
import yaml
import logging
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class SentimentType(Enum):
    """Enumeration of sentiment types"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    CONCERNED = "concerned"


class ToneType(Enum):
    """Enumeration of tone types for nudges"""
    CONGRATULATORY = "congratulatory"
    ENCOURAGING = "encouraging"
    GENTLE = "gentle"
    SUPPORTIVE = "supportive"
    ALERT = "alert"


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    sentiment: SentimentType
    tone: ToneType
    description: str
    urgency: str
    confidence: float


class SentimentAnalyzer:
    """
    Rule-based sentiment analyzer for educational and wellness data
    """

    def __init__(self,
                 config_path: str = "templates/sentiment_mappings.json",
                 language_config_path: str = "config/language_config.yaml",
                 nudge_config_path: str = "config/nudge_config.yaml"):
        """
        Initialize the sentiment analyzer with configuration

        Args:
            config_path: Path to sentiment mappings configuration file
            language_config_path: Path to language configuration file
            nudge_config_path: Path to nudge configuration file
        """
        self.config_path = config_path
        self.language_config_path = language_config_path
        self.nudge_config_path = nudge_config_path
        self.config = self._load_config()
        self.language_config = self._load_language_config()
        self.nudge_config = self._load_nudge_config()
        self.supported_languages = self._get_supported_languages()
        self.fallback_language = self.language_config.get('fallback_language', 'en')

    def _load_config(self) -> Dict[str, Any]:
        """Load sentiment configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Sentiment config file not found: {self.config_path}. Using defaults.")
            return self._get_default_sentiment_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in sentiment config: {e}. Using defaults.")
            return self._get_default_sentiment_config()

    def _load_language_config(self) -> Dict[str, Any]:
        """Load language configuration from YAML file"""
        try:
            with open(self.language_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Language config file not found: {self.language_config_path}. Using defaults.")
            return self._get_default_language_config()
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in language config: {e}")
            return self._get_default_language_config()

    def _load_nudge_config(self) -> Dict[str, Any]:
        """Load nudge configuration from YAML file"""
        try:
            with open(self.nudge_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Nudge config file not found: {self.nudge_config_path}. Using defaults.")
            return self._get_default_nudge_config()
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in nudge config: {e}")
            return self._get_default_nudge_config()

    def _get_default_language_config(self) -> Dict[str, Any]:
        """Get default language configuration"""
        return {
            'languages_supported': [
                {'code': 'en', 'name': 'English', 'enabled': True},
                {'code': 'hi', 'name': 'Hindi', 'enabled': True},
                {'code': 'bn', 'name': 'Bengali', 'enabled': True},
                {'code': 'gu', 'name': 'Gujarati', 'enabled': True},
                {'code': 'mr', 'name': 'Marathi', 'enabled': True},
                {'code': 'ta', 'name': 'Tamil', 'enabled': True},
                {'code': 'te', 'name': 'Telugu', 'enabled': True},
                {'code': 'kn', 'name': 'Kannada', 'enabled': True}
            ],
            'fallback_language': 'en',
            'logging': {'log_fallbacks': True}
        }

    def _get_default_nudge_config(self) -> Dict[str, Any]:
        """Get default nudge configuration"""
        return {
            'risk_thresholds': {
                'edumentor': {'overall_risk': 0.65},
                'wellness_bot': {'overall_risk': 0.75}
            },
            'tone_mapping': {
                'score_ranges': {
                    'excellent': {'min_score': 85, 'tone': 'congratulatory', 'urgency': 'low'},
                    'good': {'min_score': 70, 'tone': 'encouraging', 'urgency': 'low'},
                    'average': {'min_score': 50, 'tone': 'gentle', 'urgency': 'medium'},
                    'below_average': {'min_score': 30, 'tone': 'supportive', 'urgency': 'high'},
                    'poor': {'min_score': 0, 'tone': 'alert', 'urgency': 'high'}
                }
            },
            'nudge_behavior': {
                'frequency': {'max_nudges_per_day': 5}
            }
        }

    def _get_default_sentiment_config(self) -> Dict[str, Any]:
        """Get default sentiment configuration"""
        return {
            "sentiment_rules": {
                "score_based": {
                    "excellent": {
                        "threshold": 85,
                        "sentiment": "positive",
                        "tone": "congratulatory",
                        "description": {"en": "excellent progress"}
                    },
                    "good": {
                        "threshold": 70,
                        "sentiment": "positive",
                        "tone": "encouraging",
                        "description": {"en": "good progress"}
                    },
                    "average": {
                        "threshold": 50,
                        "sentiment": "neutral",
                        "tone": "gentle",
                        "description": {"en": "steady progress"}
                    },
                    "below_average": {
                        "threshold": 30,
                        "sentiment": "concerned",
                        "tone": "supportive",
                        "description": {"en": "concerning trends"}
                    },
                    "poor": {
                        "threshold": 0,
                        "sentiment": "concerned",
                        "tone": "alert",
                        "description": {"en": "significant challenges"}
                    }
                }
            },
            "tone_mappings": {
                "congratulatory": {
                    "en": {"prefix": "🎉 Fantastic! ", "style": "enthusiastic", "emoji": "🌟"}
                },
                "encouraging": {
                    "en": {"prefix": "👍 Great work! ", "style": "positive", "emoji": "💪"}
                },
                "gentle": {
                    "en": {"prefix": "💡 ", "style": "neutral", "emoji": "🤝"}
                },
                "supportive": {
                    "en": {"prefix": "🤗 We're here to help. ", "style": "caring", "emoji": "💙"}
                },
                "alert": {
                    "en": {"prefix": "⚠️ Important: ", "style": "urgent", "emoji": "🚨"}
                }
            }
        }

    def _get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        languages = self.language_config.get('languages_supported', [])
        return [lang['code'] for lang in languages if lang.get('enabled', True)]

    def _validate_and_fallback_language(self, language: str) -> str:
        """Validate language and apply fallback if needed"""
        if language in self.supported_languages:
            return language

        # Log fallback if configured
        if self.language_config.get('logging', {}).get('log_fallbacks', True):
            logger.warning(f"Unsupported language '{language}' requested. Falling back to '{self.fallback_language}'")

        return self.fallback_language

    def _convert_nudge_config_to_score_rules(self, nudge_score_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Convert nudge config score ranges to sentiment analyzer format"""
        score_rules = {}

        for category, config in nudge_score_ranges.items():
            score_rules[category] = {
                "threshold": config.get("min_score", 0),
                "sentiment": self._map_tone_to_sentiment(config.get("tone", "neutral")),
                "tone": config.get("tone", "neutral"),
                "description": {
                    "en": f"{category.replace('_', ' ')} progress"
                }
            }

        # Ensure we have a 'poor' category for fallback
        if "poor" not in score_rules:
            score_rules["poor"] = {
                "threshold": 0,
                "sentiment": "concerned",
                "tone": "alert",
                "description": {
                    "en": "significant challenges"
                }
            }

        return score_rules

    def _map_tone_to_sentiment(self, tone: str) -> str:
        """Map tone to sentiment type"""
        tone_to_sentiment = {
            "congratulatory": "positive",
            "encouraging": "positive",
            "gentle": "neutral",
            "supportive": "concerned",
            "alert": "concerned"
        }
        return tone_to_sentiment.get(tone, "neutral")

    def analyze_score_sentiment(self, score: float, language: str = "en",
                               override_thresholds: Optional[Dict[str, Any]] = None) -> SentimentResult:
        """
        Analyze sentiment based on numerical score

        Args:
            score: Numerical score (0-100)
            language: Language code for description
            override_thresholds: Optional threshold overrides

        Returns:
            SentimentResult object with analysis results
        """
        # Validate and apply fallback for language
        validated_language = self._validate_and_fallback_language(language)

        # Use configurable thresholds or overrides
        if override_thresholds:
            score_rules = override_thresholds.get("score_based", self.config["sentiment_rules"]["score_based"])
            logger.info("Using override thresholds for sentiment analysis")
        else:
            # Use thresholds from nudge config if available, otherwise fall back to sentiment config
            nudge_tone_mapping = self.nudge_config.get("tone_mapping", {}).get("score_ranges", {})
            if nudge_tone_mapping:
                score_rules = self._convert_nudge_config_to_score_rules(nudge_tone_mapping)
            else:
                score_rules = self.config["sentiment_rules"]["score_based"]

        # Find appropriate sentiment category based on score
        for category, rules in score_rules.items():
            if score >= rules["threshold"]:
                sentiment = SentimentType(rules["sentiment"])
                tone = ToneType(rules["tone"])
                description = rules["description"].get(validated_language, rules["description"]["en"])

                # Calculate confidence based on how far the score is from threshold
                confidence = min(1.0, (score - rules["threshold"]) / 20 + 0.6)

                return SentimentResult(
                    sentiment=sentiment,
                    tone=tone,
                    description=description,
                    urgency="low" if score >= 70 else "medium" if score >= 50 else "high",
                    confidence=confidence
                )

        # Fallback for very low scores
        fallback_description = "significant challenges"
        if "poor" in score_rules:
            fallback_description = score_rules["poor"]["description"].get(validated_language, score_rules["poor"]["description"]["en"])

        return SentimentResult(
            sentiment=SentimentType.CONCERNED,
            tone=ToneType.ALERT,
            description=fallback_description,
            urgency="high",
            confidence=0.9
        )

    def analyze_risk_sentiment(self, risk_level: str) -> SentimentResult:
        """
        Analyze sentiment based on risk level

        Args:
            risk_level: Risk level ("low", "medium", "high")

        Returns:
            SentimentResult object with analysis results
        """
        risk_rules = self.config["sentiment_rules"]["risk_based"]

        if risk_level not in risk_rules:
            risk_level = "medium"  # Default fallback

        rules = risk_rules[risk_level]

        return SentimentResult(
            sentiment=SentimentType(rules["sentiment"]),
            tone=ToneType(rules["tone"]),
            description=f"{risk_level} risk indicators",
            urgency=rules["nudge_urgency"],
            confidence=0.8
        )

    def analyze_engagement_sentiment(self, engagement_score: float, language: str = "en") -> SentimentResult:
        """
        Analyze sentiment based on engagement metrics

        Args:
            engagement_score: Engagement score (0-100)
            language: Language code for description

        Returns:
            SentimentResult object with analysis results
        """
        # Validate and apply fallback for language
        validated_language = self._validate_and_fallback_language(language)

        engagement_rules = self.config["sentiment_rules"]["engagement_based"]

        for category, rules in engagement_rules.items():
            if engagement_score >= rules["threshold"]:
                sentiment = SentimentType(rules["sentiment"])
                tone = ToneType(rules["tone"])

                # Map engagement level to description with extended language support
                description_map = {
                    "high_engagement": {
                        "en": "high engagement", "hi": "उच्च सहभागिता", "bn": "উচ্চ সম্পৃক্ততা",
                        "gu": "ઉચ્ચ સંલગ્નતા", "mr": "उच्च सहभाग", "ta": "அதிக ஈடுபாடு",
                        "te": "అధిక నిమగ్నత", "kn": "ಹೆಚ್ಚಿನ ನಿರತತೆ"
                    },
                    "medium_engagement": {
                        "en": "moderate engagement", "hi": "मध्यम सहभागिता", "bn": "মাঝারি সম্পৃক্ততা",
                        "gu": "મધ્યમ સંલગ્નતા", "mr": "मध्यम सहभाग", "ta": "மிதமான ஈடுபாடு",
                        "te": "మధ్యస్థ నిమగ్నత", "kn": "ಮಧ್ಯಮ ನಿರತತೆ"
                    },
                    "low_engagement": {
                        "en": "low engagement", "hi": "कम सहभागिता", "bn": "কম সম্পৃক্ততা",
                        "gu": "ઓછી સંલગ્નતા", "mr": "कमी सहभाग", "ta": "குறைந்த ஈடுபாடு",
                        "te": "తక్కువ నిమగ్నత", "kn": "ಕಡಿಮೆ ನಿರತತೆ"
                    }
                }

                description = description_map[category].get(validated_language, description_map[category]["en"])

                return SentimentResult(
                    sentiment=sentiment,
                    tone=tone,
                    description=description,
                    urgency="low" if engagement_score >= 80 else "medium" if engagement_score >= 50 else "high",
                    confidence=0.75
                )

        # Fallback for very low engagement
        return SentimentResult(
            sentiment=SentimentType.CONCERNED,
            tone=ToneType.SUPPORTIVE,
            description="very low engagement",
            urgency="high",
            confidence=0.9
        )

    def get_tone_formatting(self, tone: ToneType, language: str = "en") -> Dict[str, str]:
        """
        Get formatting information for a specific tone

        Args:
            tone: Tone type
            language: Language code

        Returns:
            Dictionary with formatting information (prefix, style, emoji)
        """
        # Validate and apply fallback for language
        validated_language = self._validate_and_fallback_language(language)

        tone_mappings = self.config["tone_mappings"]
        tone_key = tone.value

        if tone_key in tone_mappings:
            return tone_mappings[tone_key].get(validated_language, tone_mappings[tone_key]["en"])

        # Fallback formatting
        return {"prefix": "", "style": "neutral", "emoji": ""}

    def should_trigger_nudge(self, context: str, metric_type: str, value: float) -> Tuple[bool, str]:
        """
        Determine if a nudge should be triggered based on thresholds

        Args:
            context: Context ("edumentor" or "wellness")
            metric_type: Type of metric being evaluated
            value: Current value of the metric

        Returns:
            Tuple of (should_trigger, urgency_level)
        """
        thresholds = self.config["nudge_thresholds"]

        if context not in thresholds:
            return False, "low"

        context_thresholds = thresholds[context]

        # Handle nested wellness categories
        if context == "wellness" and "." in metric_type:
            category, metric = metric_type.split(".", 1)
            if category in context_thresholds and metric in context_thresholds[category]:
                threshold_config = context_thresholds[category][metric]
            else:
                return False, "low"
        else:
            if metric_type not in context_thresholds:
                return False, "low"
            threshold_config = context_thresholds[metric_type]

        threshold = threshold_config["threshold"]
        urgency = threshold_config["urgency"]

        # Determine if threshold is exceeded (logic varies by metric type)
        if metric_type in ["overspending"]:
            # For overspending, trigger if value exceeds threshold (percentage of budget)
            should_trigger = value >= threshold
        elif metric_type in ["low_savings", "low_activity"]:
            # For these metrics, trigger if value is below threshold
            should_trigger = value <= threshold
        else:
            # For most metrics, trigger if value meets or exceeds threshold
            should_trigger = value >= threshold

        return should_trigger, urgency if should_trigger else "low"

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """
        Get list of supported languages with metadata

        Returns:
            List of language dictionaries with code, name, and native_name
        """
        languages = self.language_config.get('languages_supported', [])
        return [
            {
                'code': lang['code'],
                'name': lang['name'],
                'native_name': lang.get('native_name', lang['name']),
                'enabled': lang.get('enabled', True)
            }
            for lang in languages if lang.get('enabled', True)
        ]

    def is_language_supported(self, language: str) -> bool:
        """
        Check if a language is supported

        Args:
            language: Language code to check

        Returns:
            True if language is supported, False otherwise
        """
        return language in self.supported_languages

    def get_risk_threshold(self, context: str, override_thresholds: Optional[Dict[str, Any]] = None) -> float:
        """
        Get risk threshold for a specific context

        Args:
            context: Context ("edumentor" or "wellness")
            override_thresholds: Optional threshold overrides

        Returns:
            Risk threshold value
        """
        if override_thresholds and "risk_thresholds" in override_thresholds:
            thresholds = override_thresholds["risk_thresholds"]
            logger.info(f"Using override risk threshold for context: {context}")
        else:
            thresholds = self.nudge_config.get("risk_thresholds", {})

        # Map context names
        context_key = "wellness_bot" if context == "wellness" else context

        return thresholds.get(context_key, {}).get("overall_risk", 0.7)

    def get_nudge_threshold(self, context: str, metric_type: str,
                           override_thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get nudge threshold for a specific metric

        Args:
            context: Context ("edumentor" or "wellness")
            metric_type: Type of metric
            override_thresholds: Optional threshold overrides

        Returns:
            Threshold configuration
        """
        if override_thresholds and "nudge_rules" in override_thresholds:
            rules = override_thresholds["nudge_rules"]
            logger.info(f"Using override nudge rules for {context}.{metric_type}")
        else:
            rules = self.nudge_config.get("nudge_rules", {})

        # Navigate to the specific rule
        if context == "wellness":
            # Handle nested wellness structure (financial/emotional)
            if "." in metric_type:
                category, metric = metric_type.split(".", 1)
                return rules.get("wellness_bot", {}).get(category, {}).get(metric, {})
            else:
                return rules.get("wellness_bot", {}).get(metric_type, {})
        else:
            return rules.get(context, {}).get("triggers", {}).get(metric_type, {})

    def validate_override_thresholds(self, override_thresholds: Dict[str, Any]) -> bool:
        """
        Validate override threshold values

        Args:
            override_thresholds: Threshold overrides to validate

        Returns:
            True if valid, False otherwise
        """
        if not self.nudge_config.get("override_settings", {}).get("allow_api_overrides", True):
            logger.warning("API overrides are disabled in configuration")
            return False

        validation_config = self.nudge_config.get("override_settings", {}).get("override_validation", {})
        min_threshold = validation_config.get("min_threshold", 0)
        max_threshold = validation_config.get("max_threshold", 100)

        # Validate threshold values are within acceptable range
        for key, value in override_thresholds.items():
            if isinstance(value, (int, float)):
                if not (min_threshold <= value <= max_threshold):
                    logger.error(f"Override threshold {key}={value} outside valid range [{min_threshold}, {max_threshold}]")
                    return False

        return True