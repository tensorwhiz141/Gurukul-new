"""
Template Loader Module for Karthikeya Multilingual Reporting Engine
Handles loading and management of multilingual report templates
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class TemplateLoader:
    """
    Manages loading and caching of multilingual report templates
    """
    
    def __init__(self, templates_dir: str = "templates"):
        """
        Initialize template loader
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = Path(templates_dir)
        self.templates_cache = {}
        self.sentiment_mappings_cache = {}
        self._load_all_templates()
    
    def _load_all_templates(self):
        """Load all templates into cache"""
        try:
            # Load report templates
            report_templates_path = self.templates_dir / "report_templates.json"
            if report_templates_path.exists():
                with open(report_templates_path, 'r', encoding='utf-8') as f:
                    self.templates_cache = json.load(f)
                logger.info(f"Loaded report templates from {report_templates_path}")
            else:
                logger.warning(f"Report templates file not found: {report_templates_path}")
                self.templates_cache = self._get_default_templates()
            
            # Load sentiment mappings
            sentiment_mappings_path = self.templates_dir / "sentiment_mappings.json"
            if sentiment_mappings_path.exists():
                with open(sentiment_mappings_path, 'r', encoding='utf-8') as f:
                    self.sentiment_mappings_cache = json.load(f)
                logger.info(f"Loaded sentiment mappings from {sentiment_mappings_path}")
            else:
                logger.warning(f"Sentiment mappings file not found: {sentiment_mappings_path}")
                self.sentiment_mappings_cache = self._get_default_sentiment_mappings()
                
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self.templates_cache = self._get_default_templates()
            self.sentiment_mappings_cache = self._get_default_sentiment_mappings()
    
    def get_template(self, context: str, language: str = "en") -> Dict[str, Any]:
        """
        Get template for specific context and language
        
        Args:
            context: Context (edumentor or wellness)
            language: Language code
            
        Returns:
            Template dictionary
        """
        try:
            return self.templates_cache[context][language]
        except KeyError:
            logger.warning(f"Template not found for context={context}, language={language}")
            # Fallback to English
            try:
                return self.templates_cache[context]["en"]
            except KeyError:
                logger.error(f"No fallback template found for context={context}")
                return self._get_default_template()
    
    def get_sentiment_mapping(self, sentiment_type: str, language: str = "en") -> Dict[str, str]:
        """
        Get sentiment mapping for specific type and language
        
        Args:
            sentiment_type: Type of sentiment mapping
            language: Language code
            
        Returns:
            Sentiment mapping dictionary
        """
        try:
            return self.sentiment_mappings_cache["sentiment_rules"][sentiment_type]
        except KeyError:
            logger.warning(f"Sentiment mapping not found for type={sentiment_type}")
            return {"description": {"en": "neutral progress"}}
    
    def get_tone_formatting(self, tone: str, language: str = "en") -> Dict[str, str]:
        """
        Get tone formatting for specific tone and language
        
        Args:
            tone: Tone type
            language: Language code
            
        Returns:
            Tone formatting dictionary
        """
        try:
            return self.sentiment_mappings_cache["tone_mappings"][tone][language]
        except KeyError:
            logger.warning(f"Tone formatting not found for tone={tone}, language={language}")
            # Fallback to English
            try:
                return self.sentiment_mappings_cache["tone_mappings"][tone]["en"]
            except KeyError:
                return {"prefix": "", "style": "neutral", "emoji": ""}
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages from templates
        
        Returns:
            List of language codes
        """
        languages = set()
        for context_templates in self.templates_cache.values():
            if isinstance(context_templates, dict):
                languages.update(context_templates.keys())
        return sorted(list(languages))
    
    def get_supported_contexts(self) -> List[str]:
        """
        Get list of supported contexts
        
        Returns:
            List of context names
        """
        return list(self.templates_cache.keys())
    
    def reload_templates(self):
        """Reload templates from disk"""
        self.templates_cache.clear()
        self.sentiment_mappings_cache.clear()
        self._load_all_templates()
        logger.info("Templates reloaded from disk")
    
    def _get_default_template(self) -> Dict[str, Any]:
        """Get default template structure"""
        return {
            "title": "Report",
            "summary": "Summary not available",
            "risk_assessment": {
                "low": "Status: Good",
                "medium": "Status: Moderate", 
                "high": "Status: Needs attention"
            },
            "recommendations": {
                "low": "Continue current approach",
                "medium": "Consider improvements",
                "high": "Immediate action recommended"
            },
            "nudge_triggers": {
                "default": "Stay engaged with your progress"
            }
        }
    
    def _get_default_templates(self) -> Dict[str, Any]:
        """Get default templates structure"""
        default_template = self._get_default_template()
        return {
            "edumentor": {"en": default_template},
            "wellness": {"en": default_template}
        }
    
    def _get_default_sentiment_mappings(self) -> Dict[str, Any]:
        """Get default sentiment mappings"""
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
                    "poor": {
                        "threshold": 0,
                        "sentiment": "concerned",
                        "tone": "alert",
                        "description": {"en": "needs improvement"}
                    }
                }
            },
            "tone_mappings": {
                "congratulatory": {
                    "en": {"prefix": "🎉 Excellent! ", "style": "enthusiastic", "emoji": "🌟"}
                },
                "encouraging": {
                    "en": {"prefix": "👍 Good work! ", "style": "positive", "emoji": "💪"}
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
    
    def validate_template(self, template: Dict[str, Any]) -> bool:
        """
        Validate template structure
        
        Args:
            template: Template to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["title", "summary", "risk_assessment", "recommendations"]
        
        for field in required_fields:
            if field not in template:
                logger.error(f"Missing required field in template: {field}")
                return False
        
        return True
