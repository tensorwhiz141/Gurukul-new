"""
Configuration Loader Module for Karthikeya Multilingual Reporting Engine
Handles loading and management of YAML configuration files
"""

import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Manages loading and caching of YAML configuration files
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration loader
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.language_config = {}
        self.nudge_config = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files"""
        self._load_language_config()
        self._load_nudge_config()
    
    def _load_language_config(self):
        """Load language configuration"""
        config_path = self.config_dir / "language_config.yaml"
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.language_config = yaml.safe_load(f)
                logger.info(f"Loaded language config from {config_path}")
            else:
                logger.warning(f"Language config file not found: {config_path}")
                self.language_config = self._get_default_language_config()
                
        except yaml.YAMLError as e:
            logger.error(f"Error parsing language config YAML: {e}")
            self.language_config = self._get_default_language_config()
        except Exception as e:
            logger.error(f"Error loading language config: {e}")
            self.language_config = self._get_default_language_config()
    
    def _load_nudge_config(self):
        """Load nudge configuration"""
        config_path = self.config_dir / "nudge_config.yaml"
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.nudge_config = yaml.safe_load(f)
                logger.info(f"Loaded nudge config from {config_path}")
            else:
                logger.warning(f"Nudge config file not found: {config_path}")
                self.nudge_config = self._get_default_nudge_config()
                
        except yaml.YAMLError as e:
            logger.error(f"Error parsing nudge config YAML: {e}")
            self.nudge_config = self._get_default_nudge_config()
        except Exception as e:
            logger.error(f"Error loading nudge config: {e}")
            self.nudge_config = self._get_default_nudge_config()
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """
        Get list of supported languages with metadata
        
        Returns:
            List of language dictionaries
        """
        return self.language_config.get('languages_supported', [])
    
    def get_language_codes(self) -> List[str]:
        """
        Get list of supported language codes
        
        Returns:
            List of language codes
        """
        languages = self.get_supported_languages()
        return [lang['code'] for lang in languages if lang.get('enabled', True)]
    
    def get_fallback_language(self) -> str:
        """
        Get fallback language code
        
        Returns:
            Fallback language code
        """
        return self.language_config.get('fallback_language', 'en')
    
    def is_language_supported(self, language_code: str) -> bool:
        """
        Check if language is supported
        
        Args:
            language_code: Language code to check
            
        Returns:
            True if supported, False otherwise
        """
        return language_code in self.get_language_codes()
    
    def get_risk_threshold(self, context: str) -> float:
        """
        Get risk threshold for context
        
        Args:
            context: Context name (edumentor or wellness)
            
        Returns:
            Risk threshold value
        """
        thresholds = self.nudge_config.get('risk_thresholds', {})
        
        # Map context names
        context_key = "wellness_bot" if context == "wellness" else context
        
        return thresholds.get(context_key, {}).get('overall_risk', 0.7)
    
    def get_tone_mapping(self) -> Dict[str, Any]:
        """
        Get tone mapping configuration
        
        Returns:
            Tone mapping dictionary
        """
        return self.nudge_config.get('tone_mapping', {})
    
    def get_nudge_rules(self, context: str) -> Dict[str, Any]:
        """
        Get nudge rules for context
        
        Args:
            context: Context name
            
        Returns:
            Nudge rules dictionary
        """
        rules = self.nudge_config.get('nudge_rules', {})
        
        # Map context names
        context_key = "wellness_bot" if context == "wellness" else context
        
        return rules.get(context_key, {})
    
    def get_nudge_behavior_settings(self) -> Dict[str, Any]:
        """
        Get nudge behavior settings
        
        Returns:
            Nudge behavior settings dictionary
        """
        return self.nudge_config.get('nudge_behavior', {})
    
    def validate_override_thresholds(self, overrides: Dict[str, Any]) -> bool:
        """
        Validate threshold override values
        
        Args:
            overrides: Override values to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.nudge_config.get('override_settings', {}).get('allow_api_overrides', True):
            logger.warning("API overrides are disabled in configuration")
            return False
        
        validation_config = self.nudge_config.get('override_settings', {}).get('override_validation', {})
        min_threshold = validation_config.get('min_threshold', 0)
        max_threshold = validation_config.get('max_threshold', 100)
        
        # Validate threshold values are within acceptable range
        for key, value in overrides.items():
            if isinstance(value, (int, float)):
                if not (min_threshold <= value <= max_threshold):
                    logger.error(f"Override threshold {key}={value} outside valid range [{min_threshold}, {max_threshold}]")
                    return False
        
        return True
    
    def get_regional_preferences(self, region: str = "IN") -> Dict[str, Any]:
        """
        Get regional language preferences
        
        Args:
            region: Region code
            
        Returns:
            Regional preferences dictionary
        """
        regional_mappings = self.language_config.get('regional_mappings', {})
        return regional_mappings.get(region, {})
    
    def reload_configs(self):
        """Reload all configurations from disk"""
        self.language_config.clear()
        self.nudge_config.clear()
        self._load_all_configs()
        logger.info("All configurations reloaded from disk")
    
    def _get_default_language_config(self) -> Dict[str, Any]:
        """Get default language configuration"""
        return {
            'languages_supported': [
                {'code': 'en', 'name': 'English', 'native_name': 'English', 'enabled': True},
                {'code': 'hi', 'name': 'Hindi', 'native_name': 'हिन्दी', 'enabled': True},
                {'code': 'bn', 'name': 'Bengali', 'native_name': 'বাংলা', 'enabled': True},
                {'code': 'gu', 'name': 'Gujarati', 'native_name': 'ગુજરાતી', 'enabled': True},
                {'code': 'mr', 'name': 'Marathi', 'native_name': 'मराठी', 'enabled': True},
                {'code': 'ta', 'name': 'Tamil', 'native_name': 'தமிழ்', 'enabled': True},
                {'code': 'te', 'name': 'Telugu', 'native_name': 'తెలుగు', 'enabled': True},
                {'code': 'kn', 'name': 'Kannada', 'native_name': 'ಕನ್ನಡ', 'enabled': True}
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
            },
            'override_settings': {
                'allow_api_overrides': True,
                'override_validation': {'min_threshold': 0, 'max_threshold': 100}
            }
        }
    
    def validate_config(self, config_type: str = "all") -> bool:
        """
        Validate configuration structure
        
        Args:
            config_type: Type of config to validate ("language", "nudge", or "all")
            
        Returns:
            True if valid, False otherwise
        """
        if config_type in ["language", "all"]:
            if not self._validate_language_config():
                return False
        
        if config_type in ["nudge", "all"]:
            if not self._validate_nudge_config():
                return False
        
        return True
    
    def _validate_language_config(self) -> bool:
        """Validate language configuration structure"""
        required_fields = ['languages_supported', 'fallback_language']
        
        for field in required_fields:
            if field not in self.language_config:
                logger.error(f"Missing required field in language config: {field}")
                return False
        
        # Validate languages_supported structure
        languages = self.language_config.get('languages_supported', [])
        if not isinstance(languages, list):
            logger.error("languages_supported must be a list")
            return False
        
        for lang in languages:
            if not isinstance(lang, dict) or 'code' not in lang:
                logger.error("Each language must be a dict with 'code' field")
                return False
        
        return True
    
    def _validate_nudge_config(self) -> bool:
        """Validate nudge configuration structure"""
        required_fields = ['risk_thresholds', 'tone_mapping']
        
        for field in required_fields:
            if field not in self.nudge_config:
                logger.error(f"Missing required field in nudge config: {field}")
                return False
        
        return True
