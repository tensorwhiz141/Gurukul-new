"""
Karthikeya Nudge Engine - Modular Architecture
Production-ready multilingual nudge system for Edumentor and Wellness Bot
"""

from .sentiment_analyzer import SentimentAnalyzer, SentimentType, ToneType, SentimentResult
from .template_loader import TemplateLoader
from .report_generator import ReportGenerator, ReportData, GeneratedReport
from .config_loader import ConfigLoader
from .api_endpoints import create_app
from .nudge_engine import NudgeEngine, NudgeContext, NudgeType, UrgencyLevel

__version__ = "2.0.0"
__author__ = "Karthikeya Team"

__all__ = [
    "SentimentAnalyzer",
    "SentimentType",
    "ToneType",
    "SentimentResult",
    "TemplateLoader",
    "ReportGenerator",
    "ReportData",
    "GeneratedReport",
    "ConfigLoader",
    "create_app",
    "NudgeEngine",
    "NudgeContext",
    "NudgeType",
    "UrgencyLevel"
]
