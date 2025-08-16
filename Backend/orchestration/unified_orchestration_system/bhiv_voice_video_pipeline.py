"""
BHIV Voice and Video Pipeline
Handles the final step in the BHIV Core flow: VOICE → VIDEO

This module provides:
- Advanced TTS (Text-to-Speech) generation
- Voice response optimization
- Video response generation (future)
- Multi-language voice support
- Voice quality enhancement
- Audio file management and CDN integration
"""

import os
import json
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import tempfile
import uuid

# Audio processing imports
try:
    import requests
    import aiofiles
    import aiohttp
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logging.warning("Audio processing libraries not available")

# Local imports
from bhiv_core_schema import (
    UserInfo, OutputResponse, VoiceProcessingResult,
    OutputFormat
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TTSProvider:
    """Base class for TTS providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_available = False
        
    async def initialize(self) -> bool:
        """Initialize the TTS provider"""
        raise NotImplementedError
        
    async def generate_speech(
        self, 
        text: str, 
        language: str = "en",
        voice_model: str = "default",
        speed: float = 1.0,
        pitch: float = 1.0
    ) -> VoiceProcessingResult:
        """Generate speech from text"""
        raise NotImplementedError


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs TTS provider"""
    
    def __init__(self):
        super().__init__("ElevenLabs")
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default voice
        
    async def initialize(self) -> bool:
        """Initialize ElevenLabs TTS"""
        if not self.api_key:
            logger.warning("ElevenLabs API key not found")
            return False
            
        try:
            # Test API connection
            headers = {"xi-api-key": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/voices", headers=headers) as response:
                    if response.status == 200:
                        self.is_available = True
                        logger.info("ElevenLabs TTS initialized successfully")
                        return True
                    else:
                        logger.warning(f"ElevenLabs API test failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs TTS: {e}")
            return False
    
    async def generate_speech(
        self, 
        text: str, 
        language: str = "en",
        voice_model: str = "default",
        speed: float = 1.0,
        pitch: float = 1.0
    ) -> VoiceProcessingResult:
        """Generate speech using ElevenLabs"""
        if not self.is_available:
            return VoiceProcessingResult(error="ElevenLabs TTS not available")
            
        try:
            start_time = datetime.now()
            
            # Prepare request
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            # Generate speech
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        
                        # Save audio file
                        audio_url = await self._save_audio_file(audio_data, "elevenlabs")
                        
                        processing_time = (datetime.now() - start_time).total_seconds() * 1000
                        
                        return VoiceProcessingResult(
                            audio_url=audio_url,
                            duration_seconds=len(audio_data) / 16000,  # Estimate
                            voice_model="ElevenLabs",
                            processing_time_ms=processing_time
                        )
                    else:
                        error_text = await response.text()
                        return VoiceProcessingResult(
                            error=f"ElevenLabs API error: {response.status} - {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"ElevenLabs TTS generation failed: {e}")
            return VoiceProcessingResult(error=str(e))
    
    async def _save_audio_file(self, audio_data: bytes, provider: str) -> str:
        """Save audio file and return URL"""
        try:
            # Create unique filename
            file_id = str(uuid.uuid4())
            filename = f"{provider}_{file_id}.mp3"
            
            # Save to temporary directory (in production, use cloud storage)
            audio_dir = Path(tempfile.gettempdir()) / "bhiv_audio"
            audio_dir.mkdir(exist_ok=True)
            
            file_path = audio_dir / filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(audio_data)
            
            # Return URL (in production, this would be a CDN URL)
            base_url = os.getenv("AUDIO_BASE_URL", "http://localhost:8010/audio")
            return f"{base_url}/{filename}"
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            raise


class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS provider"""
    
    def __init__(self):
        super().__init__("OpenAI")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        
    async def initialize(self) -> bool:
        """Initialize OpenAI TTS"""
        if not self.api_key:
            logger.warning("OpenAI API key not found")
            return False
            
        self.is_available = True
        logger.info("OpenAI TTS initialized successfully")
        return True
    
    async def generate_speech(
        self, 
        text: str, 
        language: str = "en",
        voice_model: str = "alloy",
        speed: float = 1.0,
        pitch: float = 1.0
    ) -> VoiceProcessingResult:
        """Generate speech using OpenAI TTS"""
        if not self.is_available:
            return VoiceProcessingResult(error="OpenAI TTS not available")
            
        try:
            start_time = datetime.now()
            
            # Prepare request
            url = f"{self.base_url}/audio/speech"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "tts-1",
                "input": text,
                "voice": voice_model,
                "speed": speed
            }
            
            # Generate speech
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        
                        # Save audio file
                        audio_url = await self._save_audio_file(audio_data, "openai")
                        
                        processing_time = (datetime.now() - start_time).total_seconds() * 1000
                        
                        return VoiceProcessingResult(
                            audio_url=audio_url,
                            duration_seconds=len(audio_data) / 24000,  # Estimate
                            voice_model=f"OpenAI-{voice_model}",
                            processing_time_ms=processing_time
                        )
                    else:
                        error_text = await response.text()
                        return VoiceProcessingResult(
                            error=f"OpenAI API error: {response.status} - {error_text}"
                        )
                        
        except Exception as e:
            logger.error(f"OpenAI TTS generation failed: {e}")
            return VoiceProcessingResult(error=str(e))
    
    async def _save_audio_file(self, audio_data: bytes, provider: str) -> str:
        """Save audio file and return URL"""
        try:
            # Create unique filename
            file_id = str(uuid.uuid4())
            filename = f"{provider}_{file_id}.mp3"
            
            # Save to temporary directory (in production, use cloud storage)
            audio_dir = Path(tempfile.gettempdir()) / "bhiv_audio"
            audio_dir.mkdir(exist_ok=True)
            
            file_path = audio_dir / filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(audio_data)
            
            # Return URL (in production, this would be a CDN URL)
            base_url = os.getenv("AUDIO_BASE_URL", "http://localhost:8010/audio")
            return f"{base_url}/{filename}"
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            raise


class LocalTTSProvider(TTSProvider):
    """Local TTS provider using system TTS"""
    
    def __init__(self):
        super().__init__("Local")
        
    async def initialize(self) -> bool:
        """Initialize local TTS"""
        try:
            # Check if system TTS is available
            import pyttsx3
            self.engine = pyttsx3.init()
            self.is_available = True
            logger.info("Local TTS initialized successfully")
            return True
        except ImportError:
            logger.warning("pyttsx3 not available for local TTS")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize local TTS: {e}")
            return False
    
    async def generate_speech(
        self, 
        text: str, 
        language: str = "en",
        voice_model: str = "default",
        speed: float = 1.0,
        pitch: float = 1.0
    ) -> VoiceProcessingResult:
        """Generate speech using local TTS"""
        if not self.is_available:
            return VoiceProcessingResult(error="Local TTS not available")
            
        try:
            start_time = datetime.now()
            
            # Create unique filename
            file_id = str(uuid.uuid4())
            filename = f"local_{file_id}.wav"
            
            # Save to temporary directory
            audio_dir = Path(tempfile.gettempdir()) / "bhiv_audio"
            audio_dir.mkdir(exist_ok=True)
            file_path = audio_dir / filename
            
            # Configure TTS engine
            self.engine.setProperty('rate', int(200 * speed))
            
            # Generate speech
            self.engine.save_to_file(text, str(file_path))
            self.engine.runAndWait()
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Return URL
            base_url = os.getenv("AUDIO_BASE_URL", "http://localhost:8010/audio")
            audio_url = f"{base_url}/{filename}"
            
            return VoiceProcessingResult(
                audio_url=audio_url,
                duration_seconds=len(text) / 10,  # Rough estimate
                voice_model="Local-System",
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Local TTS generation failed: {e}")
            return VoiceProcessingResult(error=str(e))


class BHIVVoiceVideoOrchestrator:
    """
    Main orchestrator for BHIV Voice and Video Pipeline
    Manages multiple TTS providers and future video generation
    """

    def __init__(self):
        self.tts_providers = {}
        self.primary_provider = None
        self.fallback_providers = []
        self.voice_cache = {}
        self.performance_metrics = {
            "total_requests": 0,
            "successful_generations": 0,
            "cache_hits": 0,
            "avg_processing_time": 0.0,
            "provider_usage": {},
            "last_updated": datetime.now()
        }

    async def initialize(self):
        """Initialize all available TTS providers"""
        logger.info("Initializing BHIV Voice and Video Pipeline...")

        # Initialize providers
        providers = [
            ElevenLabsTTSProvider(),
            OpenAITTSProvider(),
            LocalTTSProvider()
        ]

        for provider in providers:
            try:
                if await provider.initialize():
                    self.tts_providers[provider.name] = provider
                    logger.info(f"TTS Provider '{provider.name}' initialized successfully")
                else:
                    logger.warning(f"TTS Provider '{provider.name}' failed to initialize")
            except Exception as e:
                logger.error(f"Error initializing TTS Provider '{provider.name}': {e}")

        # Set primary and fallback providers
        self._configure_provider_hierarchy()

        logger.info(f"Voice pipeline initialized with {len(self.tts_providers)} providers")

    def _configure_provider_hierarchy(self):
        """Configure provider hierarchy based on availability and preference"""
        # Preferred order: ElevenLabs > OpenAI > Local
        preferred_order = ["ElevenLabs", "OpenAI", "Local"]

        available_providers = list(self.tts_providers.keys())

        # Set primary provider
        for provider_name in preferred_order:
            if provider_name in available_providers:
                self.primary_provider = provider_name
                break

        # Set fallback providers
        self.fallback_providers = [
            name for name in preferred_order
            if name in available_providers and name != self.primary_provider
        ]

        logger.info(f"Primary TTS provider: {self.primary_provider}")
        logger.info(f"Fallback TTS providers: {self.fallback_providers}")

    async def generate_voice_response(
        self,
        text: str,
        user_info: UserInfo,
        output_format: OutputFormat = OutputFormat.RICH_TEXT_WITH_TTS,
        voice_preferences: Optional[Dict[str, Any]] = None
    ) -> VoiceProcessingResult:
        """
        Generate voice response with automatic provider fallback
        """
        if not text or not text.strip():
            return VoiceProcessingResult(error="No text provided for voice generation")

        # Check if voice generation is needed
        if output_format != OutputFormat.RICH_TEXT_WITH_TTS:
            return VoiceProcessingResult()  # No voice needed

        start_time = datetime.now()
        self.performance_metrics["total_requests"] += 1

        try:
            # Check cache first
            cache_key = self._generate_cache_key(text, user_info, voice_preferences)
            if cache_key in self.voice_cache:
                cached_result = self.voice_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self.performance_metrics["cache_hits"] += 1
                    logger.info(f"Voice cache hit for text: {text[:50]}...")
                    return cached_result["result"]

            # Prepare voice parameters
            voice_params = self._prepare_voice_parameters(user_info, voice_preferences)

            # Try primary provider first
            result = await self._try_provider(
                self.primary_provider, text, voice_params
            )

            # Try fallback providers if primary fails
            if result.error and self.fallback_providers:
                logger.warning(f"Primary provider {self.primary_provider} failed: {result.error}")

                for fallback_provider in self.fallback_providers:
                    logger.info(f"Trying fallback provider: {fallback_provider}")
                    result = await self._try_provider(
                        fallback_provider, text, voice_params
                    )

                    if not result.error:
                        break

            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            if not result.error:
                self.performance_metrics["successful_generations"] += 1

                # Cache successful result
                self.voice_cache[cache_key] = {
                    "result": result,
                    "timestamp": datetime.now(),
                    "ttl": timedelta(hours=24)  # Cache for 24 hours
                }

            self._update_performance_metrics(processing_time, result.voice_model)

            return result

        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
            return VoiceProcessingResult(error=str(e))

    async def _try_provider(
        self,
        provider_name: str,
        text: str,
        voice_params: Dict[str, Any]
    ) -> VoiceProcessingResult:
        """Try a specific TTS provider"""
        if provider_name not in self.tts_providers:
            return VoiceProcessingResult(error=f"Provider {provider_name} not available")

        provider = self.tts_providers[provider_name]

        try:
            result = await provider.generate_speech(
                text=text,
                language=voice_params.get("language", "en"),
                voice_model=voice_params.get("voice_model", "default"),
                speed=voice_params.get("speed", 1.0),
                pitch=voice_params.get("pitch", 1.0)
            )

            # Update provider usage metrics
            if provider_name not in self.performance_metrics["provider_usage"]:
                self.performance_metrics["provider_usage"][provider_name] = 0
            self.performance_metrics["provider_usage"][provider_name] += 1

            return result

        except Exception as e:
            logger.error(f"Provider {provider_name} failed: {e}")
            return VoiceProcessingResult(error=str(e))

    def _prepare_voice_parameters(
        self,
        user_info: UserInfo,
        voice_preferences: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare voice parameters based on user info and preferences"""
        params = {
            "language": user_info.lang or "en",
            "voice_model": "default",
            "speed": 1.0,
            "pitch": 1.0
        }

        # Apply voice preferences if provided
        if voice_preferences:
            params.update(voice_preferences)

        # Language-specific adjustments
        if user_info.lang:
            if user_info.lang.startswith("hi"):  # Hindi
                params["voice_model"] = "hindi_default"
            elif user_info.lang.startswith("es"):  # Spanish
                params["voice_model"] = "spanish_default"
            # Add more language mappings as needed

        return params

    def _generate_cache_key(
        self,
        text: str,
        user_info: UserInfo,
        voice_preferences: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for voice request"""
        cache_data = {
            "text": text,
            "language": user_info.lang,
            "preferences": voice_preferences or {}
        }

        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _is_cache_valid(self, cached_item: Dict[str, Any]) -> bool:
        """Check if cached voice result is still valid"""
        timestamp = cached_item.get("timestamp")
        ttl = cached_item.get("ttl", timedelta(hours=24))

        if not timestamp:
            return False

        return datetime.now() - timestamp < ttl

    def _update_performance_metrics(self, processing_time_ms: float, provider_used: Optional[str]):
        """Update performance metrics"""
        # Update average processing time
        total_requests = self.performance_metrics["total_requests"]
        current_avg = self.performance_metrics["avg_processing_time"]

        new_avg = ((current_avg * (total_requests - 1)) + processing_time_ms) / total_requests
        self.performance_metrics["avg_processing_time"] = new_avg
        self.performance_metrics["last_updated"] = datetime.now()

    async def generate_video_response(
        self,
        text: str,
        user_info: UserInfo,
        video_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate video response (future implementation)
        This is a placeholder for future video generation capabilities
        """
        logger.info("Video generation requested - feature coming soon")

        # Future implementation could include:
        # - Avatar-based video generation
        # - Lip-sync with generated audio
        # - Educational visual content
        # - Interactive video responses

        return {
            "video_url": None,
            "status": "not_implemented",
            "message": "Video generation feature coming soon",
            "estimated_implementation": "Q2 2024"
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get voice pipeline performance metrics"""
        return {
            **self.performance_metrics,
            "available_providers": list(self.tts_providers.keys()),
            "primary_provider": self.primary_provider,
            "fallback_providers": self.fallback_providers,
            "cache_size": len(self.voice_cache),
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] /
                max(self.performance_metrics["total_requests"], 1)
            ) * 100,
            "success_rate": (
                self.performance_metrics["successful_generations"] /
                max(self.performance_metrics["total_requests"], 1)
            ) * 100
        }

    def clear_cache(self):
        """Clear voice cache"""
        self.voice_cache.clear()
        logger.info("Voice cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for voice and video pipeline"""
        health_status = {
            "status": "healthy",
            "providers": {},
            "performance": self.get_performance_metrics(),
            "timestamp": datetime.now().isoformat()
        }

        # Check each provider
        for name, provider in self.tts_providers.items():
            health_status["providers"][name] = {
                "available": provider.is_available,
                "status": "healthy" if provider.is_available else "unavailable"
            }

        # Test voice generation
        try:
            test_result = await self.generate_voice_response(
                "Test voice generation",
                UserInfo(user_id="test", auth_token="test"),
                OutputFormat.RICH_TEXT_WITH_TTS
            )
            health_status["test_generation"] = "success" if not test_result.error else f"failed: {test_result.error}"
        except Exception as e:
            health_status["test_generation"] = f"failed: {str(e)}"
            health_status["status"] = "degraded"

        return health_status
