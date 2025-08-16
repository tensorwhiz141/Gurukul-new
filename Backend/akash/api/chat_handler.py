"""
Chat Handler for Agent API Integration
Handles communication with Vedant's /ask-agent API and chat history management
"""

import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from models.schemas import ChatMessage

logger = logging.getLogger(__name__)

class ChatHandler:
    def __init__(self):
        self.agent_api_url = os.getenv("AGENT_API_URL", "http://localhost:8000/ask-agent")
        self.agent_api_key = os.getenv("AGENT_API_KEY")
        self.tts_enabled = os.getenv("TTS_ENABLED", "false").lower() == "true"
        self.tts_api_url = os.getenv("TTS_API_URL")
        
        if not self.agent_api_key:
            logger.warning("AGENT_API_KEY not set - API calls may fail")
    
    def format_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
        """
        Format chat history for inclusion in agent prompt
        
        Args:
            chat_history: List of chat messages from MongoDB
            
        Returns:
            Formatted string representation of chat history
        """
        if not chat_history:
            return ""
        
        formatted_history = "Previous conversation:\n"
        for msg in reversed(chat_history):  # Reverse to show chronological order
            timestamp = msg.get("timestamp", "").strftime("%Y-%m-%d %H:%M:%S") if msg.get("timestamp") else ""
            formatted_history += f"[{timestamp}] User: {msg.get('message', '')}\n"
            formatted_history += f"[{timestamp}] Assistant: {msg.get('response', '')}\n\n"
        
        return formatted_history
    
    async def call_agent_api(
        self, 
        user_message: str, 
        chat_history: Optional[List[Dict[str, Any]]] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Call Vedant's /ask-agent API with user message and history
        
        Args:
            user_message: User's current message
            chat_history: Previous chat messages
            user_id: User identifier for context
            
        Returns:
            API response containing agent's reply
        """
        try:
            # Prepare the request payload
            payload = {
                "query": user_message,
                "user_id": user_id
            }
            
            # Include chat history if available
            if chat_history:
                formatted_history = self.format_chat_history(chat_history)
                payload["context"] = formatted_history
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.agent_api_key:
                headers["Authorization"] = f"Bearer {self.agent_api_key}"
            
            # Make the API call
            logger.info(f"Calling agent API for user {user_id}")
            response = requests.post(
                self.agent_api_url,
                json=payload,
                headers=headers,
                timeout=30  # 30 second timeout
            )
            
            response.raise_for_status()
            
            api_response = response.json()
            logger.info(f"Agent API call successful for user {user_id}")
            
            return {
                "success": True,
                "response": api_response.get("response", ""),
                "metadata": {
                    "api_response_time": response.elapsed.total_seconds(),
                    "status_code": response.status_code,
                    "model_used": api_response.get("model", "unknown"),
                    "tokens_used": api_response.get("tokens", 0)
                }
            }
            
        except requests.exceptions.Timeout:
            logger.error("Agent API call timed out")
            return {
                "success": False,
                "error": "Agent API timeout",
                "response": "I'm sorry, but I'm experiencing some delays. Please try again in a moment."
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Agent API call failed: {str(e)}")
            return {
                "success": False,
                "error": f"Agent API error: {str(e)}",
                "response": "I'm having trouble connecting to my brain right now. Please try again later."
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in agent API call: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "response": "Something unexpected happened. Please try again."
            }
    
    async def generate_tts_audio(self, text: str, user_id: str) -> Optional[str]:
        """
        Generate TTS audio for the response (if enabled)
        
        Args:
            text: Text to convert to speech
            user_id: User identifier
            
        Returns:
            URL to the generated audio file or None if TTS is disabled/failed
        """
        if not self.tts_enabled or not self.tts_api_url:
            return None
        
        try:
            payload = {
                "text": text,
                "user_id": user_id,
                "voice": "default"  # Can be made configurable
            }
            
            response = requests.post(
                self.tts_api_url,
                json=payload,
                timeout=15
            )
            
            response.raise_for_status()
            tts_response = response.json()
            
            audio_url = tts_response.get("audio_url")
            if audio_url:
                logger.info(f"TTS audio generated for user {user_id}")
                return audio_url
            else:
                logger.warning("TTS API did not return audio URL")
                return None
                
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            return None
    
    def validate_message(self, message: str) -> bool:
        """
        Validate user message before processing
        
        Args:
            message: User's message
            
        Returns:
            True if message is valid, False otherwise
        """
        if not message or not message.strip():
            return False
        
        if len(message) > 10000:  # Reasonable limit
            return False
        
        return True
