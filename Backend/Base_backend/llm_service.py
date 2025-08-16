"""
Enhanced LLM Service with multiple providers and fallback support
Supports: Groq, OpenAI, and local fallback responses
"""

import os
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    """Enhanced LLM service with multiple providers and fallback"""
    
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY', '').strip("'\"")
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '').strip("'\"")
        
        # Provider priority order
        self.providers = ['groq', 'openai', 'fallback']
        
        # Model configurations
        self.models = {
            'groq': {
                'default': 'llama3-8b-8192',
                'alternatives': ['llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
            },
            'openai': {
                'default': 'gpt-3.5-turbo',
                'alternatives': ['gpt-4', 'gpt-4-turbo-preview']
            }
        }
    
    def call_groq_api(self, prompt: str, model: str = None) -> Dict[str, Any]:
        """Call Groq API with improved error handling"""
        
        if not self.groq_api_key:
            return {"success": False, "error": "No Groq API key configured"}
        
        model = model or self.models['groq']['default']
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 1.0
        }
        
        try:
            logger.info(f"Calling Groq API with model: {model}")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                logger.info("✅ Groq API call successful")
                return {
                    "success": True,
                    "content": content,
                    "provider": "groq",
                    "model": model
                }
            
            elif response.status_code == 401:
                logger.error("❌ Groq API: Invalid or expired API key")
                return {"success": False, "error": "Invalid Groq API key"}
            
            elif response.status_code == 429:
                logger.error("⚠️ Groq API: Rate limit exceeded")
                return {"success": False, "error": "Groq rate limit exceeded"}
            
            else:
                logger.error(f"❌ Groq API error: {response.status_code} - {response.text}")
                return {"success": False, "error": f"Groq API error: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            logger.error("⏰ Groq API timeout")
            return {"success": False, "error": "Groq API timeout"}
            
        except requests.exceptions.ConnectionError:
            logger.error("🌐 Groq API connection error")
            return {"success": False, "error": "Groq API connection error"}
            
        except Exception as e:
            logger.error(f"❌ Groq API unexpected error: {e}")
            return {"success": False, "error": f"Groq API error: {str(e)}"}
    
    def call_openai_api(self, prompt: str, model: str = None) -> Dict[str, Any]:
        """Call OpenAI API as fallback"""
        
        if not self.openai_api_key:
            return {"success": False, "error": "No OpenAI API key configured"}
        
        model = model or self.models['openai']['default']
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        try:
            logger.info(f"Calling OpenAI API with model: {model}")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                logger.info("✅ OpenAI API call successful")
                return {
                    "success": True,
                    "content": content,
                    "provider": "openai",
                    "model": model
                }
            
            else:
                logger.error(f"❌ OpenAI API error: {response.status_code}")
                return {"success": False, "error": f"OpenAI API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return {"success": False, "error": f"OpenAI API error: {str(e)}"}
    
    def get_fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Generate fallback response when APIs are unavailable"""
        
        # Simple keyword-based responses
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            content = "Hello! I'm your AI assistant. How can I help you today? (Note: I'm currently running in fallback mode as the main AI services are temporarily unavailable.)"
        
        elif any(word in prompt_lower for word in ['how are you', 'how do you do']):
            content = "I'm doing well, thank you for asking! I'm here to help you with any questions you might have. How can I assist you today?"
        
        elif any(word in prompt_lower for word in ['help', 'assist', 'support']):
            content = "I'm here to help! I can assist you with various topics including education, general questions, and guidance. What would you like to know about?"
        
        elif any(word in prompt_lower for word in ['thank', 'thanks']):
            content = "You're very welcome! I'm glad I could help. Is there anything else you'd like to know?"
        
        elif any(word in prompt_lower for word in ['bye', 'goodbye', 'see you']):
            content = "Goodbye! It was nice chatting with you. Feel free to come back anytime if you have more questions!"
        
        else:
            content = f"I understand you're asking about '{prompt[:50]}...'. While I'm currently in fallback mode, I'd be happy to help once our main AI services are restored. In the meantime, could you please rephrase your question or try again later?"
        
        logger.info("📝 Using fallback response")
        return {
            "success": True,
            "content": content,
            "provider": "fallback",
            "model": "rule-based"
        }
    
    def generate_response(self, prompt: str, preferred_provider: str = None) -> str:
        """
        Generate response with automatic fallback between providers
        
        Args:
            prompt: User input text
            preferred_provider: 'groq', 'openai', or None for auto
            
        Returns:
            Generated response text
        """
        
        providers_to_try = [preferred_provider] if preferred_provider else self.providers
        
        for provider in providers_to_try:
            if provider == 'groq':
                result = self.call_groq_api(prompt)
                if result["success"]:
                    return result["content"]
                logger.warning(f"Groq failed: {result.get('error', 'Unknown error')}")
            
            elif provider == 'openai':
                result = self.call_openai_api(prompt)
                if result["success"]:
                    return result["content"]
                logger.warning(f"OpenAI failed: {result.get('error', 'Unknown error')}")
            
            elif provider == 'fallback':
                result = self.get_fallback_response(prompt)
                return result["content"]
        
        # Final fallback if everything fails
        return "I apologize, but I'm experiencing technical difficulties right now. Please try again in a few moments, or contact support if the issue persists."
    
    def test_providers(self) -> Dict[str, bool]:
        """Test all available providers"""
        
        test_prompt = "Hello"
        results = {}
        
        # Test Groq
        groq_result = self.call_groq_api(test_prompt)
        results['groq'] = groq_result["success"]
        
        # Test OpenAI
        openai_result = self.call_openai_api(test_prompt)
        results['openai'] = openai_result["success"]
        
        # Fallback always works
        results['fallback'] = True
        
        return results

# Global instance
llm_service = LLMService()

# Backward compatibility function
def call_groq_llama3(prompt: str) -> str:
    """Backward compatible function with enhanced capabilities"""
    return llm_service.generate_response(prompt)

# Test function
if __name__ == "__main__":
    service = LLMService()
    
    print("🧪 Testing LLM Service...")
    test_results = service.test_providers()
    
    for provider, status in test_results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {provider.capitalize()}: {'Working' if status else 'Failed'}")
    
    print("\n🗣️ Testing response generation...")
    response = service.generate_response("Hello, how are you?")
    print(f"Response: {response}")
