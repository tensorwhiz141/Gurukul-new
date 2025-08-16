#!/usr/bin/env python3
"""
Test script for OpenAI API integration
"""

import os
import requests
from dotenv import load_dotenv

def test_openai_api():
    """Test OpenAI API connection and key validity"""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No OPENAI_API_KEY found in .env file")
        return False
    
    # Remove quotes if present
    api_key = api_key.strip("'\"")
    
    print(f"🔑 API Key format: {api_key[:10]}...{api_key[-10:] if len(api_key) > 20 else api_key}")
    print(f"📏 API Key length: {len(api_key)}")
    print(f"✅ Starts with 'sk-': {api_key.startswith('sk-')}")
    
    # Test API connection with minimal request
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'user', 'content': 'Hello'}
        ],
        'max_tokens': 10,
        'temperature': 0.1
    }
    
    print("\n🧪 Testing API connection...")
    
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"✅ SUCCESS! API Response: {message}")
            return True
            
        elif response.status_code == 401:
            print("❌ 401 Unauthorized - API key is invalid or expired")
            print("💡 Solution: Get a new API key from https://platform.openai.com/api-keys")
            
        elif response.status_code == 429:
            print("⚠️  429 Rate Limited - Too many requests or insufficient credits")
            print("💡 Solution: Check your OpenAI billing and usage limits")
            
        elif response.status_code == 400:
            print("❌ 400 Bad Request - Invalid request format")
            print(f"Response: {response.text}")
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
        return False
        
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - Check your internet connection")
        return False
        
    except requests.exceptions.ConnectionError:
        print("🌐 Connection error - Check your internet connection")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 OpenAI API Integration Test")
    print("=" * 40)
    
    success = test_openai_api()
    
    if success:
        print("\n✅ OpenAI API is working! This can be used as fallback.")
    else:
        print("\n❌ OpenAI API is not working.")
    
    print("\n" + "=" * 40)
    print("Test completed!")
