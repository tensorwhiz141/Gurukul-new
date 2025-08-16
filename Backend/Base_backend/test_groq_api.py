#!/usr/bin/env python3
"""
Test script for Groq API integration
"""

import os
import requests
from dotenv import load_dotenv

def test_groq_api():
    """Test Groq API connection and key validity"""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("❌ No GROQ_API_KEY found in .env file")
        return False
    
    # Remove quotes if present
    api_key = api_key.strip("'\"")
    
    print(f"🔑 API Key format: {api_key[:10]}...{api_key[-10:] if len(api_key) > 20 else api_key}")
    print(f"📏 API Key length: {len(api_key)}")
    print(f"✅ Starts with 'gsk_': {api_key.startswith('gsk_')}")
    
    # Test 1: Check API key format
    if not api_key.startswith('gsk_'):
        print("⚠️  Warning: Groq API keys typically start with 'gsk_'")
    
    # Test 2: Test API connection with minimal request
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Use a smaller model for testing
    payload = {
        'model': 'llama3-8b-8192',  # Smaller model, faster response
        'messages': [
            {'role': 'user', 'content': 'Hello'}
        ],
        'max_tokens': 10,
        'temperature': 0.1
    }
    
    print("\n🧪 Testing API connection...")
    
    try:
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
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
            print("💡 Solution: Get a new API key from https://console.groq.com/keys")
            
        elif response.status_code == 429:
            print("⚠️  429 Rate Limited - Too many requests")
            print("💡 Solution: Wait a moment and try again")
            
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

def test_alternative_models():
    """Test different Groq models"""
    
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY', '').strip("'\"")
    
    if not api_key:
        print("❌ No API key available for model testing")
        return
    
    models = [
        'llama3-8b-8192',
        'llama3-70b-8192', 
        'mixtral-8x7b-32768',
        'gemma-7b-it'
    ]
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    print("\n🔄 Testing different models...")
    
    for model in models:
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'max_tokens': 5
        }
        
        try:
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✅ {model}: Working")
            else:
                print(f"❌ {model}: Error {response.status_code}")
                
        except Exception as e:
            print(f"❌ {model}: Connection error")

def get_new_api_key_instructions():
    """Provide instructions for getting a new API key"""
    
    print("\n📋 How to get a new Groq API key:")
    print("1. Go to https://console.groq.com/")
    print("2. Sign up or log in to your account")
    print("3. Navigate to 'API Keys' section")
    print("4. Click 'Create API Key'")
    print("5. Copy the key (starts with 'gsk_')")
    print("6. Update your .env file:")
    print("   GROQ_API_KEY='your_new_key_here'")
    print("7. Restart your API service")

if __name__ == "__main__":
    print("🚀 Groq API Integration Test")
    print("=" * 40)
    
    success = test_groq_api()
    
    if success:
        test_alternative_models()
    else:
        get_new_api_key_instructions()
    
    print("\n" + "=" * 40)
    print("Test completed!")
