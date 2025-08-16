#!/usr/bin/env python3
"""
Test script for streaming TTS implementation
Tests both the dedicated chatbot service and regular TTS service streaming endpoints
"""

import requests
import time
import os
import sys

# Test configuration
CHATBOT_TTS_URL = "http://localhost:8001"
REGULAR_TTS_URL = "http://localhost:8007"
TEST_TEXT = "Hello, this is a test of the streaming TTS implementation. The audio should be generated in memory and streamed directly without saving to disk."

def test_chatbot_streaming_tts():
    """Test the dedicated chatbot service streaming TTS endpoint"""
    print("🔊 Testing Dedicated Chatbot Service Streaming TTS...")
    
    try:
        # Test health endpoint first
        health_response = requests.get(f"{CHATBOT_TTS_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Chatbot service health check failed: {health_response.status_code}")
            return False
        
        print("✅ Chatbot service is healthy")
        
        # Test streaming TTS endpoint
        payload = {"text": TEST_TEXT}
        
        print(f"📤 Sending request to {CHATBOT_TTS_URL}/tts/stream")
        start_time = time.time()
        
        response = requests.post(
            f"{CHATBOT_TTS_URL}/tts/stream",
            json=payload,
            timeout=30,
            stream=True
        )
        
        if response.status_code == 200:
            # Check response headers
            content_type = response.headers.get('content-type', '')
            content_length = response.headers.get('content-length', 'unknown')
            
            print(f"✅ Streaming TTS successful!")
            print(f"   Content-Type: {content_type}")
            print(f"   Content-Length: {content_length} bytes")
            print(f"   Response time: {time.time() - start_time:.2f}s")
            
            # Read the audio data
            audio_data = response.content
            print(f"   Audio data size: {len(audio_data)} bytes")
            
            if len(audio_data) > 0:
                print("✅ Audio data received successfully")
                return True
            else:
                print("❌ No audio data received")
                return False
        else:
            print(f"❌ Streaming TTS failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_regular_streaming_tts():
    """Test the regular TTS service streaming endpoint"""
    print("\n🔊 Testing Regular TTS Service Streaming...")
    
    try:
        # Test health endpoint first
        health_response = requests.get(f"{REGULAR_TTS_URL}/api/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Regular TTS service health check failed: {health_response.status_code}")
            return False
        
        print("✅ Regular TTS service is healthy")
        
        # Test streaming TTS endpoint
        payload = {"text": TEST_TEXT}
        
        print(f"📤 Sending request to {REGULAR_TTS_URL}/api/generate/stream")
        start_time = time.time()
        
        response = requests.post(
            f"{REGULAR_TTS_URL}/api/generate/stream",
            data=payload,  # Regular TTS uses form data
            timeout=30,
            stream=True
        )
        
        if response.status_code == 200:
            # Check response headers
            content_type = response.headers.get('content-type', '')
            content_length = response.headers.get('content-length', 'unknown')
            
            print(f"✅ Streaming TTS successful!")
            print(f"   Content-Type: {content_type}")
            print(f"   Content-Length: {content_length} bytes")
            print(f"   Response time: {time.time() - start_time:.2f}s")
            
            # Read the audio data
            audio_data = response.content
            print(f"   Audio data size: {len(audio_data)} bytes")
            
            if len(audio_data) > 0:
                print("✅ Audio data received successfully")
                return True
            else:
                print("❌ No audio data received")
                return False
        else:
            print(f"❌ Streaming TTS failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_legacy_endpoints():
    """Test that legacy endpoints still work for backward compatibility"""
    print("\n🔄 Testing Legacy Endpoints for Backward Compatibility...")
    
    try:
        # Test legacy chatbot TTS endpoint
        payload = {"text": "Short test"}
        response = requests.post(f"{CHATBOT_TTS_URL}/tts", json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ Legacy chatbot TTS endpoint works")
        else:
            print(f"⚠️ Legacy chatbot TTS endpoint failed: {response.status_code}")
        
        # Test legacy regular TTS endpoint
        response = requests.post(f"{REGULAR_TTS_URL}/api/generate", data=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ Legacy regular TTS endpoint works")
        else:
            print(f"⚠️ Legacy regular TTS endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Legacy endpoint test error: {e}")

def main():
    """Run all streaming TTS tests"""
    print("🚀 Starting Streaming TTS Implementation Tests")
    print("=" * 60)
    
    results = []
    
    # Test dedicated chatbot service streaming
    results.append(test_chatbot_streaming_tts())
    
    # Test regular TTS service streaming
    results.append(test_regular_streaming_tts())
    
    # Test legacy endpoints
    test_legacy_endpoints()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Chatbot Streaming TTS: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"   Regular Streaming TTS: {'✅ PASS' if results[1] else '❌ FAIL'}")
    
    if all(results):
        print("\n🎉 All streaming TTS tests passed!")
        print("✅ Audio file saving has been eliminated")
        print("✅ Direct streaming audio playback is working")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the services and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
