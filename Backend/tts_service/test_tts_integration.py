#!/usr/bin/env python3
"""
TTS Integration Test Script
Tests the complete TTS integration workflow for Gurukul application
"""

import requests
import time
import json
import os
from pathlib import Path

# Configuration
TTS_SERVICE_URL = "http://localhost:8007"
SUBJECT_GENERATION_URL = "http://localhost:8005"

def test_tts_service_health():
    """Test if TTS service is running and healthy"""
    print("🔊 Testing TTS Service Health...")
    
    try:
        response = requests.get(f"{TTS_SERVICE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ TTS Service is healthy")
            print(f"   Status: {health_data.get('status')}")
            print(f"   TTS Engine: {health_data.get('tts_engine')}")
            print(f"   Voices Available: {health_data.get('voices_available')}")
            print(f"   Audio Files Count: {health_data.get('audio_files_count')}")
            return True
        else:
            print(f"❌ TTS Service health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ TTS Service health check error: {e}")
        return False

def test_tts_generation():
    """Test TTS generation with sample text"""
    print("\n🔊 Testing TTS Generation...")
    
    sample_text = "Welcome to Gurukul! This is a test of our text-to-speech integration. The system can now automatically convert AI-generated lesson content into natural speech."
    
    try:
        # Prepare form data
        form_data = {'text': sample_text}
        
        response = requests.post(
            f"{TTS_SERVICE_URL}/api/generate",
            data=form_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ TTS Generation successful")
            print(f"   Filename: {result.get('filename')}")
            print(f"   Audio URL: {result.get('audio_url')}")
            print(f"   File Size: {result.get('file_size')} bytes")
            print(f"   Text Length: {result.get('text_length')} characters")
            
            # Test audio file access
            audio_url = f"{TTS_SERVICE_URL}{result.get('audio_url')}"
            audio_response = requests.head(audio_url, timeout=10)
            if audio_response.status_code == 200:
                print(f"✅ Audio file is accessible")
                return True
            else:
                print(f"❌ Audio file not accessible: {audio_response.status_code}")
                return False
        else:
            print(f"❌ TTS Generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ TTS Generation error: {e}")
        return False

def test_subject_generation_tts():
    """Test TTS integration with subject generation service"""
    print("\n📚 Testing Subject Generation TTS Integration...")
    
    try:
        # Test if subject generation service is running
        health_response = requests.get(f"{SUBJECT_GENERATION_URL}/health", timeout=10)
        if health_response.status_code != 200:
            print(f"❌ Subject Generation service not available: {health_response.status_code}")
            return False
        
        print("✅ Subject Generation service is running")
        
        # Test TTS generation endpoint
        tts_request = {
            "text": "This is a test lesson about ancient Indian mathematics. The concept of zero was first developed in India and revolutionized mathematical thinking worldwide.",
            "user_id": "test-user",
            "description": "Test TTS integration"
        }
        
        response = requests.post(
            f"{SUBJECT_GENERATION_URL}/tts/generate",
            json=tts_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Subject Generation TTS integration successful")
            print(f"   Status: {result.get('status')}")
            print(f"   Message: {result.get('message')}")
            
            tts_result = result.get('tts_result', {})
            if tts_result.get('status') == 'success':
                print(f"   TTS Audio URL: {tts_result.get('access_info', {}).get('audio_url')}")
                return True
            else:
                print(f"❌ TTS generation failed in subject service")
                return False
        else:
            print(f"❌ Subject Generation TTS failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Subject Generation TTS error: {e}")
        return False

def test_audio_file_listing():
    """Test audio file listing endpoint"""
    print("\n📁 Testing Audio File Listing...")
    
    try:
        response = requests.get(f"{TTS_SERVICE_URL}/api/list-audio-files", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Audio file listing successful")
            print(f"   Total Files: {result.get('count')}")
            
            files = result.get('audio_files', [])
            if files:
                print(f"   Recent Files: {files[:3]}")  # Show first 3 files
            return True
        else:
            print(f"❌ Audio file listing failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Audio file listing error: {e}")
        return False

def test_frontend_integration():
    """Test frontend integration points"""
    print("\n🌐 Testing Frontend Integration Points...")
    
    # Check if frontend TTS service file exists
    frontend_tts_path = Path("../../new frontend/src/services/ttsService.js")
    if frontend_tts_path.exists():
        print("✅ Frontend TTS service file exists")
    else:
        print("❌ Frontend TTS service file not found")
        return False
    
    # Check if TTS hooks exist
    frontend_hooks_path = Path("../../new frontend/src/hooks/useTTS.js")
    if frontend_hooks_path.exists():
        print("✅ Frontend TTS hooks file exists")
    else:
        print("❌ Frontend TTS hooks file not found")
        return False
    
    print("✅ Frontend integration files are in place")
    return True

def run_complete_test():
    """Run complete TTS integration test suite"""
    print("=" * 60)
    print("🚀 GURUKUL TTS INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("TTS Service Health", test_tts_service_health),
        ("TTS Generation", test_tts_generation),
        ("Subject Generation TTS", test_subject_generation_tts),
        ("Audio File Listing", test_audio_file_listing),
        ("Frontend Integration", test_frontend_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 40}")
        print(f"Running: {test_name}")
        print(f"{'=' * 40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'=' * 60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<10} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TTS integration is working correctly.")
        print("\n🔧 Next Steps:")
        print("   1. Start all services using Backend/start_all_services.bat")
        print("   2. Start the frontend: cd 'new frontend' && npm start")
        print("   3. Test TTS auto-play in AvatarChatInterface and Subjects page")
        print("   4. Verify audio plays automatically when AI generates responses")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure all backend services are running")
        print("   2. Check TTS service is on port 8007")
        print("   3. Verify pyttsx3 is installed: pip install pyttsx3")
        print("   4. Check system audio/TTS engine availability")
    
    return passed == total

if __name__ == "__main__":
    success = run_complete_test()
    exit(0 if success else 1)
