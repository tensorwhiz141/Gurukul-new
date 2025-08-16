#!/usr/bin/env python3
"""
Complete TTS Integration Test Script
This script demonstrates the full workflow of automatic TTS generation during lesson creation
"""

import requests
import json
import time
import os
from datetime import datetime

# Configuration
GURUKUL_API_BASE = "http://192.168.0.83:8000"
TTS_SERVER = "192.168.0.119:8001"
DOWNLOAD_DIR = "complete_lesson_audio"

def setup_test_environment():
    """Setup test environment"""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"📁 Test directory: {os.path.abspath(DOWNLOAD_DIR)}")

def test_lesson_generation_with_automatic_tts():
    """Test complete lesson generation with automatic TTS"""
    print("\n🎓 Testing Lesson Generation with Automatic TTS...")
    
    # Generate a new lesson
    lesson_request = {
        "subject": "science",
        "topic": "solar_system",
        "user_id": "tts-test-user",
        "include_wikipedia": True,
        "force_regenerate": True
    }
    
    try:
        print("📝 Starting lesson generation...")
        response = requests.post(
            f"{GURUKUL_API_BASE}/lessons",
            json=lesson_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            task_id = result['task_id']
            
            print(f"✅ Lesson generation started!")
            print(f"📋 Task ID: {task_id}")
            print(f"📚 Subject: {lesson_request['subject']}")
            print(f"📖 Topic: {lesson_request['topic']}")
            
            # Wait for lesson generation to complete
            print("\n⏳ Waiting for lesson generation and automatic TTS...")
            
            max_wait_time = 120  # 2 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                status_response = requests.get(f"{GURUKUL_API_BASE}/lessons/status/{task_id}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    
                    print(f"📊 Status: {status_data['status']}")
                    print(f"💬 Message: {status_data['progress_message']}")
                    
                    if status_data['status'] == 'completed':
                        lesson_data = status_data.get('lesson_data', {})
                        
                        print(f"\n🎉 Lesson generation completed!")
                        print(f"📚 Title: {lesson_data.get('title', 'Unknown')}")
                        
                        # Check if TTS was automatically generated
                        if lesson_data.get('audio_available'):
                            print(f"🎵 Automatic TTS: ✅ SUCCESS")
                            print(f"🎵 Audio filename: {lesson_data.get('audio_filename', 'Unknown')}")
                            print(f"🎵 Audio URL: {lesson_data.get('audio_url', 'Unknown')}")
                            
                            # Test downloading the audio
                            return test_download_lesson_audio(lesson_data.get('audio_filename'))
                        else:
                            print(f"❌ Automatic TTS: FAILED")
                            tts_error = lesson_data.get('tts_error', 'Unknown error')
                            print(f"❌ TTS Error: {tts_error}")
                            return False
                    
                    elif status_data['status'] == 'failed':
                        print(f"❌ Lesson generation failed: {status_data.get('error_message', 'Unknown error')}")
                        return False
                
                time.sleep(5)  # Wait 5 seconds before checking again
            
            print(f"⏰ Timeout waiting for lesson generation")
            return False
            
        else:
            print(f"❌ Failed to start lesson generation: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during lesson generation: {e}")
        return False

def test_download_lesson_audio(audio_filename):
    """Test downloading the automatically generated lesson audio"""
    if not audio_filename:
        print("❌ No audio filename provided")
        return False
    
    print(f"\n⬇️ Testing audio download: {audio_filename}")
    
    try:
        # Download the audio file
        audio_url = f"{GURUKUL_API_BASE}/api/audio/{audio_filename}"
        local_path = os.path.join(DOWNLOAD_DIR, f"lesson_{audio_filename}")
        
        response = requests.get(audio_url, stream=True)
        
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(local_path)
            
            print(f"✅ Audio downloaded successfully!")
            print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"📁 Saved to: {os.path.abspath(local_path)}")
            
            return True
        else:
            print(f"❌ Failed to download audio: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading audio: {e}")
        return False

def test_manual_tts_generation():
    """Test manual TTS generation for custom text"""
    print("\n🎤 Testing Manual TTS Generation...")
    
    test_text = """
    Welcome to our comprehensive lesson on the Solar System. 
    Let us begin with a sacred verse: सूर्यो वै जगतः प्राणः. 
    The meaning of this verse is: The Sun is indeed the life force of the universe.
    Now, let me explain this concept in detail about our magnificent solar system.
    """
    
    tts_request = {
        "text": test_text.strip(),
        "user_id": "manual-tts-user",
        "description": "Manual TTS test for solar system introduction"
    }
    
    try:
        response = requests.post(
            f"{GURUKUL_API_BASE}/tts/generate",
            json=tts_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Manual TTS generation successful!")
            print(f"📊 Text length: {result['request_info']['text_length']} characters")
            print(f"📝 Preview: {result['request_info']['text_preview']}")
            
            tts_result = result.get('tts_result', {})
            if tts_result.get('status') == 'success':
                audio_info = tts_result.get('audio_info', {})
                audio_filename = audio_info.get('filename')
                
                print(f"🎵 Audio file: {audio_filename}")
                print(f"⚡ Generation time: {tts_result.get('tts_service', {}).get('response_time_seconds')} seconds")
                
                # Test downloading the manual TTS audio
                if audio_filename:
                    return test_download_lesson_audio(audio_filename)
            else:
                print(f"❌ TTS generation failed: {tts_result.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ Manual TTS request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during manual TTS generation: {e}")
        return False

def test_audio_file_listing():
    """Test listing all available audio files"""
    print("\n📋 Testing Audio File Listing...")
    
    try:
        response = requests.get(f"{GURUKUL_API_BASE}/api/audio-files")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Audio file listing successful!")
            print(f"🔗 External server: {result['external_server']}")
            print(f"📊 Total files: {result['count']}")
            
            if result['audio_files']:
                print(f"\n🎵 Available audio files:")
                for i, filename in enumerate(result['audio_files'], 1):
                    print(f"  {i}. {filename}")
                
                print(f"\n💡 Access info:")
                print(f"   Base URL: {result['access_info']['base_url']}")
                print(f"   Example: {result['access_info']['example']}")
                
                return True
            else:
                print("📭 No audio files found")
                return False
        else:
            print(f"❌ Failed to list audio files: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error listing audio files: {e}")
        return False

def test_tts_server_connectivity():
    """Test connectivity to TTS server"""
    print("\n🔗 Testing TTS Server Connectivity...")
    
    try:
        response = requests.get(f"{GURUKUL_API_BASE}/check_external_server")
        result = response.json()
        
        print(f"📊 Server status: {result['status']}")
        print(f"🔗 Server address: {result['server']}")
        
        if result['status'] == 'reachable':
            print(f"✅ TTS server is reachable!")
            if 'response_time_seconds' in result:
                print(f"⚡ Response time: {result['response_time_seconds']} seconds")
            return True
        else:
            print(f"❌ TTS server is not reachable")
            print(f"💡 Message: {result.get('message', 'No message')}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking TTS server: {e}")
        return False

def main():
    """Main test function"""
    print("🎵 Gurukul AI - Complete TTS Integration Test")
    print("=" * 70)
    print(f"🔗 Gurukul API: {GURUKUL_API_BASE}")
    print(f"🔗 TTS Server: {TTS_SERVER}")
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Setup
    setup_test_environment()
    
    # Test results
    results = {
        "tts_connectivity": False,
        "audio_listing": False,
        "manual_tts": False,
        "automatic_tts": False
    }
    
    # Run tests
    results["tts_connectivity"] = test_tts_server_connectivity()
    results["audio_listing"] = test_audio_file_listing()
    results["manual_tts"] = test_manual_tts_generation()
    results["automatic_tts"] = test_lesson_generation_with_automatic_tts()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 Complete TTS Integration Test Summary:")
    print("=" * 70)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        test_display = test_name.replace("_", " ").title()
        print(f"{test_display:.<50} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall Result: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Your TTS integration is fully functional!")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
