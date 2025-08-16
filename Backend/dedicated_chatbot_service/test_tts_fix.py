#!/usr/bin/env python3
"""
Test TTS Fix - Verify text_to_speech_stream function
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_tts_streaming_function():
    """Test the text_to_speech_stream function directly"""
    print("🔍 Testing TTS Streaming Function...")
    
    try:
        # Import the function from chatbot_api
        from chatbot_api import text_to_speech_stream
        print("✅ text_to_speech_stream import: SUCCESS")
        
        # Test with sample text
        test_text = "Hello, this is a test of the TTS streaming function."
        print(f"🎵 Testing with text: '{test_text}'")
        
        # Generate audio stream
        audio_data = text_to_speech_stream(test_text)
        
        if audio_data:
            print(f"✅ TTS streaming: SUCCESS (generated {len(audio_data)} bytes)")
            return True
        else:
            print("❌ TTS streaming: FAILED (no audio data returned)")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ TTS streaming failed: {e}")
        return False

def test_tts_dependencies():
    """Test TTS dependencies"""
    print("\n🔍 Testing TTS Dependencies...")
    
    dependencies = ["gtts", "io"]
    all_good = True
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}: OK")
        except ImportError as e:
            print(f"❌ {dep}: FAILED - {e}")
            all_good = False
    
    return all_good

def test_api_endpoint():
    """Test the TTS API endpoint"""
    print("\n🔍 Testing TTS API Endpoint...")
    
    try:
        import requests
        
        # Test the streaming TTS endpoint
        url = "http://localhost:8001/tts/stream"
        data = {"text": "Hello, this is a test."}
        
        print(f"🌐 Testing endpoint: {url}")
        
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ TTS API endpoint: SUCCESS (status {response.status_code})")
            print(f"📊 Response size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ TTS API endpoint: FAILED (status {response.status_code})")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  TTS API endpoint: Service not running (expected if not started)")
        return None  # Not a failure, just not running
    except Exception as e:
        print(f"❌ TTS API endpoint test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 TTS STREAMING FIX VERIFICATION")
    print("=" * 45)
    
    # Test dependencies
    deps_ok = test_tts_dependencies()
    
    # Test streaming function
    function_ok = test_tts_streaming_function()
    
    # Test API endpoint (if service is running)
    api_ok = test_api_endpoint()
    
    # Summary
    print("\n" + "=" * 45)
    print("📋 TEST SUMMARY:")
    print(f"  TTS Dependencies: {'✅ OK' if deps_ok else '❌ FAILED'}")
    print(f"  TTS Function: {'✅ OK' if function_ok else '❌ FAILED'}")
    if api_ok is not None:
        print(f"  TTS API Endpoint: {'✅ OK' if api_ok else '❌ FAILED'}")
    else:
        print(f"  TTS API Endpoint: ⚠️  Service not running")
    
    if deps_ok and function_ok:
        print("\n🎉 TTS STREAMING FIX SUCCESSFUL!")
        print("✅ text_to_speech_stream function is working")
        print("✅ No more 'name 'text_to_speech_stream' is not defined' errors")
        print("\n🚀 Next Steps:")
        print("  1. Restart the Dedicated Chatbot Service")
        print("  2. Test TTS functionality in the frontend")
        print("  3. Verify streaming audio works properly")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("💡 Check the error messages above")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        input("\nPress Enter to exit...")
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
