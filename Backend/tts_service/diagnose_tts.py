#!/usr/bin/env python3
"""
TTS Diagnostic Script
Comprehensive testing of TTS functionality and audio system
"""

import pyttsx3
import os
import sys
import time
import subprocess
import platform

def test_system_audio():
    """Test if system audio is working"""
    print("🔊 Testing System Audio...")
    
    try:
        if platform.system() == "Windows":
            # Try to play a system sound
            import winsound
            print("   Playing system beep...")
            winsound.Beep(1000, 500)  # 1000Hz for 500ms
            print("   ✅ System beep played (if you heard a beep, audio is working)")
            return True
        else:
            print("   ⚠️ Non-Windows system - skipping system beep test")
            return True
    except Exception as e:
        print(f"   ❌ System audio test failed: {e}")
        return False

def test_pyttsx3_voices():
    """Test pyttsx3 voice availability and configuration"""
    print("\n🎤 Testing pyttsx3 Voices...")
    
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        if not voices:
            print("   ❌ No voices available")
            return False
        
        print(f"   ✅ Found {len(voices)} voices:")
        for i, voice in enumerate(voices):
            print(f"      {i+1}. {voice.name}")
            print(f"         ID: {voice.id}")
            print(f"         Languages: {getattr(voice, 'languages', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"   ❌ Voice test failed: {e}")
        return False

def test_pyttsx3_properties():
    """Test pyttsx3 engine properties"""
    print("\n⚙️ Testing pyttsx3 Properties...")
    
    try:
        engine = pyttsx3.init()
        
        # Get current properties
        rate = engine.getProperty('rate')
        volume = engine.getProperty('volume')
        voice = engine.getProperty('voice')
        
        print(f"   Current Rate: {rate} words/minute")
        print(f"   Current Volume: {volume} (0.0-1.0)")
        print(f"   Current Voice: {voice}")
        
        # Test setting properties
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
            print(f"   ✅ Set voice to: {voices[0].name}")
        
        return True
    except Exception as e:
        print(f"   ❌ Properties test failed: {e}")
        return False

def test_pyttsx3_direct_speech():
    """Test direct speech output"""
    print("\n🗣️ Testing Direct Speech...")
    
    try:
        engine = pyttsx3.init()
        
        # Configure for best results
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        
        test_messages = [
            "Testing direct speech output.",
            "This is a test of the text to speech system.",
            "If you can hear this, the TTS engine is working correctly."
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"   Test {i}: Speaking '{message}'")
            engine.say(message)
            engine.runAndWait()
            time.sleep(1)
        
        print("   ✅ Direct speech test completed")
        return True
    except Exception as e:
        print(f"   ❌ Direct speech test failed: {e}")
        return False

def test_file_generation():
    """Test TTS file generation"""
    print("\n💾 Testing File Generation...")
    
    try:
        engine = pyttsx3.init()
        
        # Configure engine
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        
        # Generate test file
        test_file = "test_tts_output.wav"
        test_text = "This is a test file generation. If you can play this file, TTS file generation is working."
        
        print(f"   Generating file: {test_file}")
        engine.save_to_file(test_text, test_file)
        engine.runAndWait()
        
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            print(f"   ✅ File generated: {test_file} ({file_size} bytes)")
            
            # Try to play the file
            if platform.system() == "Windows":
                print("   Attempting to play generated file...")
                try:
                    os.startfile(test_file)
                    print("   ✅ File opened with default player")
                except Exception as e:
                    print(f"   ⚠️ Could not auto-play file: {e}")
            
            return True
        else:
            print("   ❌ File was not generated")
            return False
            
    except Exception as e:
        print(f"   ❌ File generation test failed: {e}")
        return False

def test_audio_devices():
    """Test audio device availability"""
    print("\n🔌 Testing Audio Devices...")
    
    try:
        if platform.system() == "Windows":
            # Try to get audio device info using PowerShell
            cmd = 'powershell "Get-WmiObject -Class Win32_SoundDevice | Select-Object Name, Status"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   Audio devices found:")
                lines = result.stdout.strip().split('\n')
                for line in lines[2:]:  # Skip header lines
                    if line.strip():
                        print(f"      {line.strip()}")
                print("   ✅ Audio devices detected")
                return True
            else:
                print("   ⚠️ Could not query audio devices")
                return False
        else:
            print("   ⚠️ Non-Windows system - skipping device check")
            return True
            
    except Exception as e:
        print(f"   ❌ Audio device test failed: {e}")
        return False

def test_windows_speech_api():
    """Test Windows Speech API directly"""
    print("\n🪟 Testing Windows Speech API...")
    
    try:
        if platform.system() == "Windows":
            import win32com.client
            
            # Create SAPI voice object
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            
            # Get available voices
            voices = voice.GetVoices()
            print(f"   Found {voices.Count} SAPI voices")
            
            # Test speech
            print("   Testing SAPI speech...")
            voice.Speak("Testing Windows Speech API directly. This is a SAPI test.")
            
            print("   ✅ Windows Speech API test completed")
            return True
        else:
            print("   ⚠️ Not a Windows system - skipping SAPI test")
            return True
            
    except ImportError:
        print("   ⚠️ pywin32 not available - skipping SAPI test")
        return True
    except Exception as e:
        print(f"   ❌ Windows Speech API test failed: {e}")
        return False

def main():
    """Run comprehensive TTS diagnostics"""
    print("=" * 60)
    print("🔍 COMPREHENSIVE TTS DIAGNOSTIC")
    print("=" * 60)
    
    tests = [
        ("System Audio", test_system_audio),
        ("pyttsx3 Voices", test_pyttsx3_voices),
        ("pyttsx3 Properties", test_pyttsx3_properties),
        ("Direct Speech", test_pyttsx3_direct_speech),
        ("File Generation", test_file_generation),
        ("Audio Devices", test_audio_devices),
        ("Windows Speech API", test_windows_speech_api)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC RESULTS")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<10} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{len(results)} tests passed")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if passed == len(results):
        print("🎉 All tests passed! TTS should be working.")
        print("If you still can't hear sound in the browser:")
        print("   1. Check browser volume settings")
        print("   2. Enable autoplay in browser")
        print("   3. Click on the webpage first")
    else:
        print("⚠️ Some tests failed. Try these solutions:")
        print("   1. Check system volume and speakers")
        print("   2. Restart audio service: services.msc → Windows Audio")
        print("   3. Try different audio output device")
        print("   4. Install/reinstall audio drivers")
        print("   5. Run as administrator")
    
    print("\n🔧 Next steps:")
    print("   1. If system beep worked, audio hardware is OK")
    print("   2. If file generation worked, try playing the generated file")
    print("   3. Check Windows Sound settings")
    print("   4. Try different TTS voice")

if __name__ == "__main__":
    main()
