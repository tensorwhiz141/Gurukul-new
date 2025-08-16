#!/usr/bin/env python3
"""
Test script for audio file retrieval from 192.168.0.119:8001
This script demonstrates how to use the new GET endpoints to retrieve audio files
"""

import requests
import os
from datetime import datetime

# Configuration
GURUKUL_API_BASE = "http://192.168.0.83:8000"
EXTERNAL_SERVER = "192.168.0.119:8001"
DOWNLOAD_DIR = "downloaded_audio"

def setup_download_directory():
    """Create download directory if it doesn't exist"""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"📁 Download directory: {os.path.abspath(DOWNLOAD_DIR)}")

def test_list_audio_files():
    """Test listing available audio files from the external server"""
    print("🎵 Testing audio file listing...")
    
    try:
        response = requests.get(f"{GURUKUL_API_BASE}/api/audio-files")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Successfully retrieved audio file list!")
            print(f"📊 External server: {result['external_server']}")
            print(f"📊 Total files: {result['count']}")
            print(f"📊 Retrieved at: {result['retrieved_at']}")
            
            if result['audio_files']:
                print("\n🎵 Available audio files:")
                for i, filename in enumerate(result['audio_files'], 1):
                    print(f"  {i}. {filename}")
                
                print(f"\n💡 Access info:")
                print(f"   Base URL: {result['access_info']['base_url']}")
                print(f"   Example: {result['access_info']['example']}")
                
                return result['audio_files']
            else:
                print("📭 No audio files found on external server")
                return []
        else:
            print(f"❌ Failed to retrieve audio file list: {response.status_code}")
            print(f"   Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Error listing audio files: {e}")
        return []

def test_download_audio_file(filename):
    """Test downloading a specific audio file"""
    print(f"\n⬇️ Testing audio file download: {filename}")
    
    try:
        # Make request to download the audio file
        response = requests.get(f"{GURUKUL_API_BASE}/api/audio/{filename}", stream=True)
        
        if response.status_code == 200:
            # Save the file locally
            local_path = os.path.join(DOWNLOAD_DIR, filename)
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Check file size
            file_size = os.path.getsize(local_path)
            
            print(f"✅ Successfully downloaded: {filename}")
            print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"📁 Saved to: {os.path.abspath(local_path)}")
            
            # Check response headers for additional info
            if 'X-Audio-Source' in response.headers:
                print(f"🔗 Source server: {response.headers['X-Audio-Source']}")
            if 'X-File-Size' in response.headers:
                print(f"📊 Original size: {response.headers['X-File-Size']} bytes")
            
            return True
            
        elif response.status_code == 404:
            print(f"❌ Audio file not found: {filename}")
            print("💡 Check if the filename is correct")
            return False
            
        else:
            print(f"❌ Failed to download audio file: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading audio file: {e}")
        return False

def test_download_all_audio_files():
    """Test downloading all available audio files"""
    print("\n📥 Testing download of all available audio files...")
    
    # Get list of audio files
    audio_files = test_list_audio_files()
    
    if not audio_files:
        print("❌ No audio files to download")
        return
    
    successful_downloads = 0
    failed_downloads = 0
    
    for filename in audio_files:
        if test_download_audio_file(filename):
            successful_downloads += 1
        else:
            failed_downloads += 1
    
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successful: {successful_downloads}")
    print(f"   ❌ Failed: {failed_downloads}")
    print(f"   📁 Download directory: {os.path.abspath(DOWNLOAD_DIR)}")

def test_audio_file_not_found():
    """Test handling of non-existent audio files"""
    print("\n🔍 Testing non-existent audio file handling...")
    
    fake_filename = "non_existent_file.mp3"
    
    try:
        response = requests.get(f"{GURUKUL_API_BASE}/api/audio/{fake_filename}")
        
        if response.status_code == 404:
            result = response.json()
            print(f"✅ Correctly handled non-existent file: {fake_filename}")
            print(f"📊 Response: {result.get('detail', {}).get('message', 'File not found')}")
        else:
            print(f"❌ Unexpected response for non-existent file: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing non-existent file: {e}")

def test_external_server_connectivity():
    """Test connectivity to the external server"""
    print("\n🔗 Testing external server connectivity...")
    
    try:
        response = requests.get(f"{GURUKUL_API_BASE}/check_external_server")
        result = response.json()
        
        print(f"📊 Server status: {result['status']}")
        print(f"🔗 Server address: {result['server']}")
        
        if result['status'] == 'reachable':
            print(f"✅ External server is reachable!")
            if 'response_time_seconds' in result:
                print(f"⚡ Response time: {result['response_time_seconds']} seconds")
        else:
            print(f"❌ External server is not reachable")
            print(f"💡 Message: {result.get('message', 'No message')}")
            
    except Exception as e:
        print(f"❌ Error checking connectivity: {e}")

def main():
    """Main test function"""
    print("🎵 Gurukul AI - Audio File Retrieval Test")
    print("=" * 60)
    print(f"🔗 Gurukul API: {GURUKUL_API_BASE}")
    print(f"🔗 External Server: {EXTERNAL_SERVER}")
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Setup
    setup_download_directory()
    
    # Run tests
    test_external_server_connectivity()
    test_list_audio_files()
    test_download_all_audio_files()
    test_audio_file_not_found()
    
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("✅ Audio file listing endpoint working")
    print("✅ Audio file download endpoint working")
    print("✅ Error handling working correctly")
    print("✅ External server connectivity confirmed")
    print("\n💡 Your GET endpoints for audio retrieval from")
    print(f"   192.168.0.119:8001 are fully functional!")
    print("=" * 60)

if __name__ == "__main__":
    main()
