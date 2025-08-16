#!/usr/bin/env python3
"""
Test script for Lesson Generator endpoint
"""

import requests
import json

def test_lesson_generator():
    """Test the /lessons endpoint on port 8000"""
    
    # Test data matching the frontend format
    test_data = {
        "subject": "ved",
        "topic": "sound",
        "user_id": "test-user-123",
        "include_wikipedia": True,
        "force_regenerate": False
    }
    
    print("🧪 Testing Lesson Generator Endpoint")
    print("=" * 50)
    print(f"📡 URL: http://localhost:8000/lessons")
    print(f"📊 Test Data: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8000/lessons",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Lesson Generator is working")
            print(f"📋 Response: {json.dumps(result, indent=2)}")
            
            if "task_id" in result:
                print(f"🎯 Task ID: {result['task_id']}")
                return True
            else:
                print("⚠️  No task_id in response")
                return False
                
        elif response.status_code == 404:
            print("❌ 404 Not Found - Endpoint doesn't exist")
            print("💡 Check if the Lesson Generator service is running correctly")
            
        elif response.status_code == 422:
            print("❌ 422 Validation Error - Invalid request format")
            print(f"📄 Response: {response.text}")
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error - Lesson Generator service is not running")
        print("💡 Start the service: cd Backend/pipline-24-master && python app.py")
        return False
        
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_lesson_service_status():
    """Test if the Lesson Generator service is running"""
    
    print("\n🔍 Checking Lesson Generator Service Status")
    print("=" * 50)
    
    try:
        # Test health/docs endpoint
        response = requests.get("http://localhost:8000/docs", timeout=5)
        
        if response.status_code == 200:
            print("✅ Lesson Generator service is running")
            print("📚 API documentation is accessible at http://localhost:8000/docs")
            return True
        else:
            print(f"⚠️  Service responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Lesson Generator service is not running on port 8000")
        return False
        
    except Exception as e:
        print(f"❌ Error checking service: {e}")
        return False

def test_get_lesson_endpoint():
    """Test the GET /lessons/{subject}/{topic} endpoint"""
    
    print("\n🔍 Testing GET Lesson Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:8000/lessons/ved/sound", timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ GET endpoint working - lesson exists")
            print(f"📋 Lesson Title: {result.get('title', 'N/A')}")
            return True
        elif response.status_code == 404:
            print("📝 404 - Lesson not found (this is normal for new lessons)")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing GET endpoint: {e}")
        return False

if __name__ == "__main__":
    print("📚 Lesson Generator Endpoint Test")
    print("=" * 60)
    
    # First check if service is running
    service_running = test_lesson_service_status()
    
    if service_running:
        # Test GET endpoint
        get_working = test_get_lesson_endpoint()
        
        # Test POST endpoint
        post_working = test_lesson_generator()
        
        if post_working:
            print("\n🎉 SUCCESS! Lesson Generator is working correctly")
            print("✅ Frontend should now be able to connect to port 8000")
        else:
            print("\n❌ Lesson Generator POST endpoint has issues")
    else:
        print("\n❌ Lesson Generator service is not running")
        print("💡 Start it with: cd Backend/pipline-24-master && python app.py")
    
    print("\n" + "=" * 60)
    print("Test completed!")
