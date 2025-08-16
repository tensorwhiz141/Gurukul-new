#!/usr/bin/env python3
"""
End-to-end test for lesson generation flow
Tests both lesson creation and status polling endpoints
"""

import requests
import json
import time

def test_complete_lesson_flow():
    """Test the complete lesson generation flow"""
    
    print("🧪 Testing Complete Lesson Generation Flow")
    print("=" * 60)
    
    # Step 1: Create lesson
    print("\n📝 Step 1: Creating lesson...")
    
    lesson_data = {
        "subject": "ved",
        "topic": "sound",
        "user_id": "test-user-123",
        "include_wikipedia": True,
        "force_regenerate": True
    }
    
    try:
        create_response = requests.post(
            "http://localhost:8000/lessons",
            json=lesson_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📊 Create Status Code: {create_response.status_code}")
        
        if create_response.status_code != 200:
            print(f"❌ Create failed: {create_response.text}")
            return False
        
        create_result = create_response.json()
        print(f"✅ Lesson creation initiated")
        print(f"📋 Response: {json.dumps(create_result, indent=2)}")
        
        task_id = create_result.get('task_id')
        if not task_id:
            print("❌ No task_id in response")
            return False
        
        print(f"🎯 Task ID: {task_id}")
        
        # Step 2: Poll for status
        print(f"\n🔄 Step 2: Polling for lesson completion...")
        
        max_attempts = 10  # Reduced for testing
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            print(f"📡 Polling attempt {attempts}/{max_attempts}...")
            
            # Wait between attempts
            if attempts > 1:
                time.sleep(2)
            
            try:
                status_response = requests.get(
                    f"http://localhost:8000/lessons/status/{task_id}",
                    timeout=10
                )
                
                print(f"📊 Status Code: {status_response.status_code}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"📋 Status: {status_data.get('status', 'unknown')}")
                    
                    if status_data.get('status') == 'completed':
                        print("✅ Lesson generation completed!")
                        if status_data.get('lesson_data'):
                            print("📚 Lesson data received")
                            lesson_title = status_data['lesson_data'].get('title', 'N/A')
                            print(f"📖 Title: {lesson_title}")
                        return True
                    
                    elif status_data.get('status') == 'failed':
                        print(f"❌ Lesson generation failed: {status_data.get('error_message', 'Unknown error')}")
                        return False
                    
                    elif status_data.get('status') in ['pending', 'in_progress']:
                        print(f"⏳ Status: {status_data.get('status')} - continuing to poll...")
                        continue
                    
                    else:
                        print(f"⚠️  Unknown status: {status_data.get('status')}")
                        continue
                
                elif status_response.status_code == 404:
                    print("❌ 404 - Task not found or endpoint doesn't exist")
                    print(f"Response: {status_response.text}")
                    return False
                
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
                    print(f"Response: {status_response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error during status check: {e}")
                if attempts == max_attempts:
                    return False
                continue
        
        print("⏰ Polling timed out - lesson may still be generating")
        return False
        
    except Exception as e:
        print(f"❌ Error during lesson creation: {e}")
        return False

def test_status_endpoint_directly():
    """Test the status endpoint with a known invalid task ID"""
    
    print("\n🔍 Testing Status Endpoint Directly")
    print("=" * 50)
    
    try:
        response = requests.get(
            "http://localhost:8000/lessons/status/invalid-task-id",
            timeout=5
        )
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Response: {response.text}")
        
        if response.status_code == 404:
            response_data = response.json()
            if "Task" in response_data.get("detail", {}).get("message", ""):
                print("✅ Status endpoint is working correctly (returns 404 for invalid task)")
                return True
        
        print("⚠️  Unexpected response from status endpoint")
        return False
        
    except Exception as e:
        print(f"❌ Error testing status endpoint: {e}")
        return False

def test_service_endpoints():
    """Test all lesson-related endpoints"""
    
    print("\n🔍 Testing All Lesson Service Endpoints")
    print("=" * 50)
    
    endpoints = [
        ("GET", "http://localhost:8000/docs", "API Documentation"),
        ("GET", "http://localhost:8000/lessons/ved/sound", "Get Existing Lesson"),
        ("GET", "http://localhost:8000/lessons/status/test", "Status Endpoint"),
    ]
    
    results = {}
    
    for method, url, description in endpoints:
        try:
            if method == "GET":
                response = requests.get(url, timeout=5)
            
            print(f"📡 {description}: {response.status_code}")
            results[description] = response.status_code
            
        except Exception as e:
            print(f"❌ {description}: Error - {e}")
            results[description] = "Error"
    
    return results

if __name__ == "__main__":
    print("🏫 Lesson Generation End-to-End Test")
    print("=" * 70)
    
    # Test 1: Service endpoints
    print("\n🔧 Phase 1: Testing Service Endpoints")
    endpoint_results = test_service_endpoints()
    
    # Test 2: Status endpoint
    print("\n🔧 Phase 2: Testing Status Endpoint")
    status_working = test_status_endpoint_directly()
    
    # Test 3: Complete flow
    print("\n🔧 Phase 3: Testing Complete Lesson Flow")
    flow_working = test_complete_lesson_flow()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    print(f"🔗 Service Endpoints: {endpoint_results}")
    print(f"📡 Status Endpoint: {'✅ Working' if status_working else '❌ Failed'}")
    print(f"🔄 Complete Flow: {'✅ Working' if flow_working else '❌ Failed'}")
    
    if status_working and flow_working:
        print("\n🎉 SUCCESS! Lesson generation flow is working correctly")
        print("✅ Frontend should now work without 404 errors")
    else:
        print("\n❌ Some issues detected in lesson generation flow")
        
        if not status_working:
            print("💡 Status endpoint issue - check if it exists on port 8000")
        
        if not flow_working:
            print("💡 Complete flow issue - check lesson generation service")
    
    print("\n" + "=" * 70)
    print("Test completed!")
