#!/usr/bin/env python3
"""
BHIV Core API Test Client
Easy way to test all BHIV Core endpoints
"""

import requests
import json
import time
from datetime import datetime

# BHIV Core API base URL
import os
BASE_URL = os.getenv("BHIV_BASE_URL", "http://localhost:8010")

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_response(response, test_name):
    """Print formatted response"""
    print(f"\n📋 {test_name}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS")
        try:
            data = response.json()
            print("📄 Response:")
            print(json.dumps(data, indent=2))
        except:
            print("📄 Response (text):")
            print(response.text)
    else:
        print("❌ FAILED")
        print(f"Error: {response.text}")

def test_health_check():
    """Test health check endpoint"""
    print_header("Health Check Test")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response(response, "Health Check")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Cannot connect to BHIV Core API")
        print("💡 Make sure the API is running: python bhiv_core_api.py")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_classification():
    """Test query classification"""
    print_header("Query Classification Tests")
    
    test_cases = [
        {
            "name": "Educational Query",
            "data": {
                "user_id": "test_student_001",
                "query": "How do I solve quadratic equations?",
                "lang": "en-IN"
            }
        },
        {
            "name": "Wellness Query",
            "data": {
                "user_id": "test_student_002", 
                "query": "I'm feeling very stressed about my exams",
                "lang": "en-IN"
            }
        },
        {
            "name": "Spiritual Query",
            "data": {
                "user_id": "test_student_003",
                "query": "What do the Vedas say about the purpose of life?",
                "lang": "en-IN"
            }
        }
    ]
    
    success_count = 0
    
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/bhiv-core/classify",
                json=test_case["data"],
                headers={"Content-Type": "application/json"}
            )
            print_response(response, test_case["name"])
            
            if response.status_code == 200:
                success_count += 1
                # Show key classification results
                data = response.json()
                print(f"   🎯 Classification: {data.get('classification', 'unknown')}")
                print(f"   🤖 Recommended Agent: {data.get('recommended_agent', 'unknown')}")
                print(f"   📊 Confidence: {data.get('confidence', 0):.2f}")
                
        except Exception as e:
            print(f"❌ {test_case['name']} failed: {e}")
    
    return success_count == len(test_cases)

def test_quick_query():
    """Test quick query processing"""
    print_header("Quick Query Tests")
    
    test_cases = [
        {
            "name": "Simple Educational Query",
            "data": {
                "user_id": "student_123",
                "query": "Explain machine learning in simple terms",
                "voice_enabled": False,
                "lang": "en-IN"
            }
        },
        {
            "name": "Voice-Enabled Query",
            "data": {
                "user_id": "student_456",
                "query": "What is photosynthesis?",
                "voice_enabled": True,
                "lang": "en-IN"
            }
        }
    ]
    
    success_count = 0
    
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/bhiv-core/quick-query",
                json=test_case["data"],
                headers={"Content-Type": "application/json"}
            )
            print_response(response, test_case["name"])
            
            if response.status_code == 200:
                success_count += 1
                # Show key response details
                data = response.json()
                print(f"   🤖 Agent Used: {data.get('agent_used', 'unknown')}")
                print(f"   📊 Confidence: {data.get('confidence_score', 0):.2f}")
                print(f"   ⏱️ Processing Time: {data.get('processing_time_ms', 0):.1f}ms")
                print(f"   🔊 Voice URL: {data.get('voice_response_url', 'None')}")
                
        except Exception as e:
            print(f"❌ {test_case['name']} failed: {e}")
    
    return success_count == len(test_cases)

def test_full_bhiv_processing():
    """Test full BHIV Core processing"""
    print_header("Full BHIV Core Processing Test")
    
    # Complete BHIV request
    bhiv_request = {
        "session_id": f"test-session-{int(time.time())}",
        "timestamp": datetime.now().isoformat(),
        "user_info": {
            "user_id": "advanced_user_001",
            "auth_token": "bhiv-core-key",
            "lang": "en-IN",
            "persona": "student",
            "permissions": ["read", "write"]
        },
        "input": {
            "raw_text": "I want to understand quantum computing and its applications",
            "voice_enabled": True,
            "mode": "chat",
            "context": {
                "course": "Advanced Physics",
                "session_context": {
                    "learning_level": "intermediate"
                }
            }
        },
        "knowledgebase_query": {
            "use_vector": True,
            "vector_type": "qdrant",
            "top_k": 5,
            "include_sources": True,
            "filters": {
                "lang": "en",
                "curriculum_tag": ["physics", "quantum"],
                "difficulty_level": "intermediate"
            },
            "similarity_threshold": 0.7
        },
        "llm_request": {
            "model": "gemini",
            "prompt_style": "instruction",
            "max_tokens": 1024,
            "temperature": 0.3,
            "response_language": "en"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/bhiv-core/process",
            json=bhiv_request,
            headers={"Content-Type": "application/json"}
        )
        print_response(response, "Full BHIV Processing")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🎯 Key Results:")
            print(f"   Session ID: {data.get('session_id', 'unknown')}")
            print(f"   Agent: {data.get('agent_directives', {}).get('agent_name', 'unknown')}")
            print(f"   Classification: {data.get('agent_directives', {}).get('classification', 'unknown')}")
            print(f"   Response Length: {len(data.get('output', {}).get('response_text', ''))}")
            print(f"   Processing Time: {data.get('logs', {}).get('processing_time_ms', 0):.1f}ms")
            print(f"   Reward Score: {data.get('logs', {}).get('reward_score', 0):.2f}")
            return True
            
    except Exception as e:
        print(f"❌ Full BHIV processing failed: {e}")
    
    return False

def test_session_management():
    """Test session management"""
    print_header("Session Management Test")
    
    # First, create a session by doing a quick query
    session_response = requests.post(
        f"{BASE_URL}/api/v1/bhiv-core/quick-query",
        json={
            "user_id": "session_test_user",
            "query": "Test session creation",
            "voice_enabled": False
        }
    )
    
    if session_response.status_code == 200:
        session_data = session_response.json()
        session_id = session_data.get('session_id')
        
        if session_id:
            # Now get session info
            try:
                response = requests.get(f"{BASE_URL}/api/v1/bhiv-core/session/{session_id}")
                print_response(response, "Session Information")
                return response.status_code == 200
            except Exception as e:
                print(f"❌ Session info failed: {e}")
    
    return False

def test_system_stats():
    """Test system statistics"""
    print_header("System Statistics Test")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/bhiv-core/stats")
        print_response(response, "System Statistics")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 System Overview:")
            print(f"   Total Sessions: {data.get('total_sessions', 0)}")
            print(f"   Active Sessions: {data.get('active_sessions', 0)}")
            print(f"   Components Status: {data.get('components_status', {})}")
            return True
            
    except Exception as e:
        print(f"❌ System stats failed: {e}")
    
    return False

def main():
    """Run all tests"""
    print("🧠 BHIV Core API Test Suite")
    print(f"Testing API at: {BASE_URL}")
    print(f"Test started at: {datetime.now().isoformat()}")
    
    tests = [
        ("Health Check", test_health_check),
        ("Query Classification", test_classification),
        ("Quick Query Processing", test_quick_query),
        ("Full BHIV Processing", test_full_bhiv_processing),
        ("Session Management", test_session_management),
        ("System Statistics", test_system_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print_header("Test Results Summary")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BHIV Core is working perfectly!")
    elif passed >= total * 0.7:
        print("✅ Most tests passed! BHIV Core is mostly functional.")
    else:
        print("⚠️ Many tests failed. Check the BHIV Core setup.")
    
    print(f"\n💡 To explore more, visit: {BASE_URL}/docs")

if __name__ == "__main__":
    main()
