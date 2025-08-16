"""
Test frontend integration with dedicated chatbot service
This simulates the exact flow that the frontend uses
"""

import requests
import json
import time

# Configuration
CHATBOT_BASE_URL = "http://localhost:8001"
FRONTEND_BASE_URL = "http://localhost:5174"

def test_frontend_flow():
    """Test the exact flow that the frontend uses"""
    print("🧪 Testing Frontend Integration Flow")
    print("=" * 50)
    
    user_id = "frontend-test-user"
    test_messages = [
        "Hello!",
        "How are you doing?",
        "What can you help me with?",
        "Tell me about artificial intelligence"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Testing Message {i}: '{message}' ---")
        
        # Step 1: Send message (like frontend does)
        print(f"📤 Step 1: Sending message to /chatpost")
        try:
            response = requests.post(
                f"{CHATBOT_BASE_URL}/chatpost",
                params={"user_id": user_id},
                json={
                    "message": message,
                    "llm": "grok",
                    "type": "chat_message"
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Message sent successfully")
                print(f"   Message ID: {result['data']['id']}")
            else:
                print(f"❌ Failed to send message: {response.status_code}")
                print(f"   Response: {response.text}")
                continue
                
        except Exception as e:
            print(f"❌ Error sending message: {e}")
            continue
        
        # Step 2: Wait (like frontend does)
        print(f"⏳ Step 2: Waiting for processing (1 second)...")
        time.sleep(1)
        
        # Step 3: Get response (like frontend does)
        print(f"📥 Step 3: Getting AI response from /chatbot")
        try:
            response = requests.get(
                f"{CHATBOT_BASE_URL}/chatbot",
                params={"user_id": user_id}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "error" in result:
                    if result["error"] == "No queries yet":
                        print(f"⚠️ Got 'No queries yet' - retrying after 2 seconds...")
                        time.sleep(2)
                        
                        # Retry once
                        retry_response = requests.get(
                            f"{CHATBOT_BASE_URL}/chatbot",
                            params={"user_id": user_id}
                        )
                        
                        if retry_response.status_code == 200:
                            retry_result = retry_response.json()
                            if "error" in retry_result:
                                print(f"❌ Still got error after retry: {retry_result['error']}")
                            else:
                                ai_message = retry_result.get("response", {}).get("message", "No message found")
                                print(f"✅ Got AI response after retry: {ai_message[:100]}...")
                        else:
                            print(f"❌ Retry failed: {retry_response.status_code}")
                    else:
                        print(f"❌ Got error: {result['error']}")
                else:
                    ai_message = result.get("response", {}).get("message", "No message found")
                    print(f"✅ Got AI response: {ai_message[:100]}...")
                    
            else:
                print(f"❌ Failed to get response: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error getting response: {e}")
        
        # Small delay between messages
        time.sleep(0.5)
    
    print(f"\n📚 Getting chat history...")
    try:
        response = requests.get(
            f"{CHATBOT_BASE_URL}/chat-history",
            params={"user_id": user_id, "limit": 10}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat history retrieved: {result['total']} messages")
            for msg in result['messages'][:3]:  # Show first 3
                print(f"   User: {msg['user_message']}")
                print(f"   AI: {msg['ai_response'][:80]}...")
                print()
        else:
            print(f"❌ Failed to get chat history: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting chat history: {e}")

def test_error_scenarios():
    """Test error scenarios that frontend might encounter"""
    print(f"\n🚨 Testing Error Scenarios")
    print("=" * 30)
    
    # Test 1: Get response without sending message
    print(f"Test 1: Getting response without sending message")
    try:
        response = requests.get(
            f"{CHATBOT_BASE_URL}/chatbot",
            params={"user_id": "no-messages-user"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "error" in result and result["error"] == "No queries yet":
                print(f"✅ Correctly got 'No queries yet' error")
            else:
                print(f"❌ Unexpected response: {result}")
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error in test: {e}")
    
    # Test 2: Invalid message format
    print(f"\nTest 2: Sending invalid message format")
    try:
        response = requests.post(
            f"{CHATBOT_BASE_URL}/chatpost",
            params={"user_id": "error-test-user"},
            json={"invalid": "format"},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 422:
            print(f"✅ Correctly rejected invalid format with 422")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error in test: {e}")

def check_service_health():
    """Check if all required services are running"""
    print(f"🔍 Checking Service Health")
    print("=" * 25)
    
    # Check chatbot service
    try:
        response = requests.get(f"{CHATBOT_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chatbot Service: {result['status']}")
            print(f"   MongoDB: {result['mongodb']}")
            print(f"   LLM Providers: {result['llm_providers']}")
        else:
            print(f"❌ Chatbot Service: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Chatbot Service: {e}")
    
    # Check frontend (basic check)
    try:
        response = requests.get(FRONTEND_BASE_URL, timeout=5)
        if response.status_code == 200:
            print(f"✅ Frontend: Running on {FRONTEND_BASE_URL}")
        else:
            print(f"❌ Frontend: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend: {e}")

if __name__ == "__main__":
    print("🧪 Frontend Integration Test Suite")
    print("=" * 50)
    
    # Check services first
    check_service_health()
    
    # Test main flow
    test_frontend_flow()
    
    # Test error scenarios
    test_error_scenarios()
    
    print(f"\n✅ Integration testing completed!")
    print(f"\n💡 Next steps:")
    print(f"   1. Open browser: {FRONTEND_BASE_URL}")
    print(f"   2. Navigate to Chatbot page")
    print(f"   3. Test sending messages")
    print(f"   4. Verify no 'No queries yet' errors on second message")
