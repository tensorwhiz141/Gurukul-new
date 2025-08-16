"""
Test script for the dedicated chatbot service
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_chat_flow():
    """Test complete chat flow"""
    print("\n🤖 Testing chat flow...")
    
    user_id = "test-user-123"
    test_message = "Hello, how are you?"
    
    # Step 1: Send message
    print(f"📤 Sending message: {test_message}")
    try:
        response = requests.post(
            f"{BASE_URL}/chatpost",
            json={
                "message": test_message,
                "llm": "grok",
                "type": "chat_message"
            },
            params={"user_id": user_id}
        )
        
        if response.status_code == 200:
            print("✅ Message sent successfully")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed to send message: {response.status_code}")
            print(response.text)
            return
            
    except Exception as e:
        print(f"❌ Error sending message: {e}")
        return
    
    # Step 2: Wait a moment for processing
    print("⏳ Waiting for processing...")
    time.sleep(2)
    
    # Step 3: Get response
    print("📥 Getting AI response...")
    try:
        response = requests.get(
            f"{BASE_URL}/chatbot",
            params={"user_id": user_id}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                print(f"⚠️ Got error: {result['error']}")
            else:
                print("✅ Got AI response")
                print(json.dumps(result, indent=2))
        else:
            print(f"❌ Failed to get response: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error getting response: {e}")

def test_chat_history():
    """Test chat history endpoint"""
    print("\n📚 Testing chat history...")
    
    user_id = "test-user-123"
    
    try:
        response = requests.get(
            f"{BASE_URL}/chat-history",
            params={"user_id": user_id, "limit": 10}
        )
        
        if response.status_code == 200:
            print("✅ Chat history retrieved")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed to get chat history: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error getting chat history: {e}")

def test_multiple_messages():
    """Test multiple messages to ensure no 'No queries yet' error"""
    print("\n🔄 Testing multiple messages...")
    
    user_id = "test-user-multi"
    messages = [
        "Hello!",
        "How are you?",
        "What's the weather like?",
        "Tell me a joke"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n--- Message {i}: {message} ---")
        
        # Send message
        try:
            response = requests.post(
                f"{BASE_URL}/chatpost",
                json={
                    "message": message,
                    "llm": "grok",
                    "type": "chat_message"
                },
                params={"user_id": user_id}
            )
            
            if response.status_code == 200:
                print(f"✅ Message {i} sent")
            else:
                print(f"❌ Message {i} failed: {response.status_code}")
                continue
                
        except Exception as e:
            print(f"❌ Error sending message {i}: {e}")
            continue
        
        # Wait and get response
        time.sleep(1)
        
        try:
            response = requests.get(
                f"{BASE_URL}/chatbot",
                params={"user_id": user_id}
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    print(f"❌ Message {i} got error: {result['error']}")
                else:
                    ai_message = result.get("response", {}).get("message", "No message found")
                    print(f"✅ Message {i} response: {ai_message[:100]}...")
            else:
                print(f"❌ Failed to get response for message {i}")
                
        except Exception as e:
            print(f"❌ Error getting response for message {i}: {e}")

if __name__ == "__main__":
    print("🧪 Testing Dedicated Chatbot Service")
    print("=" * 50)
    
    # Test health
    test_health()
    
    # Test basic chat flow
    test_chat_flow()
    
    # Test chat history
    test_chat_history()
    
    # Test multiple messages
    test_multiple_messages()
    
    print("\n✅ Testing completed!")
