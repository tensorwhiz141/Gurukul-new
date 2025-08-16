#!/usr/bin/env python3
"""
Test script to verify Ollama wellness integration
"""

import requests
import json
import time

def test_wellness_api():
    """Test the wellness API with Ollama integration"""
    
    base_url = "http://localhost:8007"
    
    # Test data
    test_cases = [
        {
            "name": "Basic wellness query",
            "data": {
                "query": "I'm feeling stressed about work and need some guidance",
                "user_id": "test_user"
            }
        },
        {
            "name": "Wellness with context",
            "data": {
                "query": "I'm having trouble sleeping and feeling anxious",
                "user_id": "test_user_2",
                "mood_score": 3.0,
                "stress_level": 9.0
            }
        }
    ]
    
    print("🧪 Testing Ollama Wellness Integration")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ {test_case['name']}")
        print("-" * 30)
        
        try:
            # Test POST /wellness endpoint
            print("📤 Sending request to /wellness...")
            start_time = time.time()
            
            response = requests.post(
                f"{base_url}/wellness",
                json=test_case["data"],
                timeout=120  # 2 minutes timeout for Ollama
            )
            
            response_time = time.time() - start_time
            
            print(f"⏱️  Response time: {response_time:.2f}s")
            print(f"📊 Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Success!")
                print(f"🔍 Query: {data.get('query', 'N/A')}")
                print(f"📝 Response length: {len(data.get('response', ''))} chars")
                print(f"🏷️  Endpoint: {data.get('endpoint', 'N/A')}")
                
                # Show first 200 chars of response
                response_text = data.get('response', '')
                if response_text:
                    print(f"💬 Response preview: {response_text[:200]}...")
                    
                    # Check if it looks like Ollama or fallback
                    if 'Thank you for reaching out' in response_text and 'gentle suggestions' in response_text:
                        print("⚠️  WARNING: This looks like a hardcoded fallback response!")
                    else:
                        print("✅ Response appears to be LLM-generated (likely Ollama)")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"📄 Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out (this might be normal for Ollama)")
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - is the server running on port 8007?")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test the enhanced endpoint
    print(f"\n3️⃣ Enhanced wellness endpoint")
    print("-" * 30)
    
    try:
        enhanced_data = {
            "query": "I need help managing work-life balance",
            "user_id": "test_user_3",
            "mood_score": 5.0,
            "stress_level": 7.0
        }
        
        print("📤 Sending request to /ask-wellness...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/ask-wellness",
            json=enhanced_data,
            timeout=120
        )
        
        response_time = time.time() - start_time
        
        print(f"⏱️  Response time: {response_time:.2f}s")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Enhanced endpoint success!")
            print(f"🔍 Query: {data.get('query', 'N/A')}")
            print(f"👤 User context: {data.get('user_context', {})}")
            print(f"🤖 LLM provider: {data.get('llm_provider', 'N/A')}")
            print(f"📝 Response length: {len(data.get('response', ''))} chars")
            
            response_text = data.get('response', '')
            if response_text:
                print(f"💬 Response preview: {response_text[:200]}...")
        else:
            print(f"❌ Enhanced endpoint error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Enhanced endpoint error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_wellness_api()
