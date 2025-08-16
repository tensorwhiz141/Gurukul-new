#!/usr/bin/env python3
"""
Test script for Financial Simulator endpoint
"""

import requests
import json

def test_financial_simulator():
    """Test the /start-simulation endpoint on port 8002"""
    
    # Test data matching the frontend format
    test_data = {
        "user_id": "test-user-123",
        "user_name": "Test User",
        "income": 5000,
        "expenses": [
            {"name": "Rent", "amount": 1500},
            {"name": "Food", "amount": 500},
            {"name": "Transportation", "amount": 300}
        ],
        "total_expenses": 2300,
        "goal": "Save for emergency fund",
        "financial_type": "conservative",
        "risk_level": "low"
    }
    
    print("🧪 Testing Financial Simulator Endpoint")
    print("=" * 50)
    print(f"📡 URL: http://localhost:8002/start-simulation")
    print(f"📊 Test Data: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8002/start-simulation",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Financial Simulator is working")
            print(f"📋 Response: {json.dumps(result, indent=2)}")
            
            if "task_id" in result:
                print(f"🎯 Task ID: {result['task_id']}")
                return True
            else:
                print("⚠️  No task_id in response")
                return False
                
        elif response.status_code == 404:
            print("❌ 404 Not Found - Endpoint doesn't exist")
            print("💡 Check if the Financial Simulator service is running correctly")
            
        elif response.status_code == 422:
            print("❌ 422 Validation Error - Invalid request format")
            print(f"📄 Response: {response.text}")
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error - Financial Simulator service is not running")
        print("💡 Start the service: cd Backend/Financial_simulator/Financial_simulator && python langgraph_api.py")
        return False
        
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_service_status():
    """Test if the Financial Simulator service is running"""
    
    print("\n🔍 Checking Financial Simulator Service Status")
    print("=" * 50)
    
    try:
        # Test health/docs endpoint
        response = requests.get("http://localhost:8002/docs", timeout=5)
        
        if response.status_code == 200:
            print("✅ Financial Simulator service is running")
            print("📚 API documentation is accessible at http://localhost:8002/docs")
            return True
        else:
            print(f"⚠️  Service responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Financial Simulator service is not running on port 8002")
        return False
        
    except Exception as e:
        print(f"❌ Error checking service: {e}")
        return False

if __name__ == "__main__":
    print("🏦 Financial Simulator Endpoint Test")
    print("=" * 60)
    
    # First check if service is running
    service_running = test_service_status()
    
    if service_running:
        # Test the endpoint
        endpoint_working = test_financial_simulator()
        
        if endpoint_working:
            print("\n🎉 SUCCESS! Financial Simulator is working correctly")
            print("✅ Frontend should now be able to connect to port 8002")
        else:
            print("\n❌ Financial Simulator endpoint has issues")
    else:
        print("\n❌ Financial Simulator service is not running")
        print("💡 Start it with: cd Backend/Financial_simulator/Financial_simulator && python langgraph_api.py")
    
    print("\n" + "=" * 60)
    print("Test completed!")
