#!/usr/bin/env python3
"""
Test script for MVD endpoints
Run this to verify the endpoints are working
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an endpoint and print the result"""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n{'='*50}")
    print(f"Testing: {method} {endpoint}")
    print(f"URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Success!")
            try:
                result = response.json()
                print("Response:")
                print(json.dumps(result, indent=2))
            except:
                print("Response (text):", response.text)
        else:
            print("❌ Failed!")
            print("Response:", response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🚀 Testing MVD Endpoints")
    print("Make sure your server is running on http://localhost:8000")
    
    # Test MVD endpoints
    test_endpoint("/dashboard/user123")
    test_endpoint("/syllabus/math_001")
    test_endpoint("/lesson/lesson_001")
    test_endpoint("/quiz/lesson_001")
    
    # Test lesson complete endpoint
    test_data = {
        "user_id": "user123",
        "lesson_id": "lesson_001",
        "quiz_score": 85.5,
        "time_spent": 45,
        "completed_sections": ["section_1", "section_2"]
    }
    test_endpoint("/lesson-complete", method="POST", data=test_data)
    
    # Test existing endpoints
    print(f"\n{'='*50}")
    print("Testing existing endpoints for comparison:")
    test_endpoint("/health")
    test_endpoint("/subjects")
    
    print(f"\n{'='*50}")
    print("✅ MVD Endpoint Testing Complete!")

if __name__ == "__main__":
    main()
