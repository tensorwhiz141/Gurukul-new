#!/usr/bin/env python3
"""
Quick Test Script for Advanced Forecasting Implementation
Tests if the forecasting system is working correctly
"""

import requests
import json
import sys
from datetime import datetime, timedelta

def test_backend_status():
    """Test if the backend is running and forecasting is enabled"""
    print("🔍 Testing Backend Status...")
    
    try:
        response = requests.get('http://localhost:8002/forecast/status', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend is running!")
            print(f"   Forecasting Enabled: {data.get('forecasting_enabled', False)}")
            print(f"   Available Models: {data.get('available_models', [])}")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend not accessible: {e}")
        print("💡 Start backend with: python simple_api.py --port 8002")
        return False

def test_forecast_api():
    """Test the forecast API endpoint"""
    print("\n📊 Testing Forecast API...")
    
    # Generate sample data
    sample_data = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(30):
        date = start_date + timedelta(days=i)
        value = 100 + 20 * (i / 30) + (i % 7) * 5  # Simple trend with weekly pattern
        sample_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": round(value, 2)
        })
    
    try:
        response = requests.post(
            'http://localhost:8002/forecast',
            json={
                "data": sample_data,
                "metric_type": "general",
                "forecast_periods": 7,
                "user_id": "test_user"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Forecast API working!")
            
            if 'content' in data and 'forecast_data' in data['content']:
                forecast_data = data['content']['forecast_data']
                print(f"   Generated {len(forecast_data)} forecast points")
                print(f"   Model Used: {data['content'].get('model_used', 'Unknown')}")
                
                if data['content'].get('accuracy_metrics'):
                    metrics = data['content']['accuracy_metrics']
                    print(f"   MAE: {metrics.get('mae', 'N/A')}")
                
                return True
            else:
                print("⚠️ Forecast generated but unexpected format")
                return False
        else:
            print(f"❌ Forecast API returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Forecast API test failed: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are installed"""
    print("\n📦 Testing Dependencies...")
    
    try:
        import prophet
        print("✅ Prophet installed")
    except ImportError:
        print("❌ Prophet not installed - run: pip install prophet")
        return False
    
    try:
        import statsmodels
        print("✅ Statsmodels installed")
    except ImportError:
        print("❌ Statsmodels not installed - run: pip install statsmodels")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn installed")
    except ImportError:
        print("❌ Scikit-learn not installed - run: pip install scikit-learn")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Quick Test - Advanced Forecasting Implementation")
    print("=" * 60)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test backend status
    backend_ok = test_backend_status()
    
    # Test forecast API (only if backend is running)
    forecast_ok = False
    if backend_ok:
        forecast_ok = test_forecast_api()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Dependencies: {'✅ OK' if deps_ok else '❌ FAILED'}")
    print(f"   Backend Status: {'✅ OK' if backend_ok else '❌ FAILED'}")
    print(f"   Forecast API: {'✅ OK' if forecast_ok else '❌ FAILED'}")
    
    if deps_ok and backend_ok and forecast_ok:
        print("\n🎉 All tests passed! Advanced forecasting is working correctly.")
        print("\n🎯 Next Steps:")
        print("   1. Open your frontend: http://localhost:3000")
        print("   2. Navigate to 'Forecasting' in the sidebar")
        print("   3. Click 'Generate Forecast' to see it in action!")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        
        if not deps_ok:
            print("\n💡 Install missing dependencies:")
            print("   pip install prophet statsmodels scikit-learn")
        
        if not backend_ok:
            print("\n💡 Start the backend:")
            print("   python simple_api.py --port 8002")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
