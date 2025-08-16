#!/usr/bin/env python3
"""
Test Domain-Specific Forecasting Models
Comprehensive testing for Edumentor and Wellness prediction agents
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_edumentor_agent():
    """Test Edumentor prediction agent"""
    print("🎓 Testing Edumentor Prediction Agent...")
    
    try:
        from edumentor_prediction_agent import EdumentorPredictionAgent
        
        # Initialize agent
        agent = EdumentorPredictionAgent(use_prophet=True)
        print("✅ Edumentor agent initialized successfully")
        
        # Test data loading
        data = agent.load_edumentor_data()
        print(f"✅ Educational data loaded: {len(data)} metrics")
        
        # List available metrics
        print("📊 Available educational metrics:")
        for metric in data.keys():
            print(f"   • {metric}")
        
        # Test prediction
        result = agent.predict_learning_outcomes("test_student_123", forecast_days=14)
        
        if 'error' in result:
            print(f"❌ Prediction failed: {result['error']}")
            return False
        
        print("✅ Learning outcome prediction successful")
        print(f"📈 Predictions generated for {len(result['predictions'])} metrics")
        print(f"💡 Insights generated: {len(result['insights'])}")
        print(f"⚠️  Risk assessments: {len(result['risk_assessment'])}")
        print(f"📋 Recommendations: {len(result['recommendations'])}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_wellness_agent():
    """Test Wellness prediction agent"""
    print("\n🧘 Testing Wellness Prediction Agent...")
    
    try:
        from wellness_prediction_agent import WellnessPredictionAgent
        
        # Initialize agent
        agent = WellnessPredictionAgent(use_prophet=True)
        print("✅ Wellness agent initialized successfully")
        
        # Test data loading
        data = agent.load_wellness_data()
        print(f"✅ Wellness data loaded: {len(data)} metrics")
        
        # List available metrics
        print("📊 Available wellness metrics:")
        for metric in data.keys():
            print(f"   • {metric}")
        
        # Test prediction
        result = agent.predict_wellness_outcomes("test_user_456", forecast_days=14)
        
        if 'error' in result:
            print(f"❌ Prediction failed: {result['error']}")
            return False
        
        print("✅ Wellness outcome prediction successful")
        print(f"📈 Predictions generated for {len(result['predictions'])} metrics")
        print(f"💡 Insights generated: {len(result['insights'])}")
        print(f"⚠️  Risk assessments: {len(result['risk_assessment'])}")
        print(f"📋 Recommendations: {len(result['recommendations'])}")
        print(f"🚨 Intervention priority: {result['intervention_priority']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_domain_forecasting():
    """Test combined domain forecasting"""
    print("\n🔮 Testing Domain-Specific Forecasting Integration...")
    
    try:
        from domain_specific_forecasting import DomainSpecificForecasting
        
        # Initialize forecasting system
        forecasting = DomainSpecificForecasting()
        print("✅ Domain forecasting system initialized")
        
        # Test Edumentor forecasting
        print("\n📚 Testing Edumentor forecasting...")
        edumentor_result = await forecasting.generate_edumentor_forecast("test_student_789", 21)
        
        if 'error' in edumentor_result:
            print(f"❌ Edumentor forecasting failed: {edumentor_result['error']}")
        else:
            print("✅ Edumentor forecasting successful")
            print(f"   Domain: {edumentor_result['domain']}")
            print(f"   Forecast type: {edumentor_result['forecast_type']}")
        
        # Test Wellness forecasting
        print("\n🌿 Testing Wellness forecasting...")
        wellness_result = await forecasting.generate_wellness_forecast("test_user_789", 21)
        
        if 'error' in wellness_result:
            print(f"❌ Wellness forecasting failed: {wellness_result['error']}")
        else:
            print("✅ Wellness forecasting successful")
            print(f"   Domain: {wellness_result['domain']}")
            print(f"   Forecast type: {wellness_result['forecast_type']}")
        
        # Test Combined forecasting
        print("\n🔗 Testing Combined forecasting...")
        combined_result = await forecasting.generate_combined_forecast("test_user_789", 21)
        
        if 'error' in combined_result:
            print(f"❌ Combined forecasting failed: {combined_result['error']}")
            return False
        else:
            print("✅ Combined forecasting successful")
            print(f"   Domain: {combined_result['domain']}")
            print(f"   Forecast type: {combined_result['forecast_type']}")
            print(f"   Cross-domain insights: {len(combined_result['cross_domain_insights'])}")
            print(f"   Holistic recommendations: {len(combined_result['holistic_recommendations'])}")
            print(f"   Overall risk level: {combined_result['overall_risk_assessment']['overall_risk_level']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("🔍 Testing Dependencies...")
    
    dependencies = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'prophet': 'Time series forecasting (optional)',
        'statsmodels': 'Statistical models (optional)'
    }
    
    all_good = True
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep}: OK - {description}")
        except ImportError:
            if dep in ['prophet', 'statsmodels']:
                print(f"⚠️  {dep}: Missing (optional) - {description}")
            else:
                print(f"❌ {dep}: Missing (required) - {description}")
                all_good = False
    
    return all_good

def test_api_integration():
    """Test API integration"""
    print("\n🌐 Testing API Integration...")
    
    try:
        # Test import of API functions
        from domain_specific_forecasting import (
            forecast_edumentor, 
            forecast_wellness, 
            forecast_combined,
            ForecastRequest
        )
        print("✅ API functions imported successfully")
        
        # Test request model
        request = ForecastRequest(
            user_id="test_api_user",
            domain="combined",
            forecast_days=30
        )
        print("✅ ForecastRequest model working")
        print(f"   User ID: {request.user_id}")
        print(f"   Domain: {request.domain}")
        print(f"   Forecast days: {request.forecast_days}")
        
        return True
        
    except ImportError as e:
        print(f"❌ API integration test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 DOMAIN-SPECIFIC FORECASTING TEST SUITE")
    print("=" * 50)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test individual agents
    edumentor_ok = test_edumentor_agent()
    wellness_ok = test_wellness_agent()
    
    # Test integration
    integration_ok = await test_domain_forecasting()
    
    # Test API integration
    api_ok = test_api_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY:")
    print(f"  Dependencies: {'✅ OK' if deps_ok else '❌ FAILED'}")
    print(f"  Edumentor Agent: {'✅ OK' if edumentor_ok else '❌ FAILED'}")
    print(f"  Wellness Agent: {'✅ OK' if wellness_ok else '❌ FAILED'}")
    print(f"  Domain Integration: {'✅ OK' if integration_ok else '❌ FAILED'}")
    print(f"  API Integration: {'✅ OK' if api_ok else '❌ FAILED'}")
    
    if all([deps_ok, edumentor_ok, wellness_ok, integration_ok, api_ok]):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Domain-specific forecasting is ready for production")
        print("\n🚀 Available API Endpoints:")
        print("  POST /forecast/edumentor - Educational metrics forecasting")
        print("  POST /forecast/wellness - Wellness metrics forecasting") 
        print("  POST /forecast/combined - Cross-domain forecasting")
        print("  GET  /forecast/domains - Available domains info")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("💡 Check the error messages above")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        input("\nPress Enter to exit...")
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
