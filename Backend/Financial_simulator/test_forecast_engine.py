#!/usr/bin/env python3
"""
Test script for Forecast Engine v2
Tests core functionality, API endpoints, and agent integration.
"""

import requests
import json
import time
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_agent import create_prediction_agent, run_forecast_simulation
from agent_integration import run_end_to_end_simulation

def test_prediction_agent():
    """Test core prediction agent functionality."""
    print("🧪 Testing Prediction Agent Core Functionality")
    print("=" * 50)
    
    try:
        # Create prediction agent
        agent = create_prediction_agent(use_prophet=True)
        print("✅ Prediction agent created successfully")
        
        # Load historical data
        data = agent.load_task_history_data()
        print(f"✅ Historical data loaded: {len(data)} metrics")
        
        # Generate forecasts
        forecasts = agent.generate_forecasts(forecast_days=7)
        print(f"✅ Forecasts generated for {len(forecasts)} metrics")
        
        # Test risk assessment
        for metric in ['delay_risk', 'escalation_likelihood', 'daily_agent_load']:
            risk = agent.get_risk_assessment(metric)
            print(f"✅ {metric}: {risk['risk']} risk (confidence: {risk['confidence']:.2f})")
        
        # Test agent scoring
        score = agent.get_agent_score("test_agent", 15)
        print(f"✅ Agent scoring: {score['score']:.2f} ({score['capacity_status']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction agent test failed: {e}")
        return False

def test_api_endpoints(base_url="http://localhost:8002"):
    """Test API endpoints."""
    print("\n🌐 Testing API Endpoints")
    print("=" * 50)
    
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/forecast?days=7", None),
        ("GET", "/forecast-json?days=3", None),
        ("POST", "/score-agent", {"agent_id": "test_agent", "current_load": 10}),
        ("GET", "/metrics", None),
        ("POST", "/simulate-workflow", None)
    ]
    
    results = []
    
    for method, endpoint, payload in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {method} {endpoint} → {response.status_code}")
                results.append(True)
            else:
                print(f"⚠️ {method} {endpoint} → {response.status_code}")
                results.append(False)
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {method} {endpoint} → Connection failed (server not running?)")
            results.append(False)
        except Exception as e:
            print(f"❌ {method} {endpoint} → Error: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 API Test Results: {success_rate:.1f}% success rate")
    return success_rate > 80

def test_agent_integration():
    """Test multi-agent integration workflow."""
    print("\n🤖 Testing Multi-Agent Integration")
    print("=" * 50)
    
    try:
        # Run end-to-end simulation
        results = run_end_to_end_simulation()
        
        print(f"✅ Simulation completed: {results['simulation_id']}")
        print(f"✅ Workflow steps: {len(results['workflow_steps'])}")
        
        # Check each step
        for step in results['workflow_steps']:
            agent_name = step['agent']
            action = step['action']
            result = step['result']
            
            if 'error' not in result:
                print(f"✅ {agent_name} - {action}: Success")
            else:
                print(f"❌ {agent_name} - {action}: {result['error']}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")
        return False

def test_forecast_simulation():
    """Test complete forecast simulation."""
    print("\n📈 Testing Forecast Simulation")
    print("=" * 50)
    
    try:
        # Create agent and run simulation
        agent = create_prediction_agent(use_prophet=True)
        results = run_forecast_simulation(agent, simulation_days=7)
        
        print(f"✅ Simulation completed: {results['simulation_id']}")
        print(f"✅ Model type: {results['model_type']}")
        print(f"✅ Forecasts generated: {len(results['forecasts'])}")
        print(f"✅ Agent scores: {len(results['agent_scores'])}")
        
        # Check forecast quality
        for metric, forecast in results['forecasts'].items():
            predictions = forecast.get('predictions', [])
            if predictions:
                print(f"✅ {metric}: {len(predictions)} predictions generated")
            else:
                print(f"⚠️ {metric}: No predictions generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Forecast simulation test failed: {e}")
        return False

def test_edge_cases():
    """Test edge case handling."""
    print("\n⚠️ Testing Edge Cases")
    print("=" * 50)
    
    try:
        # Test with ARIMA fallback
        agent_arima = create_prediction_agent(use_prophet=False)
        forecasts_arima = agent_arima.generate_forecasts(forecast_days=3)
        print("✅ ARIMA fallback working")
        
        # Test with minimal data
        agent = create_prediction_agent(use_prophet=True)
        agent.historical_data = {}  # Clear data to test fallback
        forecasts = agent.generate_forecasts(forecast_days=3)
        print("✅ Minimal data handling working")
        
        # Test invalid agent scoring
        score = agent.get_agent_score("invalid_agent", -1)
        print("✅ Invalid input handling working")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return False

def generate_test_report():
    """Generate comprehensive test report."""
    print("\n📋 FORECAST ENGINE v2 - TEST REPORT")
    print("=" * 60)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Environment: Python {sys.version.split()[0]}")
    print("=" * 60)
    
    tests = [
        ("Core Prediction Agent", test_prediction_agent),
        ("API Endpoints", test_api_endpoints),
        ("Agent Integration", test_agent_integration),
        ("Forecast Simulation", test_forecast_simulation),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Tests...")
        start_time = time.time()
        
        try:
            success = test_func()
            duration = time.time() - start_time
            results.append((test_name, success, duration))
            
            if success:
                print(f"✅ {test_name} tests PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {test_name} tests FAILED ({duration:.2f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            results.append((test_name, False, duration))
            print(f"❌ {test_name} tests ERROR: {e} ({duration:.2f}s)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    total_time = sum(duration for _, _, duration in results)
    
    for test_name, success, duration in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<25} {status:<6} ({duration:.2f}s)")
    
    print("-" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
    else:
        print(f"\n⚠️ {total - passed} TESTS FAILED - REVIEW REQUIRED")
    
    return passed == total

if __name__ == "__main__":
    print("🚀 Starting Forecast Engine v2 Test Suite")
    print("=" * 60)
    
    # Check if server should be tested
    test_api = "--no-api" not in sys.argv
    
    if not test_api:
        print("ℹ️ Skipping API tests (--no-api flag detected)")
    
    # Run tests
    success = generate_test_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
