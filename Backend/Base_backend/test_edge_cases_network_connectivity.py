#!/usr/bin/env python3
"""
Edge Case Testing: Network and Connectivity Issues
Tests system behavior during network failures and connectivity problems
"""

import requests
import time
import json
import random
import threading
from datetime import datetime
import socket

class NetworkConnectivityTester:
    def __init__(self):
        self.financial_simulator_url = "http://localhost:8002"
        self.lesson_generator_url = "http://localhost:8000"
        self.test_results = []
        
    def test_intermittent_connectivity(self, num_requests=10):
        """Test behavior during intermittent network connectivity"""
        
        print(f"🌐 Testing Intermittent Network Connectivity ({num_requests} requests)")
        print("=" * 70)
        
        def simulate_network_request(request_id):
            """Simulate a request with potential network issues"""
            
            # Randomly introduce network delays and failures
            network_delay = random.uniform(0, 2)  # 0-2 second delay
            failure_chance = 0.3  # 30% chance of simulated failure
            
            time.sleep(network_delay)
            
            # Simulate network failure
            if random.random() < failure_chance:
                return {
                    "request_id": request_id,
                    "status": "simulated_network_failure",
                    "network_delay": network_delay,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Make actual request
            request_data = {
                "user_id": f"network-test-{request_id}",
                "user_name": f"Network Test User {request_id}",
                "income": 60000,
                "expenses": [{"name": "Test", "amount": 1500}],
                "total_expenses": 1500,
                "goal": "Network connectivity test",
                "financial_type": "moderate",
                "risk_level": "medium"
            }
            
            start_time = time.time()
            try:
                # Use shorter timeout to simulate network issues
                response = requests.post(
                    f"{self.financial_simulator_url}/start-simulation",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
                
                return {
                    "request_id": request_id,
                    "status": "success" if response.status_code == 200 else "http_error",
                    "status_code": response.status_code,
                    "response_time": time.time() - start_time,
                    "network_delay": network_delay,
                    "timestamp": datetime.now().isoformat()
                }
                
            except requests.exceptions.Timeout:
                return {
                    "request_id": request_id,
                    "status": "timeout",
                    "response_time": time.time() - start_time,
                    "network_delay": network_delay,
                    "timestamp": datetime.now().isoformat()
                }
                
            except requests.exceptions.ConnectionError:
                return {
                    "request_id": request_id,
                    "status": "connection_error",
                    "response_time": time.time() - start_time,
                    "network_delay": network_delay,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                    "response_time": time.time() - start_time,
                    "network_delay": network_delay,
                    "timestamp": datetime.now().isoformat()
                }
        
        # Execute requests with simulated network issues
        results = []
        for i in range(num_requests):
            result = simulate_network_request(i)
            results.append(result)
            print(f"   Request {i:2d}: {result['status']:20s} | Delay: {result['network_delay']:.2f}s")
        
        # Analyze results
        successful = [r for r in results if r["status"] == "success"]
        timeouts = [r for r in results if r["status"] == "timeout"]
        connection_errors = [r for r in results if r["status"] == "connection_error"]
        simulated_failures = [r for r in results if r["status"] == "simulated_network_failure"]
        
        print(f"\n📊 Intermittent Connectivity Results:")
        print(f"   Total Requests: {num_requests}")
        print(f"   Successful: {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
        print(f"   Timeouts: {len(timeouts)} ({len(timeouts)/num_requests*100:.1f}%)")
        print(f"   Connection Errors: {len(connection_errors)} ({len(connection_errors)/num_requests*100:.1f}%)")
        print(f"   Simulated Failures: {len(simulated_failures)} ({len(simulated_failures)/num_requests*100:.1f}%)")
        
        return {
            "test_type": "intermittent_connectivity",
            "total_requests": num_requests,
            "results": results,
            "summary": {
                "successful": len(successful),
                "timeouts": len(timeouts),
                "connection_errors": len(connection_errors),
                "simulated_failures": len(simulated_failures)
            }
        }
    
    def test_api_endpoint_failures(self):
        """Test behavior when API endpoints are unavailable"""
        
        print(f"\n🔌 Testing API Endpoint Failures")
        print("=" * 70)
        
        test_scenarios = [
            {
                "name": "Financial Simulator Unavailable",
                "url": "http://localhost:8002/start-simulation",
                "backup_url": "http://localhost:8001/start-simulation",  # Non-existent backup
                "data": {
                    "user_id": "endpoint-failure-test",
                    "user_name": "Endpoint Test User",
                    "income": 55000,
                    "expenses": [{"name": "Test", "amount": 1200}],
                    "total_expenses": 1200,
                    "goal": "Endpoint failure test",
                    "financial_type": "conservative",
                    "risk_level": "low"
                }
            },
            {
                "name": "Lesson Generator Unavailable",
                "url": "http://localhost:8000/lessons",
                "backup_url": "http://localhost:8001/lessons",  # Non-existent backup
                "data": {
                    "subject": "Test Subject",
                    "topic": "Endpoint Failure Test",
                    "user_id": "endpoint-failure-test",
                    "include_wikipedia": False
                }
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            print(f"\n🧪 Testing: {scenario['name']}")
            
            # Test primary endpoint failure
            primary_result = self._test_endpoint_failure(
                scenario["url"], 
                scenario["data"], 
                f"{scenario['name']} - Primary"
            )
            results.append(primary_result)
            
            # Test backup endpoint failure
            backup_result = self._test_endpoint_failure(
                scenario["backup_url"], 
                scenario["data"], 
                f"{scenario['name']} - Backup"
            )
            results.append(backup_result)
        
        return {
            "test_type": "api_endpoint_failures",
            "results": results
        }
    
    def _test_endpoint_failure(self, url, data, test_name):
        """Test a specific endpoint failure scenario"""
        
        start_time = time.time()
        try:
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=3
            )
            
            result = {
                "test_name": test_name,
                "url": url,
                "status": "unexpected_success",
                "status_code": response.status_code,
                "response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            print(f"   ⚠️ Unexpected success: {response.status_code}")
            
        except requests.exceptions.ConnectTimeout:
            result = {
                "test_name": test_name,
                "url": url,
                "status": "connect_timeout",
                "response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            print(f"   ✅ Expected connect timeout: {result['response_time']:.2f}s")
            
        except requests.exceptions.ConnectionError:
            result = {
                "test_name": test_name,
                "url": url,
                "status": "connection_refused",
                "response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            print(f"   ✅ Expected connection refused: {result['response_time']:.2f}s")
            
        except requests.exceptions.Timeout:
            result = {
                "test_name": test_name,
                "url": url,
                "status": "timeout",
                "response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            print(f"   ✅ Expected timeout: {result['response_time']:.2f}s")
            
        except Exception as e:
            result = {
                "test_name": test_name,
                "url": url,
                "status": "error",
                "error": str(e),
                "response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            print(f"   ❌ Unexpected error: {e}")
        
        return result
    
    def test_partial_data_corruption(self):
        """Test behavior with corrupted or malformed data"""
        
        print(f"\n🔧 Testing Partial Data Corruption")
        print("=" * 70)
        
        corruption_scenarios = [
            {
                "name": "Missing Required Fields",
                "url": f"{self.financial_simulator_url}/start-simulation",
                "data": {
                    "user_id": "corruption-test",
                    # Missing required fields like income, expenses
                    "goal": "Test missing fields"
                }
            },
            {
                "name": "Invalid Data Types",
                "url": f"{self.financial_simulator_url}/start-simulation",
                "data": {
                    "user_id": "corruption-test",
                    "user_name": "Corruption Test",
                    "income": "not_a_number",  # Should be numeric
                    "expenses": "not_an_array",  # Should be array
                    "total_expenses": None,
                    "goal": "Test invalid types",
                    "financial_type": 123,  # Should be string
                    "risk_level": []  # Should be string
                }
            },
            {
                "name": "Extremely Large Values",
                "url": f"{self.financial_simulator_url}/start-simulation",
                "data": {
                    "user_id": "corruption-test",
                    "user_name": "Corruption Test",
                    "income": 999999999999999,  # Extremely large number
                    "expenses": [{"name": "Test", "amount": 888888888888888}],
                    "total_expenses": 888888888888888,
                    "goal": "Test large values",
                    "financial_type": "moderate",
                    "risk_level": "medium"
                }
            },
            {
                "name": "Malformed JSON Structure",
                "url": f"{self.lesson_generator_url}/lessons",
                "data": {
                    "subject": "Test Subject",
                    "topic": "Test Topic",
                    "user_id": "corruption-test",
                    "nested_invalid": {
                        "deeply": {
                            "nested": {
                                "structure": "that_might_cause_issues"
                            }
                        }
                    },
                    "circular_reference": None  # Would be circular if we could create one
                }
            }
        ]
        
        results = []
        
        for scenario in corruption_scenarios:
            print(f"\n🧪 Testing: {scenario['name']}")
            
            start_time = time.time()
            try:
                response = requests.post(
                    scenario["url"],
                    json=scenario["data"],
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                result = {
                    "scenario": scenario["name"],
                    "status": "response_received",
                    "status_code": response.status_code,
                    "response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                if response.status_code == 400:
                    print(f"   ✅ Expected 400 Bad Request: {response.status_code}")
                    result["status"] = "expected_bad_request"
                elif response.status_code == 422:
                    print(f"   ✅ Expected 422 Unprocessable Entity: {response.status_code}")
                    result["status"] = "expected_validation_error"
                elif response.status_code == 200:
                    print(f"   ⚠️ Unexpected success with corrupted data: {response.status_code}")
                    result["status"] = "unexpected_success"
                else:
                    print(f"   ❓ Unexpected status code: {response.status_code}")
                    result["status"] = "unexpected_status"
                
                try:
                    result["response_data"] = response.json()
                except:
                    result["response_text"] = response.text[:200]  # First 200 chars
                
            except requests.exceptions.Timeout:
                result = {
                    "scenario": scenario["name"],
                    "status": "timeout",
                    "response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
                print(f"   ⏰ Request timeout: {result['response_time']:.2f}s")
                
            except Exception as e:
                result = {
                    "scenario": scenario["name"],
                    "status": "error",
                    "error": str(e),
                    "response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
                print(f"   ❌ Error: {e}")
            
            results.append(result)
        
        return {
            "test_type": "partial_data_corruption",
            "results": results
        }
    
    def test_browser_refresh_simulation(self):
        """Test behavior when browser refresh/navigation occurs during processing"""
        
        print(f"\n🔄 Testing Browser Refresh/Navigation Simulation")
        print("=" * 70)
        
        # Start a financial simulation
        simulation_data = {
            "user_id": "browser-refresh-test",
            "user_name": "Browser Refresh Test",
            "income": 70000,
            "expenses": [{"name": "Test", "amount": 2000}],
            "total_expenses": 2000,
            "goal": "Test browser refresh behavior",
            "financial_type": "moderate",
            "risk_level": "medium"
        }
        
        print("🚀 Starting financial simulation...")
        
        try:
            response = requests.post(
                f"{self.financial_simulator_url}/start-simulation",
                json=simulation_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                task_id = data.get("task_id")
                print(f"✅ Simulation started: {task_id}")
                
                # Simulate browser refresh by checking if task is still accessible
                time.sleep(2)  # Wait a bit
                
                print("🔄 Simulating browser refresh - checking task persistence...")
                
                status_response = requests.get(
                    f"{self.financial_simulator_url}/simulation-status/{task_id}",
                    timeout=5
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"✅ Task still accessible after 'refresh': {status_data.get('status')}")
                    
                    return {
                        "test_type": "browser_refresh_simulation",
                        "task_id": task_id,
                        "status": "task_persistent",
                        "task_status": status_data.get("status"),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    print(f"❌ Task not accessible after 'refresh': {status_response.status_code}")
                    return {
                        "test_type": "browser_refresh_simulation",
                        "task_id": task_id,
                        "status": "task_lost",
                        "status_code": status_response.status_code,
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                print(f"❌ Failed to start simulation: {response.status_code}")
                return {
                    "test_type": "browser_refresh_simulation",
                    "status": "simulation_start_failed",
                    "status_code": response.status_code,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error during browser refresh test: {e}")
            return {
                "test_type": "browser_refresh_simulation",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

def run_network_connectivity_tests():
    """Run all network connectivity edge case tests"""
    
    print("🌐 NETWORK CONNECTIVITY EDGE CASE TESTING")
    print("=" * 70)
    
    tester = NetworkConnectivityTester()
    all_results = []
    
    # Test 1: Intermittent Connectivity
    try:
        result1 = tester.test_intermittent_connectivity(num_requests=8)
        all_results.append(result1)
    except Exception as e:
        print(f"❌ Intermittent Connectivity test failed: {e}")
    
    # Test 2: API Endpoint Failures
    try:
        result2 = tester.test_api_endpoint_failures()
        all_results.append(result2)
    except Exception as e:
        print(f"❌ API Endpoint Failures test failed: {e}")
    
    # Test 3: Partial Data Corruption
    try:
        result3 = tester.test_partial_data_corruption()
        all_results.append(result3)
    except Exception as e:
        print(f"❌ Partial Data Corruption test failed: {e}")
    
    # Test 4: Browser Refresh Simulation
    try:
        result4 = tester.test_browser_refresh_simulation()
        all_results.append(result4)
    except Exception as e:
        print(f"❌ Browser Refresh Simulation test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 NETWORK CONNECTIVITY TEST SUMMARY")
    print("=" * 70)
    
    for result in all_results:
        if isinstance(result, dict) and "test_type" in result:
            print(f"✅ {result['test_type']}: Completed")
    
    # Save results
    with open("network_connectivity_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📄 Results saved to: network_connectivity_test_results.json")
    
    return all_results

if __name__ == "__main__":
    run_network_connectivity_tests()
