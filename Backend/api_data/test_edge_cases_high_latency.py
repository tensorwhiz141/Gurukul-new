#!/usr/bin/env python3
"""
Edge Case Testing: High-Latency Agent Scenarios
Tests system behavior when AI agents take extended time to respond
"""

import requests
import time
import json
import threading
from datetime import datetime, timedelta

class HighLatencyTester:
    def __init__(self):
        self.financial_simulator_url = "http://localhost:8002"
        self.lesson_generator_url = "http://localhost:8000"
        self.max_wait_time = 900  # 15 minutes
        self.polling_interval = 5  # 5 seconds
        
    def test_extended_financial_simulation(self, timeout_minutes=15):
        """Test Financial Simulator with extended processing time"""
        
        print(f"⏰ Testing Extended Financial Simulation (timeout: {timeout_minutes} minutes)")
        print("=" * 70)
        
        # Submit a complex financial simulation
        simulation_data = {
            "user_id": "high-latency-financial-test",
            "user_name": "High Latency Test User",
            "income": 120000,
            "expenses": [
                {"name": "Mortgage", "amount": 3500},
                {"name": "Car Payment", "amount": 800},
                {"name": "Insurance", "amount": 600},
                {"name": "Food", "amount": 1200},
                {"name": "Utilities", "amount": 400},
                {"name": "Entertainment", "amount": 500},
                {"name": "Savings", "amount": 2000}
            ],
            "total_expenses": 9000,
            "goal": "Complex financial planning with multiple investment strategies and risk assessments",
            "financial_type": "aggressive",
            "risk_level": "high"
        }
        
        start_time = time.time()
        print(f"🚀 Starting complex financial simulation at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Submit simulation
            response = requests.post(
                f"{self.financial_simulator_url}/start-simulation",
                json=simulation_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to start simulation: {response.status_code}")
                return None
            
            data = response.json()
            task_id = data.get("task_id")
            print(f"✅ Simulation started with task ID: {task_id}")
            
            # Monitor progress with extended polling
            return self._monitor_long_running_task(
                task_id, 
                "financial_simulation",
                timeout_minutes,
                f"{self.financial_simulator_url}/simulation-status/{task_id}"
            )
            
        except Exception as e:
            print(f"❌ Error starting financial simulation: {e}")
            return None
    
    def test_extended_lesson_generation(self, timeout_minutes=10):
        """Test Lesson Generator with extended processing time"""
        
        print(f"\n📚 Testing Extended Lesson Generation (timeout: {timeout_minutes} minutes)")
        print("=" * 70)
        
        # Submit a complex lesson generation request
        lesson_data = {
            "subject": "Advanced Mathematics",
            "topic": "Quantum Computing Algorithms and Their Mathematical Foundations",
            "user_id": "high-latency-lesson-test",
            "include_wikipedia": True,
            "force_regenerate": True,
            "complexity_level": "advanced",
            "detailed_explanations": True
        }
        
        start_time = time.time()
        print(f"🚀 Starting complex lesson generation at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Submit lesson generation
            response = requests.post(
                f"{self.lesson_generator_url}/lessons",
                json=lesson_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to start lesson generation: {response.status_code}")
                return None
            
            data = response.json()
            task_id = data.get("task_id")
            print(f"✅ Lesson generation started with task ID: {task_id}")
            
            # Monitor progress with extended polling
            return self._monitor_long_running_task(
                task_id,
                "lesson_generation", 
                timeout_minutes,
                f"{self.lesson_generator_url}/lessons/status/{task_id}"
            )
            
        except Exception as e:
            print(f"❌ Error starting lesson generation: {e}")
            return None
    
    def _monitor_long_running_task(self, task_id, task_type, timeout_minutes, status_url):
        """Monitor a long-running task with detailed progress tracking"""
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        poll_count = 0
        status_history = []
        
        print(f"📡 Monitoring {task_type} task: {task_id}")
        print(f"⏰ Timeout: {timeout_minutes} minutes")
        print(f"🔄 Polling interval: {self.polling_interval} seconds")
        
        while time.time() - start_time < timeout_seconds:
            poll_count += 1
            elapsed_time = time.time() - start_time
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)
            
            try:
                # Check task status
                status_response = requests.get(status_url, timeout=10)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status", "unknown")
                    
                    status_entry = {
                        "poll_count": poll_count,
                        "elapsed_time": elapsed_time,
                        "status": status,
                        "timestamp": datetime.now().isoformat(),
                        "response_time": status_response.elapsed.total_seconds()
                    }
                    status_history.append(status_entry)
                    
                    print(f"📊 Poll {poll_count:3d} | {elapsed_minutes:2d}:{elapsed_seconds:02d} | Status: {status:15s} | Response: {status_response.elapsed.total_seconds():.2f}s")
                    
                    # Check for completion
                    if status in ["completed", "success"]:
                        print(f"🎉 Task completed after {elapsed_minutes}:{elapsed_seconds:02d}")
                        
                        # Try to get results
                        if task_type == "financial_simulation":
                            results_url = f"{self.financial_simulator_url}/simulation-results/{task_id}"
                        else:
                            results_url = status_url  # Lesson status includes results
                        
                        try:
                            results_response = requests.get(results_url, timeout=10)
                            if results_response.status_code == 200:
                                results_data = results_response.json()
                                print(f"✅ Results retrieved successfully")
                                return {
                                    "task_id": task_id,
                                    "task_type": task_type,
                                    "status": "completed",
                                    "total_time": elapsed_time,
                                    "poll_count": poll_count,
                                    "status_history": status_history,
                                    "results_available": True,
                                    "results_data": results_data
                                }
                            else:
                                print(f"⚠️ Task completed but results not available: {results_response.status_code}")
                        except Exception as e:
                            print(f"⚠️ Error retrieving results: {e}")
                        
                        return {
                            "task_id": task_id,
                            "task_type": task_type,
                            "status": "completed",
                            "total_time": elapsed_time,
                            "poll_count": poll_count,
                            "status_history": status_history,
                            "results_available": False
                        }
                    
                    elif status in ["failed", "error"]:
                        print(f"❌ Task failed after {elapsed_minutes}:{elapsed_seconds:02d}")
                        return {
                            "task_id": task_id,
                            "task_type": task_type,
                            "status": "failed",
                            "total_time": elapsed_time,
                            "poll_count": poll_count,
                            "status_history": status_history,
                            "error_data": status_data
                        }
                    
                    # Show progress milestones
                    if poll_count % 12 == 0:  # Every minute
                        print(f"⏳ Still processing... {elapsed_minutes} minutes elapsed")
                        
                        # Check for concerning patterns
                        if len(status_history) >= 10:
                            recent_statuses = [s["status"] for s in status_history[-10:]]
                            if len(set(recent_statuses)) == 1 and recent_statuses[0] not in ["completed", "success"]:
                                print(f"⚠️ Status unchanged for last 10 polls: {recent_statuses[0]}")
                
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
                    status_history.append({
                        "poll_count": poll_count,
                        "elapsed_time": elapsed_time,
                        "status": "status_check_failed",
                        "status_code": status_response.status_code,
                        "timestamp": datetime.now().isoformat()
                    })
                
            except requests.exceptions.Timeout:
                print(f"⏰ Status check timeout at poll {poll_count}")
                status_history.append({
                    "poll_count": poll_count,
                    "elapsed_time": elapsed_time,
                    "status": "timeout",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"❌ Error during status check: {e}")
                status_history.append({
                    "poll_count": poll_count,
                    "elapsed_time": elapsed_time,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Wait before next poll
            time.sleep(self.polling_interval)
        
        # Timeout reached
        elapsed_time = time.time() - start_time
        print(f"⏰ Timeout reached after {timeout_minutes} minutes")
        
        return {
            "task_id": task_id,
            "task_type": task_type,
            "status": "timeout",
            "total_time": elapsed_time,
            "poll_count": poll_count,
            "status_history": status_history,
            "timeout_minutes": timeout_minutes
        }
    
    def test_agent_unresponsiveness(self):
        """Test system behavior when agents become unresponsive"""
        
        print(f"\n🔌 Testing Agent Unresponsiveness Scenarios")
        print("=" * 70)
        
        # Test with invalid/unreachable endpoints
        test_scenarios = [
            {
                "name": "Invalid Financial Simulator Port",
                "url": "http://localhost:9999/start-simulation",
                "data": {
                    "user_id": "unresponsive-test",
                    "user_name": "Test User",
                    "income": 50000,
                    "expenses": [{"name": "Test", "amount": 1000}],
                    "total_expenses": 1000,
                    "goal": "Test unresponsiveness",
                    "financial_type": "conservative",
                    "risk_level": "low"
                }
            },
            {
                "name": "Invalid Lesson Generator Port",
                "url": "http://localhost:9998/lessons",
                "data": {
                    "subject": "Test Subject",
                    "topic": "Test Topic",
                    "user_id": "unresponsive-test",
                    "include_wikipedia": False
                }
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            print(f"\n🧪 Testing: {scenario['name']}")
            
            start_time = time.time()
            try:
                response = requests.post(
                    scenario["url"],
                    json=scenario["data"],
                    headers={"Content-Type": "application/json"},
                    timeout=5  # Short timeout to simulate unresponsiveness
                )
                
                result = {
                    "scenario": scenario["name"],
                    "status": "unexpected_response",
                    "status_code": response.status_code,
                    "response_time": time.time() - start_time
                }
                
            except requests.exceptions.ConnectTimeout:
                result = {
                    "scenario": scenario["name"],
                    "status": "connection_timeout",
                    "response_time": time.time() - start_time
                }
                print(f"   ✅ Expected connection timeout after {result['response_time']:.2f}s")
                
            except requests.exceptions.ConnectionError:
                result = {
                    "scenario": scenario["name"],
                    "status": "connection_refused",
                    "response_time": time.time() - start_time
                }
                print(f"   ✅ Expected connection refused after {result['response_time']:.2f}s")
                
            except Exception as e:
                result = {
                    "scenario": scenario["name"],
                    "status": "error",
                    "error": str(e),
                    "response_time": time.time() - start_time
                }
                print(f"   ❌ Unexpected error: {e}")
            
            results.append(result)
        
        return results

def run_high_latency_tests():
    """Run all high-latency edge case tests"""
    
    print("⏰ HIGH-LATENCY AGENT EDGE CASE TESTING")
    print("=" * 70)
    
    tester = HighLatencyTester()
    all_results = []
    
    # Test 1: Extended Financial Simulation
    try:
        result1 = tester.test_extended_financial_simulation(timeout_minutes=10)
        if result1:
            all_results.append(result1)
    except Exception as e:
        print(f"❌ Extended Financial Simulation test failed: {e}")
    
    # Test 2: Extended Lesson Generation
    try:
        result2 = tester.test_extended_lesson_generation(timeout_minutes=8)
        if result2:
            all_results.append(result2)
    except Exception as e:
        print(f"❌ Extended Lesson Generation test failed: {e}")
    
    # Test 3: Agent Unresponsiveness
    try:
        result3 = tester.test_agent_unresponsiveness()
        all_results.extend(result3)
    except Exception as e:
        print(f"❌ Agent Unresponsiveness test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 HIGH-LATENCY TEST SUMMARY")
    print("=" * 70)
    
    for result in all_results:
        if isinstance(result, dict) and "task_type" in result:
            print(f"✅ {result['task_type']}: {result['status']} ({result.get('total_time', 0):.1f}s)")
        elif isinstance(result, dict) and "scenario" in result:
            print(f"✅ {result['scenario']}: {result['status']}")
    
    # Save results
    with open("high_latency_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📄 Results saved to: high_latency_test_results.json")
    
    return all_results

if __name__ == "__main__":
    run_high_latency_tests()
