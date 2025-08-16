#!/usr/bin/env python3
"""
Edge Case Testing: Bursty Workloads
Tests system behavior under high concurrent load scenarios
"""

import asyncio
import aiohttp
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from datetime import datetime

class BurstyWorkloadTester:
    def __init__(self):
        self.financial_simulator_url = "http://localhost:8002"
        self.lesson_generator_url = "http://localhost:8000"
        self.results = []
        self.errors = []
        
    def test_concurrent_financial_simulations(self, num_concurrent=10):
        """Test multiple simultaneous Financial Simulator requests"""
        
        print(f"🔥 Testing {num_concurrent} Concurrent Financial Simulations")
        print("=" * 60)
        
        # Test data for financial simulations
        base_simulation_data = {
            "user_name": "Concurrent Test User",
            "income": 75000,
            "expenses": [
                {"name": "Rent", "amount": 2000},
                {"name": "Food", "amount": 800},
                {"name": "Transportation", "amount": 500}
            ],
            "total_expenses": 3300,
            "goal": "Test concurrent processing",
            "financial_type": "moderate",
            "risk_level": "medium"
        }
        
        def submit_financial_simulation(user_id):
            """Submit a single financial simulation request"""
            simulation_data = base_simulation_data.copy()
            simulation_data["user_id"] = f"concurrent-user-{user_id}"
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.financial_simulator_url}/start-simulation",
                    json=simulation_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                response_time = time.time() - start_time
                
                result = {
                    "user_id": user_id,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": response.status_code == 200,
                    "timestamp": datetime.now().isoformat()
                }
                
                if response.status_code == 200:
                    data = response.json()
                    result["task_id"] = data.get("task_id")
                    result["message"] = data.get("message")
                else:
                    result["error"] = response.text
                    
                return result
                
            except Exception as e:
                return {
                    "user_id": user_id,
                    "status_code": 0,
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Execute concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(submit_financial_simulation, i) for i in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        print(f"📊 Concurrent Financial Simulation Results:")
        print(f"   Total Requests: {num_concurrent}")
        print(f"   Successful: {len(successful_requests)}")
        print(f"   Failed: {len(failed_requests)}")
        print(f"   Success Rate: {len(successful_requests)/num_concurrent*100:.1f}%")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   Average Response Time: {sum(r['response_time'] for r in results)/len(results):.2f}s")
        
        if failed_requests:
            print(f"\n❌ Failed Requests:")
            for req in failed_requests[:3]:  # Show first 3 failures
                print(f"   User {req['user_id']}: {req.get('error', 'Unknown error')}")
        
        return {
            "test_type": "concurrent_financial_simulations",
            "total_requests": num_concurrent,
            "successful": len(successful_requests),
            "failed": len(failed_requests),
            "success_rate": len(successful_requests)/num_concurrent,
            "total_time": total_time,
            "results": results
        }
    
    def test_concurrent_lesson_generation(self, num_concurrent=8):
        """Test multiple simultaneous lesson generation requests"""
        
        print(f"\n📚 Testing {num_concurrent} Concurrent Lesson Generations")
        print("=" * 60)
        
        # Test data for lesson generation
        subjects_topics = [
            ("Mathematics", "Algebra"),
            ("Science", "Physics"),
            ("History", "World War II"),
            ("Literature", "Shakespeare"),
            ("Geography", "Climate Change"),
            ("Biology", "Genetics"),
            ("Chemistry", "Organic Compounds"),
            ("Philosophy", "Ethics")
        ]
        
        def submit_lesson_generation(request_id):
            """Submit a single lesson generation request"""
            subject, topic = subjects_topics[request_id % len(subjects_topics)]
            
            lesson_data = {
                "subject": subject,
                "topic": topic,
                "user_id": f"concurrent-lesson-user-{request_id}",
                "include_wikipedia": True,
                "force_regenerate": True
            }
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.lesson_generator_url}/lessons",
                    json=lesson_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                response_time = time.time() - start_time
                
                result = {
                    "request_id": request_id,
                    "subject": subject,
                    "topic": topic,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": response.status_code == 200,
                    "timestamp": datetime.now().isoformat()
                }
                
                if response.status_code == 200:
                    data = response.json()
                    result["task_id"] = data.get("task_id")
                    result["status"] = data.get("status")
                else:
                    result["error"] = response.text
                    
                return result
                
            except Exception as e:
                return {
                    "request_id": request_id,
                    "subject": subject,
                    "topic": topic,
                    "status_code": 0,
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Execute concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(submit_lesson_generation, i) for i in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        print(f"📊 Concurrent Lesson Generation Results:")
        print(f"   Total Requests: {num_concurrent}")
        print(f"   Successful: {len(successful_requests)}")
        print(f"   Failed: {len(failed_requests)}")
        print(f"   Success Rate: {len(successful_requests)/num_concurrent*100:.1f}%")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   Average Response Time: {sum(r['response_time'] for r in results)/len(results):.2f}s")
        
        if failed_requests:
            print(f"\n❌ Failed Requests:")
            for req in failed_requests[:3]:  # Show first 3 failures
                print(f"   {req['subject']}/{req['topic']}: {req.get('error', 'Unknown error')}")
        
        return {
            "test_type": "concurrent_lesson_generation",
            "total_requests": num_concurrent,
            "successful": len(successful_requests),
            "failed": len(failed_requests),
            "success_rate": len(successful_requests)/num_concurrent,
            "total_time": total_time,
            "results": results
        }
    
    def test_rapid_fire_requests(self, num_requests=20, delay_ms=100):
        """Test rapid-fire requests with minimal delay"""
        
        print(f"\n⚡ Testing {num_requests} Rapid-Fire Requests (delay: {delay_ms}ms)")
        print("=" * 60)
        
        def submit_rapid_request(request_id):
            """Submit a rapid-fire request"""
            time.sleep(request_id * delay_ms / 1000)  # Stagger requests slightly
            
            # Alternate between financial and lesson requests
            if request_id % 2 == 0:
                # Financial simulation request
                data = {
                    "user_id": f"rapid-financial-{request_id}",
                    "user_name": f"Rapid User {request_id}",
                    "income": 50000 + (request_id * 1000),
                    "expenses": [{"name": "Test", "amount": 1000}],
                    "total_expenses": 1000,
                    "goal": "Rapid fire test",
                    "financial_type": "conservative",
                    "risk_level": "low"
                }
                url = f"{self.financial_simulator_url}/start-simulation"
            else:
                # Lesson generation request
                data = {
                    "subject": "Mathematics",
                    "topic": f"Topic {request_id}",
                    "user_id": f"rapid-lesson-{request_id}",
                    "include_wikipedia": False,
                    "force_regenerate": True
                }
                url = f"{self.lesson_generator_url}/lessons"
            
            start_time = time.time()
            try:
                response = requests.post(
                    url,
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                return {
                    "request_id": request_id,
                    "type": "financial" if request_id % 2 == 0 else "lesson",
                    "status_code": response.status_code,
                    "response_time": time.time() - start_time,
                    "success": response.status_code == 200,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    "request_id": request_id,
                    "type": "financial" if request_id % 2 == 0 else "lesson",
                    "status_code": 0,
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Execute rapid-fire requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(submit_rapid_request, i) for i in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        financial_requests = [r for r in results if r["type"] == "financial"]
        lesson_requests = [r for r in results if r["type"] == "lesson"]
        
        print(f"📊 Rapid-Fire Request Results:")
        print(f"   Total Requests: {num_requests}")
        print(f"   Financial Requests: {len(financial_requests)} (Success: {len([r for r in financial_requests if r['success']])})")
        print(f"   Lesson Requests: {len(lesson_requests)} (Success: {len([r for r in lesson_requests if r['success']])})")
        print(f"   Overall Success Rate: {len(successful_requests)/num_requests*100:.1f}%")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   Requests per Second: {num_requests/total_time:.2f}")
        
        return {
            "test_type": "rapid_fire_requests",
            "total_requests": num_requests,
            "successful": len(successful_requests),
            "failed": len(failed_requests),
            "success_rate": len(successful_requests)/num_requests,
            "total_time": total_time,
            "requests_per_second": num_requests/total_time,
            "results": results
        }

def run_bursty_workload_tests():
    """Run all bursty workload tests"""
    
    print("🔥 BURSTY WORKLOAD EDGE CASE TESTING")
    print("=" * 70)
    
    tester = BurstyWorkloadTester()
    all_results = []
    
    # Test 1: Concurrent Financial Simulations
    try:
        result1 = tester.test_concurrent_financial_simulations(num_concurrent=5)
        all_results.append(result1)
    except Exception as e:
        print(f"❌ Concurrent Financial Simulation test failed: {e}")
    
    # Test 2: Concurrent Lesson Generation
    try:
        result2 = tester.test_concurrent_lesson_generation(num_concurrent=4)
        all_results.append(result2)
    except Exception as e:
        print(f"❌ Concurrent Lesson Generation test failed: {e}")
    
    # Test 3: Rapid-Fire Mixed Requests
    try:
        result3 = tester.test_rapid_fire_requests(num_requests=15, delay_ms=50)
        all_results.append(result3)
    except Exception as e:
        print(f"❌ Rapid-Fire Request test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 BURSTY WORKLOAD TEST SUMMARY")
    print("=" * 70)
    
    for result in all_results:
        print(f"✅ {result['test_type']}: {result['success_rate']*100:.1f}% success rate")
    
    # Save results
    with open("bursty_workload_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📄 Results saved to: bursty_workload_test_results.json")
    
    return all_results

if __name__ == "__main__":
    run_bursty_workload_tests()
