#!/usr/bin/env python3
"""
Comprehensive Edge Case Test Runner
Executes all edge case tests and generates detailed reports
"""

import json
import time
import sys
import os
from datetime import datetime
import subprocess

# Import our test modules
from test_edge_cases_bursty_workloads import run_bursty_workload_tests
from test_edge_cases_high_latency import run_high_latency_tests
from test_edge_cases_network_connectivity import run_network_connectivity_tests
from system_monitoring import get_monitor

class EdgeCaseTestRunner:
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.monitor = get_monitor()
        
    def run_all_tests(self):
        """Run all edge case tests with comprehensive reporting"""
        
        print("🧪 COMPREHENSIVE EDGE CASE TEST SUITE")
        print("=" * 70)
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = time.time()
        
        # Test 1: Bursty Workloads
        print("🔥 TEST SUITE 1: BURSTY WORKLOADS")
        print("-" * 50)
        try:
            bursty_results = run_bursty_workload_tests()
            self.test_results["bursty_workloads"] = {
                "status": "completed",
                "results": bursty_results,
                "timestamp": datetime.now().isoformat()
            }
            print("✅ Bursty workload tests completed")
        except Exception as e:
            print(f"❌ Bursty workload tests failed: {e}")
            self.test_results["bursty_workloads"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        print("\n" + "="*70 + "\n")
        
        # Test 2: High Latency Scenarios
        print("⏰ TEST SUITE 2: HIGH LATENCY SCENARIOS")
        print("-" * 50)
        try:
            latency_results = run_high_latency_tests()
            self.test_results["high_latency"] = {
                "status": "completed",
                "results": latency_results,
                "timestamp": datetime.now().isoformat()
            }
            print("✅ High latency tests completed")
        except Exception as e:
            print(f"❌ High latency tests failed: {e}")
            self.test_results["high_latency"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        print("\n" + "="*70 + "\n")
        
        # Test 3: Network Connectivity Issues
        print("🌐 TEST SUITE 3: NETWORK CONNECTIVITY")
        print("-" * 50)
        try:
            network_results = run_network_connectivity_tests()
            self.test_results["network_connectivity"] = {
                "status": "completed",
                "results": network_results,
                "timestamp": datetime.now().isoformat()
            }
            print("✅ Network connectivity tests completed")
        except Exception as e:
            print(f"❌ Network connectivity tests failed: {e}")
            self.test_results["network_connectivity"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        print("\n" + "="*70 + "\n")
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        # Generate monitoring dashboard
        self.generate_monitoring_dashboard()
        
        return self.test_results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive test report"""
        
        total_time = self.end_time - self.start_time
        
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 70)
        print(f"🕐 Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"📅 Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test suite summary
        completed_tests = sum(1 for result in self.test_results.values() if result["status"] == "completed")
        failed_tests = sum(1 for result in self.test_results.values() if result["status"] == "failed")
        total_tests = len(self.test_results)
        
        print(f"📈 Test Suite Summary:")
        print(f"   Total Test Suites: {total_tests}")
        print(f"   Completed: {completed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {completed_tests/total_tests*100:.1f}%")
        print()
        
        # Detailed results for each test suite
        for test_name, test_data in self.test_results.items():
            print(f"🔍 {test_name.upper().replace('_', ' ')} RESULTS:")
            print(f"   Status: {'✅ PASSED' if test_data['status'] == 'completed' else '❌ FAILED'}")
            
            if test_data["status"] == "completed" and "results" in test_data:
                results = test_data["results"]
                if isinstance(results, list):
                    print(f"   Test Cases: {len(results)}")
                    for i, result in enumerate(results):
                        if isinstance(result, dict) and "test_type" in result:
                            success_rate = result.get("success_rate", 0)
                            print(f"     {i+1}. {result['test_type']}: {success_rate*100:.1f}% success")
                        elif isinstance(result, dict) and "scenario" in result:
                            print(f"     {i+1}. {result['scenario']}: {result['status']}")
            
            if test_data["status"] == "failed":
                print(f"   Error: {test_data.get('error', 'Unknown error')}")
            
            print()
        
        # Edge case analysis
        self.analyze_edge_cases()
        
        # Performance analysis
        self.analyze_performance()
        
        # Recommendations
        self.generate_recommendations()
        
        # Save detailed report
        self.save_detailed_report()
    
    def analyze_edge_cases(self):
        """Analyze edge cases discovered during testing"""
        
        print("🚨 EDGE CASE ANALYSIS:")
        
        edge_case_summary = {
            "high_latency_events": 0,
            "network_failures": 0,
            "timeout_events": 0,
            "concurrent_load_issues": 0,
            "data_corruption_cases": 0
        }
        
        # Analyze bursty workload results
        if "bursty_workloads" in self.test_results and self.test_results["bursty_workloads"]["status"] == "completed":
            bursty_results = self.test_results["bursty_workloads"]["results"]
            for result in bursty_results:
                if isinstance(result, dict) and "success_rate" in result:
                    if result["success_rate"] < 0.8:  # Less than 80% success
                        edge_case_summary["concurrent_load_issues"] += 1
        
        # Analyze high latency results
        if "high_latency" in self.test_results and self.test_results["high_latency"]["status"] == "completed":
            latency_results = self.test_results["high_latency"]["results"]
            for result in latency_results:
                if isinstance(result, dict):
                    if result.get("status") == "timeout":
                        edge_case_summary["timeout_events"] += 1
                    elif result.get("total_time", 0) > 300:  # More than 5 minutes
                        edge_case_summary["high_latency_events"] += 1
        
        # Analyze network connectivity results
        if "network_connectivity" in self.test_results and self.test_results["network_connectivity"]["status"] == "completed":
            network_results = self.test_results["network_connectivity"]["results"]
            for result in network_results:
                if isinstance(result, dict) and "results" in result:
                    for sub_result in result["results"]:
                        if isinstance(sub_result, dict):
                            if sub_result.get("status") in ["connection_error", "timeout", "simulated_network_failure"]:
                                edge_case_summary["network_failures"] += 1
                            elif sub_result.get("status") in ["expected_bad_request", "expected_validation_error"]:
                                edge_case_summary["data_corruption_cases"] += 1
        
        print(f"   🔥 High Latency Events: {edge_case_summary['high_latency_events']}")
        print(f"   🌐 Network Failures: {edge_case_summary['network_failures']}")
        print(f"   ⏰ Timeout Events: {edge_case_summary['timeout_events']}")
        print(f"   🚀 Concurrent Load Issues: {edge_case_summary['concurrent_load_issues']}")
        print(f"   🔧 Data Corruption Cases: {edge_case_summary['data_corruption_cases']}")
        print()
        
        return edge_case_summary
    
    def analyze_performance(self):
        """Analyze performance metrics from tests"""
        
        print("⚡ PERFORMANCE ANALYSIS:")
        
        performance_metrics = {
            "financial_simulator": {
                "min_response_time": float('inf'),
                "max_response_time": 0,
                "avg_response_time": 0,
                "total_requests": 0
            },
            "lesson_generator": {
                "min_response_time": float('inf'),
                "max_response_time": 0,
                "avg_response_time": 0,
                "total_requests": 0
            }
        }
        
        # Collect performance data from all test results
        for test_suite, test_data in self.test_results.items():
            if test_data["status"] == "completed" and "results" in test_data:
                results = test_data["results"]
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and "results" in result:
                            for sub_result in result["results"]:
                                if isinstance(sub_result, dict) and "response_time" in sub_result:
                                    response_time = sub_result["response_time"]
                                    
                                    # Determine service based on test type or URL
                                    service = "financial_simulator"  # Default
                                    if "lesson" in str(sub_result).lower():
                                        service = "lesson_generator"
                                    
                                    metrics = performance_metrics[service]
                                    metrics["min_response_time"] = min(metrics["min_response_time"], response_time)
                                    metrics["max_response_time"] = max(metrics["max_response_time"], response_time)
                                    metrics["total_requests"] += 1
        
        # Calculate averages and display results
        for service, metrics in performance_metrics.items():
            if metrics["total_requests"] > 0:
                print(f"   📊 {service.replace('_', ' ').title()}:")
                print(f"      Total Requests: {metrics['total_requests']}")
                print(f"      Min Response Time: {metrics['min_response_time']:.2f}s")
                print(f"      Max Response Time: {metrics['max_response_time']:.2f}s")
                print()
        
        return performance_metrics
    
    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        
        print("💡 RECOMMENDATIONS:")
        
        recommendations = []
        
        # Check for high failure rates
        for test_name, test_data in self.test_results.items():
            if test_data["status"] == "failed":
                recommendations.append(f"🔧 Fix {test_name.replace('_', ' ')} test failures")
        
        # Check for performance issues
        if "bursty_workloads" in self.test_results:
            bursty_data = self.test_results["bursty_workloads"]
            if bursty_data["status"] == "completed":
                for result in bursty_data["results"]:
                    if isinstance(result, dict) and result.get("success_rate", 1) < 0.8:
                        recommendations.append("🚀 Implement load balancing for concurrent requests")
                        break
        
        # Check for timeout issues
        if "high_latency" in self.test_results:
            latency_data = self.test_results["high_latency"]
            if latency_data["status"] == "completed":
                for result in latency_data["results"]:
                    if isinstance(result, dict) and result.get("status") == "timeout":
                        recommendations.append("⏰ Implement better timeout handling and user communication")
                        break
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "✅ System is performing well under edge case conditions",
                "🔄 Continue regular monitoring and testing",
                "📈 Consider stress testing with higher loads"
            ]
        
        for i, recommendation in enumerate(recommendations, 1):
            print(f"   {i}. {recommendation}")
        
        print()
        
        return recommendations
    
    def generate_monitoring_dashboard(self):
        """Generate monitoring dashboard data"""
        
        print("📊 GENERATING MONITORING DASHBOARD...")
        
        dashboard_data = self.monitor.get_dashboard_data()
        dashboard_data["edge_case_test_results"] = self.test_results
        dashboard_data["test_execution_time"] = self.end_time - self.start_time
        
        # Save dashboard data
        with open("edge_case_monitoring_dashboard.json", "w") as f:
            json.dump(dashboard_data, f, indent=2)
        
        print("✅ Dashboard data saved to: edge_case_monitoring_dashboard.json")
        print()
    
    def save_detailed_report(self):
        """Save detailed test report to file"""
        
        report_data = {
            "test_execution": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_time": self.end_time - self.start_time,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": self.test_results,
            "summary": {
                "total_test_suites": len(self.test_results),
                "completed_tests": sum(1 for result in self.test_results.values() if result["status"] == "completed"),
                "failed_tests": sum(1 for result in self.test_results.values() if result["status"] == "failed")
            }
        }
        
        filename = f"edge_case_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Detailed report saved to: {filename}")
        print()

def main():
    """Main function to run all edge case tests"""
    
    # Check if services are running
    print("🔍 Checking service availability...")
    
    import requests
    services_ok = True
    
    try:
        financial_response = requests.get("http://localhost:8002/docs", timeout=5)
        if financial_response.status_code != 200:
            print("❌ Financial Simulator service not available")
            services_ok = False
        else:
            print("✅ Financial Simulator service available")
    except:
        print("❌ Financial Simulator service not accessible")
        services_ok = False
    
    try:
        lesson_response = requests.get("http://localhost:8000/docs", timeout=5)
        if lesson_response.status_code != 200:
            print("❌ Lesson Generator service not available")
            services_ok = False
        else:
            print("✅ Lesson Generator service available")
    except:
        print("❌ Lesson Generator service not accessible")
        services_ok = False
    
    if not services_ok:
        print("\n⚠️  Some services are not available. Tests may fail.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    print("\n" + "="*70)
    
    # Run all tests
    runner = EdgeCaseTestRunner()
    results = runner.run_all_tests()
    
    print("🎉 ALL EDGE CASE TESTS COMPLETED!")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    main()
