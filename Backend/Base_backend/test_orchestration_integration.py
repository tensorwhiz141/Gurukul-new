"""
Test Suite for Base_backend + Orchestration Integration
Tests the enhanced lesson generation and trigger detection functionality
"""

import asyncio
import json
import requests
import time
from datetime import datetime
from typing import Dict, Any, List

# Test Configuration
BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_user_integration_001"

class OrchestrationIntegrationTester:
    """Comprehensive test suite for orchestration integration"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.test_results = []
        self.session = requests.Session()
    
    def log_test(self, test_name: str, success: bool, details: Dict[str, Any] = None):
        """Log test results"""
        result = {
            "test_name": test_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details and not success:
            print(f"   Details: {details}")
    
    def test_health_check(self) -> bool:
        """Test basic health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                orchestration_status = data.get("orchestration", "unknown")
                self.log_test("Health Check", success, {
                    "orchestration_status": orchestration_status,
                    "response_time": response.elapsed.total_seconds()
                })
            else:
                self.log_test("Health Check", success, {
                    "status_code": response.status_code,
                    "error": response.text
                })
            
            return success
            
        except Exception as e:
            self.log_test("Health Check", False, {"error": str(e)})
            return False
    
    def test_integration_status(self) -> bool:
        """Test integration status endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/integration-status")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                integration_valid = data.get("integration_status", {}).get("overall_valid", False)
                orchestration_initialized = data.get("runtime_status", {}).get("orchestration_engine_initialized", False)
                
                self.log_test("Integration Status", success, {
                    "integration_valid": integration_valid,
                    "orchestration_initialized": orchestration_initialized,
                    "sub_agent_urls": data.get("runtime_status", {}).get("sub_agent_urls", {})
                })
            else:
                self.log_test("Integration Status", success, {
                    "status_code": response.status_code,
                    "error": response.text
                })
            
            return success
            
        except Exception as e:
            self.log_test("Integration Status", False, {"error": str(e)})
            return False
    
    def test_basic_lesson_generation(self) -> bool:
        """Test basic lesson generation (fallback)"""
        try:
            payload = {
                "subject": "Mathematics",
                "topic": "Triangles",
                "user_id": TEST_USER_ID,
                "use_orchestration": False
            }
            
            response = self.session.post(f"{self.base_url}/lessons/enhanced", json=payload)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                has_content = bool(data.get("content"))
                source = data.get("source", "unknown")
                
                self.log_test("Basic Lesson Generation", success, {
                    "has_content": has_content,
                    "source": source,
                    "subject": data.get("subject"),
                    "topic": data.get("topic"),
                    "content_length": len(data.get("content", ""))
                })
            else:
                self.log_test("Basic Lesson Generation", success, {
                    "status_code": response.status_code,
                    "error": response.text
                })
            
            return success
            
        except Exception as e:
            self.log_test("Basic Lesson Generation", False, {"error": str(e)})
            return False
    
    def test_enhanced_lesson_generation(self) -> bool:
        """Test enhanced lesson generation with orchestration"""
        try:
            payload = {
                "subject": "Science",
                "topic": "Photosynthesis",
                "user_id": TEST_USER_ID,
                "use_orchestration": True,
                "quiz_score": 45.0  # Low score to trigger interventions
            }
            
            response = self.session.post(f"{self.base_url}/lessons/enhanced", json=payload)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                has_enhanced_features = bool(data.get("enhanced_features"))
                has_orchestration_data = bool(data.get("orchestration_data"))
                source = data.get("source", "unknown")
                
                self.log_test("Enhanced Lesson Generation", success, {
                    "has_enhanced_features": has_enhanced_features,
                    "has_orchestration_data": has_orchestration_data,
                    "source": source,
                    "triggers_detected": data.get("enhanced_features", {}).get("triggers_detected", 0),
                    "interventions_applied": data.get("enhanced_features", {}).get("interventions_applied", 0)
                })
            else:
                self.log_test("Enhanced Lesson Generation", success, {
                    "status_code": response.status_code,
                    "error": response.text
                })
            
            return success
            
        except Exception as e:
            self.log_test("Enhanced Lesson Generation", False, {"error": str(e)})
            return False
    
    def test_user_progress_tracking(self) -> bool:
        """Test user progress tracking"""
        try:
            response = self.session.get(f"{self.base_url}/user-progress/{TEST_USER_ID}")
            success = response.status_code == 200 or response.status_code == 503  # 503 if orchestration unavailable
            
            if response.status_code == 200:
                data = response.json()
                has_progress = "educational_progress" in data
                has_recommendations = "recommendations" in data
                
                self.log_test("User Progress Tracking", success, {
                    "has_progress": has_progress,
                    "has_recommendations": has_recommendations,
                    "interaction_count": data.get("interaction_count", 0),
                    "quiz_scores": data.get("quiz_scores", []),
                    "triggers_detected": len(data.get("triggers_detected", []))
                })
            elif response.status_code == 503:
                self.log_test("User Progress Tracking", True, {
                    "status": "orchestration_unavailable",
                    "expected_behavior": "503_service_unavailable"
                })
            else:
                self.log_test("User Progress Tracking", False, {
                    "status_code": response.status_code,
                    "error": response.text
                })
            
            return success
            
        except Exception as e:
            self.log_test("User Progress Tracking", False, {"error": str(e)})
            return False
    
    def test_trigger_intervention(self) -> bool:
        """Test manual trigger intervention"""
        try:
            # Test with low quiz score to trigger intervention
            response = self.session.post(
                f"{self.base_url}/trigger-intervention/{TEST_USER_ID}",
                params={"quiz_score": 35.0}
            )
            success = response.status_code == 200 or response.status_code == 503
            
            if response.status_code == 200:
                data = response.json()
                has_interventions = len(data.get("interventions", [])) > 0
                has_triggers = len(data.get("triggers_detected", [])) > 0
                
                self.log_test("Trigger Intervention", success, {
                    "has_interventions": has_interventions,
                    "has_triggers": has_triggers,
                    "interventions_count": len(data.get("interventions", [])),
                    "triggers_count": len(data.get("triggers_detected", []))
                })
            elif response.status_code == 503:
                self.log_test("Trigger Intervention", True, {
                    "status": "orchestration_unavailable",
                    "expected_behavior": "503_service_unavailable"
                })
            else:
                self.log_test("Trigger Intervention", False, {
                    "status_code": response.status_code,
                    "error": response.text
                })
            
            return success
            
        except Exception as e:
            self.log_test("Trigger Intervention", False, {"error": str(e)})
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        print("🧪 Starting Orchestration Integration Tests")
        print("=" * 60)
        
        # Run tests in order
        tests = [
            self.test_health_check,
            self.test_integration_status,
            self.test_basic_lesson_generation,
            self.test_enhanced_lesson_generation,
            self.test_user_progress_tracking,
            self.test_trigger_intervention
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                print(f"❌ Test {test.__name__} crashed: {e}")
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "test_results": self.test_results,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n📊 Test Summary")
        print("=" * 30)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        return summary

def main():
    """Run the integration tests"""
    tester = OrchestrationIntegrationTester()
    results = tester.run_all_tests()
    
    # Save results to file
    with open("integration_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to integration_test_results.json")
    
    # Return exit code based on success
    return 0 if results["failed"] == 0 else 1

if __name__ == "__main__":
    exit(main())
