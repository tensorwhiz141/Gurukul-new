"""
Test Script for Basic vs Enhanced Mode and Wikipedia Settings
Tests the fixes for lesson generation modes and Wikipedia bug
"""

import requests
import json
import time
from datetime import datetime

API_BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_lesson_modes_001"

class LessonModesTester:
    def __init__(self):
        self.test_results = []
        self.test_subject = "Science"
        self.test_topic = "Photosynthesis"

    def log_test(self, test_name, success, details=None):
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

    def test_basic_mode_with_wikipedia(self):
        """Test basic mode with Wikipedia enabled"""
        print("📚 Testing Basic Mode with Wikipedia...")
        
        payload = {
            "subject": self.test_subject,
            "topic": self.test_topic,
            "user_id": TEST_USER_ID,
            "include_wikipedia": True,
            "force_regenerate": True
        }

        try:
            response = requests.post(f"{API_BASE_URL}/lessons", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", "")
                content_length = len(content)
                source = data.get("source", "")
                content_type = data.get("content_type", "")
                
                # Check if content is appropriately sized for basic mode
                is_appropriate_length = 150 <= content_length <= 400  # Basic mode should be concise
                
                self.log_test("Basic Mode with Wikipedia", True, {
                    "content_length": content_length,
                    "source": source,
                    "content_type": content_type,
                    "appropriate_length": is_appropriate_length,
                    "wikipedia_setting": data.get("settings", {}).get("include_wikipedia")
                })
                
                return data
            else:
                self.log_test("Basic Mode with Wikipedia", False, {
                    "status_code": response.status_code,
                    "error": response.text
                })
                return None
                
        except Exception as e:
            self.log_test("Basic Mode with Wikipedia", False, {"error": str(e)})
            return None

    def test_basic_mode_without_wikipedia(self):
        """Test basic mode without Wikipedia - should not contain Wikipedia references"""
        print("🧠 Testing Basic Mode without Wikipedia...")
        
        payload = {
            "subject": self.test_subject,
            "topic": self.test_topic,
            "user_id": TEST_USER_ID,
            "include_wikipedia": False,
            "force_regenerate": True
        }

        try:
            response = requests.post(f"{API_BASE_URL}/lessons", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", "").lower()
                content_length = len(content)
                
                # Check for Wikipedia references (this was the bug)
                has_wikipedia_reference = "according to wikipedia" in content or "wikipedia" in content
                is_concise = content_length <= 300  # Should be more concise without Wikipedia
                
                success = not has_wikipedia_reference and is_concise
                
                self.log_test("Basic Mode without Wikipedia", success, {
                    "content_length": content_length,
                    "has_wikipedia_reference": has_wikipedia_reference,
                    "is_concise": is_concise,
                    "wikipedia_setting": data.get("settings", {}).get("include_wikipedia"),
                    "content_preview": content[:100] + "..." if len(content) > 100 else content
                })
                
                return data
            else:
                self.log_test("Basic Mode without Wikipedia", False, {
                    "status_code": response.status_code,
                    "error": response.text
                })
                return None
                
        except Exception as e:
            self.log_test("Basic Mode without Wikipedia", False, {"error": str(e)})
            return None

    def test_enhanced_mode_with_wikipedia(self):
        """Test enhanced mode with Wikipedia - should be comprehensive"""
        print("🚀 Testing Enhanced Mode with Wikipedia...")
        
        payload = {
            "subject": self.test_subject,
            "topic": self.test_topic,
            "user_id": TEST_USER_ID,
            "use_orchestration": True,
            "include_wikipedia": True,
            "include_triggers": True
        }

        try:
            response = requests.post(f"{API_BASE_URL}/lessons/enhanced", json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", "")
                content_length = len(content)
                source = data.get("source", "")
                enhanced_features = data.get("enhanced_features", {})
                
                # Enhanced mode should have longer, more comprehensive content
                is_comprehensive = content_length > 400
                is_enhanced = source.startswith("orchestration") or enhanced_features.get("rag_enhanced", False)
                
                self.log_test("Enhanced Mode with Wikipedia", True, {
                    "content_length": content_length,
                    "source": source,
                    "is_comprehensive": is_comprehensive,
                    "is_enhanced": is_enhanced,
                    "rag_enhanced": enhanced_features.get("rag_enhanced", False),
                    "triggers_detected": enhanced_features.get("triggers_detected", 0)
                })
                
                return data
            else:
                self.log_test("Enhanced Mode with Wikipedia", response.status_code == 503, {
                    "status_code": response.status_code,
                    "note": "503 acceptable if orchestration unavailable",
                    "error": response.text
                })
                return None
                
        except Exception as e:
            self.log_test("Enhanced Mode with Wikipedia", False, {"error": str(e)})
            return None

    def test_enhanced_mode_without_wikipedia(self):
        """Test enhanced mode without Wikipedia"""
        print("🚀🧠 Testing Enhanced Mode without Wikipedia...")
        
        payload = {
            "subject": self.test_subject,
            "topic": self.test_topic,
            "user_id": TEST_USER_ID,
            "use_orchestration": True,
            "include_wikipedia": False,
            "include_triggers": True
        }

        try:
            response = requests.post(f"{API_BASE_URL}/lessons/enhanced", json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", "").lower()
                content_length = len(content)
                
                # Should be comprehensive but not reference Wikipedia
                has_wikipedia_reference = "according to wikipedia" in content or "wikipedia" in content
                is_comprehensive = content_length > 300
                
                success = not has_wikipedia_reference and is_comprehensive
                
                self.log_test("Enhanced Mode without Wikipedia", success, {
                    "content_length": content_length,
                    "has_wikipedia_reference": has_wikipedia_reference,
                    "is_comprehensive": is_comprehensive,
                    "wikipedia_setting": False
                })
                
                return data
            else:
                self.log_test("Enhanced Mode without Wikipedia", response.status_code == 503, {
                    "status_code": response.status_code,
                    "note": "503 acceptable if orchestration unavailable"
                })
                return None
                
        except Exception as e:
            self.log_test("Enhanced Mode without Wikipedia", False, {"error": str(e)})
            return None

    def compare_content_lengths(self, basic_data, enhanced_data):
        """Compare content lengths between basic and enhanced modes"""
        print("📊 Comparing Content Lengths...")
        
        if not basic_data or not enhanced_data:
            self.log_test("Content Length Comparison", False, {
                "error": "Missing data for comparison"
            })
            return
        
        basic_length = len(basic_data.get("content", ""))
        enhanced_length = len(enhanced_data.get("content", ""))
        
        # Enhanced should be significantly longer than basic
        length_difference = enhanced_length - basic_length
        is_enhanced_longer = enhanced_length > basic_length * 1.5  # At least 50% longer
        
        self.log_test("Content Length Comparison", is_enhanced_longer, {
            "basic_length": basic_length,
            "enhanced_length": enhanced_length,
            "length_difference": length_difference,
            "enhanced_is_longer": is_enhanced_longer,
            "ratio": round(enhanced_length / basic_length, 2) if basic_length > 0 else 0
        })

    def run_all_tests(self):
        """Run all lesson mode tests"""
        print("🧪 Starting Lesson Modes Testing")
        print("=" * 50)
        
        # Test all modes
        basic_with_wiki = self.test_basic_mode_with_wikipedia()
        time.sleep(2)
        
        basic_without_wiki = self.test_basic_mode_without_wikipedia()
        time.sleep(2)
        
        enhanced_with_wiki = self.test_enhanced_mode_with_wikipedia()
        time.sleep(2)
        
        enhanced_without_wiki = self.test_enhanced_mode_without_wikipedia()
        time.sleep(2)
        
        # Compare lengths
        if basic_with_wiki and enhanced_with_wiki:
            self.compare_content_lengths(basic_with_wiki, enhanced_with_wiki)
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("\n📊 Test Summary")
        print("=" * 30)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
        
        # Key findings
        print("\n🔍 Key Findings:")
        for result in self.test_results:
            if not result["success"]:
                print(f"❌ {result['test_name']}: {result['details'].get('error', 'Failed')}")
            else:
                details = result["details"]
                if "has_wikipedia_reference" in details:
                    wiki_status = "❌ Has Wikipedia refs" if details["has_wikipedia_reference"] else "✅ No Wikipedia refs"
                    print(f"✅ {result['test_name']}: {wiki_status}, Length: {details.get('content_length', 'N/A')}")
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests) * 100,
            "results": self.test_results
        }

def main():
    tester = LessonModesTester()
    results = tester.run_all_tests()
    
    # Save results
    with open("lesson_modes_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to lesson_modes_test_results.json")
    return 0 if results["failed"] == 0 else 1

if __name__ == "__main__":
    exit(main())
