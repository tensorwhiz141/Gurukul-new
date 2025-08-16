#!/usr/bin/env python3
"""
BHIV Core Integration Test Script
Comprehensive integration testing for the complete BHIV ecosystem

This script tests:
1. Component initialization
2. End-to-end request processing
3. Performance benchmarks
4. Error handling and recovery
5. System health checks
"""

import asyncio
import json
import time
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import statistics

# Import BHIV components
from bhiv_core_schema import (
    create_bhiv_request, ClassificationType, UrgencyLevel, 
    InputMode, UserPersona, OutputFormat
)
from bhiv_core_orchestrator import BHIVCoreOrchestrator
from bhiv_session_manager import BHIVSessionManager
from bhiv_api_gateway import BHIVAPIGateway

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BHIVIntegrationTester:
    """Comprehensive integration tester for BHIV Core"""
    
    def __init__(self):
        self.orchestrator = None
        self.session_manager = None
        self.api_gateway = None
        self.test_results = {
            "initialization": {},
            "functionality": {},
            "performance": {},
            "error_handling": {},
            "health_checks": {}
        }
        
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🧠 Starting BHIV Core Integration Tests")
        logger.info("=" * 60)
        
        try:
            # Test 1: Component Initialization
            await self.test_component_initialization()
            
            # Test 2: Basic Functionality
            await self.test_basic_functionality()
            
            # Test 3: Advanced Features
            await self.test_advanced_features()
            
            # Test 4: Performance Benchmarks
            await self.test_performance_benchmarks()
            
            # Test 5: Error Handling
            await self.test_error_handling()
            
            # Test 6: Health Checks
            await self.test_health_checks()
            
            # Generate Report
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            raise
        
        logger.info("✅ All BHIV Core Integration Tests Completed")
    
    async def test_component_initialization(self):
        """Test initialization of all BHIV components"""
        logger.info("🔧 Testing Component Initialization...")
        
        start_time = time.time()
        
        try:
            # Initialize BHIV Core Orchestrator
            self.orchestrator = BHIVCoreOrchestrator()
            await self.orchestrator.initialize()
            logger.info("✅ BHIV Core Orchestrator initialized")
            
            # Initialize Session Manager
            self.session_manager = BHIVSessionManager(":memory:")  # Use in-memory DB for testing
            await self.session_manager.initialize()
            logger.info("✅ Session Manager initialized")
            
            # Initialize API Gateway
            self.api_gateway = BHIVAPIGateway()
            logger.info("✅ API Gateway initialized")
            
            initialization_time = time.time() - start_time
            
            self.test_results["initialization"] = {
                "status": "success",
                "time_seconds": initialization_time,
                "components_initialized": ["orchestrator", "session_manager", "api_gateway"]
            }
            
            logger.info(f"✅ Component initialization completed in {initialization_time:.2f}s")
            
        except Exception as e:
            self.test_results["initialization"] = {
                "status": "failed",
                "error": str(e),
                "time_seconds": time.time() - start_time
            }
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    async def test_basic_functionality(self):
        """Test basic BHIV functionality"""
        logger.info("🔍 Testing Basic Functionality...")
        
        test_cases = [
            {
                "name": "Educational Query",
                "query": "How do I learn machine learning?",
                "expected_classification": ClassificationType.LEARNING_QUERY,
                "expected_agent": "EduMentor"
            },
            {
                "name": "Wellness Query", 
                "query": "I'm feeling stressed and anxious",
                "expected_classification": ClassificationType.WELLNESS_QUERY,
                "expected_agent": "WellnessBot"
            },
            {
                "name": "Spiritual Query",
                "query": "What do the Vedas say about the meaning of life?",
                "expected_classification": ClassificationType.SPIRITUAL_QUERY,
                "expected_agent": "GuruAgent"
            },
            {
                "name": "General Query",
                "query": "Hello, how are you today?",
                "expected_classification": ClassificationType.GENERAL_QUERY,
                "expected_agent": "GeneralBot"
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            try:
                logger.info(f"  Testing: {test_case['name']}")
                
                # Create BHIV request
                request = create_bhiv_request(
                    user_id="integration_test_user",
                    raw_text=test_case["query"],
                    voice_enabled=False,
                    mode=InputMode.CHAT,
                    persona=UserPersona.STUDENT,
                    lang="en-IN"
                )
                
                # Process request
                start_time = time.time()
                response = await self.orchestrator.process_bhiv_request(request)
                processing_time = time.time() - start_time
                
                # Validate response
                success = (
                    response.agent_directives.classification == test_case["expected_classification"] and
                    response.agent_directives.agent_name == test_case["expected_agent"] and
                    response.output.response_text is not None and
                    len(response.output.response_text) > 0 and
                    not response.logs.error_flag
                )
                
                result = {
                    "name": test_case["name"],
                    "success": success,
                    "processing_time": processing_time,
                    "classification": response.agent_directives.classification.value,
                    "agent": response.agent_directives.agent_name,
                    "confidence": response.output.confidence_score,
                    "response_length": len(response.output.response_text)
                }
                
                results.append(result)
                
                if success:
                    logger.info(f"    ✅ {test_case['name']} - {processing_time:.2f}s")
                else:
                    logger.warning(f"    ⚠️ {test_case['name']} - Unexpected result")
                
            except Exception as e:
                logger.error(f"    ❌ {test_case['name']} - Error: {e}")
                results.append({
                    "name": test_case["name"],
                    "success": False,
                    "error": str(e)
                })
        
        self.test_results["functionality"] = {
            "total_tests": len(test_cases),
            "successful_tests": sum(1 for r in results if r.get("success", False)),
            "results": results
        }
        
        success_rate = (self.test_results["functionality"]["successful_tests"] / len(test_cases)) * 100
        logger.info(f"✅ Basic functionality tests completed - {success_rate:.1f}% success rate")
    
    async def test_advanced_features(self):
        """Test advanced BHIV features"""
        logger.info("🚀 Testing Advanced Features...")
        
        advanced_tests = []
        
        # Test 1: Voice-enabled request
        try:
            logger.info("  Testing voice-enabled request...")
            request = create_bhiv_request(
                user_id="integration_test_user",
                raw_text="Explain quantum computing",
                voice_enabled=True,
                mode=InputMode.VOICE,
                persona=UserPersona.STUDENT
            )
            
            response = await self.orchestrator.process_bhiv_request(request)
            
            voice_test_success = (
                response.input.voice_enabled is True and
                not response.logs.error_flag
            )
            
            advanced_tests.append({
                "name": "Voice-enabled request",
                "success": voice_test_success,
                "voice_url": response.output.voice_response_url
            })
            
            logger.info(f"    ✅ Voice-enabled request - Success: {voice_test_success}")
            
        except Exception as e:
            logger.error(f"    ❌ Voice-enabled request failed: {e}")
            advanced_tests.append({
                "name": "Voice-enabled request",
                "success": False,
                "error": str(e)
            })
        
        # Test 2: Session management
        try:
            logger.info("  Testing session management...")
            
            from bhiv_core_schema import UserInfo
            user_info = UserInfo(
                user_id="session_test_user",
                auth_token="test_token",
                lang="en-IN",
                persona=UserPersona.STUDENT
            )
            
            # Start session
            session_id = await self.session_manager.start_session(user_info)
            
            # Get session info
            session_info = await self.session_manager.get_session_info(session_id)
            
            # End session
            await self.session_manager.end_session(session_id)
            
            session_test_success = (
                session_id is not None and
                session_info is not None and
                session_info["user_info"]["user_id"] == "session_test_user"
            )
            
            advanced_tests.append({
                "name": "Session management",
                "success": session_test_success,
                "session_id": session_id
            })
            
            logger.info(f"    ✅ Session management - Success: {session_test_success}")
            
        except Exception as e:
            logger.error(f"    ❌ Session management failed: {e}")
            advanced_tests.append({
                "name": "Session management",
                "success": False,
                "error": str(e)
            })
        
        # Test 3: API Gateway authentication
        try:
            logger.info("  Testing API Gateway authentication...")
            
            # Test API key verification
            api_key_info = self.api_gateway.auth_manager.verify_api_key("bhiv-core-key")
            
            # Test permission checking
            has_permission = self.api_gateway.auth_manager.check_permissions(
                ["read", "write"], "read"
            )
            
            auth_test_success = (
                api_key_info is not None and
                api_key_info["name"] == "BHIV Core Service" and
                has_permission is True
            )
            
            advanced_tests.append({
                "name": "API Gateway authentication",
                "success": auth_test_success,
                "api_key_valid": api_key_info is not None,
                "permissions_work": has_permission
            })
            
            logger.info(f"    ✅ API Gateway authentication - Success: {auth_test_success}")
            
        except Exception as e:
            logger.error(f"    ❌ API Gateway authentication failed: {e}")
            advanced_tests.append({
                "name": "API Gateway authentication",
                "success": False,
                "error": str(e)
            })
        
        self.test_results["advanced_features"] = {
            "total_tests": len(advanced_tests),
            "successful_tests": sum(1 for t in advanced_tests if t.get("success", False)),
            "results": advanced_tests
        }
        
        success_rate = (self.test_results["advanced_features"]["successful_tests"] / len(advanced_tests)) * 100
        logger.info(f"✅ Advanced features tests completed - {success_rate:.1f}% success rate")
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        logger.info("⚡ Testing Performance Benchmarks...")
        
        # Performance test: Process multiple requests
        num_requests = 10
        processing_times = []
        
        logger.info(f"  Processing {num_requests} concurrent requests...")
        
        async def process_single_request(i):
            request = create_bhiv_request(
                user_id=f"perf_test_user_{i}",
                raw_text=f"Performance test query {i}",
                voice_enabled=False
            )
            
            start_time = time.time()
            response = await self.orchestrator.process_bhiv_request(request)
            processing_time = time.time() - start_time
            
            return {
                "request_id": i,
                "processing_time": processing_time,
                "success": not response.logs.error_flag,
                "confidence": response.output.confidence_score
            }
        
        # Run concurrent requests
        start_time = time.time()
        tasks = [process_single_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        processing_times = [r["processing_time"] for r in successful_results]
        
        if processing_times:
            avg_processing_time = statistics.mean(processing_times)
            median_processing_time = statistics.median(processing_times)
            max_processing_time = max(processing_times)
            min_processing_time = min(processing_times)
        else:
            avg_processing_time = median_processing_time = max_processing_time = min_processing_time = 0
        
        throughput = len(successful_results) / total_time if total_time > 0 else 0
        
        self.test_results["performance"] = {
            "total_requests": num_requests,
            "successful_requests": len(successful_results),
            "total_time": total_time,
            "avg_processing_time": avg_processing_time,
            "median_processing_time": median_processing_time,
            "max_processing_time": max_processing_time,
            "min_processing_time": min_processing_time,
            "throughput_rps": throughput,
            "success_rate": (len(successful_results) / num_requests) * 100
        }
        
        logger.info(f"  ✅ Processed {len(successful_results)}/{num_requests} requests")
        logger.info(f"  ⚡ Average processing time: {avg_processing_time:.2f}s")
        logger.info(f"  📊 Throughput: {throughput:.2f} requests/second")
    
    async def test_error_handling(self):
        """Test error handling and recovery"""
        logger.info("🛡️ Testing Error Handling...")
        
        error_tests = []
        
        # Test 1: Invalid user input
        try:
            request = create_bhiv_request(
                user_id="error_test_user",
                raw_text="",  # Empty query
                voice_enabled=False
            )
            
            response = await self.orchestrator.process_bhiv_request(request)
            
            # Should handle gracefully
            error_tests.append({
                "name": "Empty query handling",
                "success": True,  # Should not crash
                "handled_gracefully": True
            })
            
        except Exception as e:
            error_tests.append({
                "name": "Empty query handling",
                "success": False,
                "error": str(e)
            })
        
        self.test_results["error_handling"] = {
            "total_tests": len(error_tests),
            "successful_tests": sum(1 for t in error_tests if t.get("success", False)),
            "results": error_tests
        }
        
        logger.info("✅ Error handling tests completed")
    
    async def test_health_checks(self):
        """Test system health checks"""
        logger.info("❤️ Testing Health Checks...")
        
        health_results = {}
        
        # Test orchestrator health
        try:
            orchestrator_health = await self.orchestrator.health_check()
            health_results["orchestrator"] = {
                "status": orchestrator_health.get("status", "unknown"),
                "components": orchestrator_health.get("components", {})
            }
        except Exception as e:
            health_results["orchestrator"] = {"status": "error", "error": str(e)}
        
        # Test session manager health
        try:
            session_stats = self.session_manager.get_system_stats()
            health_results["session_manager"] = {
                "status": "healthy",
                "stats": session_stats
            }
        except Exception as e:
            health_results["session_manager"] = {"status": "error", "error": str(e)}
        
        # Test API gateway health
        try:
            gateway_stats = self.api_gateway.get_gateway_stats()
            health_results["api_gateway"] = {
                "status": "healthy",
                "stats": gateway_stats
            }
        except Exception as e:
            health_results["api_gateway"] = {"status": "error", "error": str(e)}
        
        self.test_results["health_checks"] = health_results
        
        healthy_components = sum(1 for h in health_results.values() if h.get("status") == "healthy")
        total_components = len(health_results)
        
        logger.info(f"✅ Health checks completed - {healthy_components}/{total_components} components healthy")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 Generating Test Report...")
        logger.info("=" * 60)
        
        # Summary
        total_tests = (
            self.test_results["functionality"].get("total_tests", 0) +
            self.test_results["advanced_features"].get("total_tests", 0) +
            self.test_results["error_handling"].get("total_tests", 0)
        )
        
        successful_tests = (
            self.test_results["functionality"].get("successful_tests", 0) +
            self.test_results["advanced_features"].get("successful_tests", 0) +
            self.test_results["error_handling"].get("successful_tests", 0)
        )
        
        overall_success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"📈 BHIV Core Integration Test Report")
        logger.info(f"   Generated: {datetime.now().isoformat()}")
        logger.info(f"   Overall Success Rate: {overall_success_rate:.1f}%")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Successful Tests: {successful_tests}")
        logger.info("")
        
        # Performance Summary
        perf = self.test_results.get("performance", {})
        if perf:
            logger.info(f"⚡ Performance Summary:")
            logger.info(f"   Average Processing Time: {perf.get('avg_processing_time', 0):.2f}s")
            logger.info(f"   Throughput: {perf.get('throughput_rps', 0):.2f} requests/second")
            logger.info(f"   Success Rate: {perf.get('success_rate', 0):.1f}%")
            logger.info("")
        
        # Component Health
        health = self.test_results.get("health_checks", {})
        if health:
            logger.info(f"❤️ Component Health:")
            for component, status in health.items():
                logger.info(f"   {component}: {status.get('status', 'unknown')}")
            logger.info("")
        
        # Save detailed report
        report_filename = f"bhiv_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"📄 Detailed report saved to: {report_filename}")
        logger.info("=" * 60)


async def main():
    """Main integration test function"""
    tester = BHIVIntegrationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
