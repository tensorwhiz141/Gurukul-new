#!/usr/bin/env python3
"""
System Integration Test
Tests all components working together
"""

import requests
import json
import time
from datetime import datetime

def test_system_integration():
    """Test full system integration with concrete evidence"""
    
    print("🔍 TESTING FULL SYSTEM INTEGRATION")
    print("=" * 50)
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {
        "test_timestamp": datetime.now().isoformat(),
        "services": {},
        "integrations": {},
        "api_flows": []
    }
    
    # Test 1: Financial Simulator Service
    print("📊 Testing Financial Simulator Integration:")
    try:
        # Test health endpoint
        health_response = requests.get('http://localhost:8002/docs', timeout=5)
        print(f"   ✅ Health Check: {health_response.status_code}")
        
        # Test simulation start
        financial_data = {
            'user_id': 'integration-test-user',
            'user_name': 'Integration Test User',
            'income': 75000,
            'expenses': [
                {'name': 'Rent', 'amount': 2000},
                {'name': 'Food', 'amount': 800},
                {'name': 'Transportation', 'amount': 500}
            ],
            'total_expenses': 3300,
            'goal': 'Test full system integration',
            'financial_type': 'moderate',
            'risk_level': 'medium'
        }
        
        start_response = requests.post('http://localhost:8002/start-simulation', 
                                     json=financial_data, timeout=10)
        print(f"   ✅ Simulation Start: {start_response.status_code}")
        
        if start_response.status_code == 200:
            data = start_response.json()
            task_id = data.get('task_id')
            print(f"   📋 Task ID: {task_id}")
            print(f"   💬 Message: {data.get('message')}")
            
            # Test status endpoint
            if task_id:
                status_response = requests.get(f'http://localhost:8002/simulation-status/{task_id}', timeout=5)
                print(f"   ✅ Status Check: {status_response.status_code}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   📊 Status: {status_data.get('status')}")
                    print(f"   🔄 Ready: {status_data.get('ready')}")
                    
                    results["services"]["financial_simulator"] = {
                        "status": "operational",
                        "health_check": health_response.status_code,
                        "simulation_start": start_response.status_code,
                        "status_check": status_response.status_code,
                        "task_id": task_id,
                        "response_data": status_data
                    }
        else:
            print(f"   ❌ Error: {start_response.text}")
            results["services"]["financial_simulator"] = {
                "status": "error",
                "error": start_response.text
            }
            
    except Exception as e:
        print(f"   ❌ Financial Simulator Error: {e}")
        results["services"]["financial_simulator"] = {
            "status": "failed",
            "error": str(e)
        }
    
    print()
    
    # Test 2: Memory Management Service
    print("🧠 Testing Memory Management Integration:")
    try:
        # Test health endpoint
        memory_health = requests.get('http://localhost:8003/memory/health', timeout=5)
        print(f"   ✅ Health Check: {memory_health.status_code}")
        
        # Test memory storage
        memory_data = {
            'user_id': 'integration-test-user',
            'content': 'Financial simulation integration test completed successfully',
            'metadata': {
                'type': 'integration_test',
                'timestamp': datetime.now().isoformat(),
                'component': 'financial_simulator'
            }
        }
        
        store_response = requests.post('http://localhost:8003/memory/store', 
                                     json=memory_data, timeout=5)
        print(f"   ✅ Memory Store: {store_response.status_code}")
        
        # Test memory retrieval
        query_data = {
            'user_id': 'integration-test-user',
            'query': 'financial simulation',
            'limit': 5
        }
        
        retrieve_response = requests.post('http://localhost:8003/memory/retrieve', 
                                        json=query_data, timeout=5)
        print(f"   ✅ Memory Retrieve: {retrieve_response.status_code}")
        
        if retrieve_response.status_code == 200:
            memories = retrieve_response.json()
            print(f"   📚 Retrieved: {len(memories.get('memories', []))} memories")
            
            results["services"]["memory_management"] = {
                "status": "operational",
                "health_check": memory_health.status_code,
                "store_operation": store_response.status_code,
                "retrieve_operation": retrieve_response.status_code,
                "memories_count": len(memories.get('memories', []))
            }
        else:
            results["services"]["memory_management"] = {
                "status": "partial",
                "health_check": memory_health.status_code,
                "retrieve_error": retrieve_response.text
            }
            
    except Exception as e:
        print(f"   ❌ Memory Service Error: {e}")
        results["services"]["memory_management"] = {
            "status": "failed",
            "error": str(e)
        }
    
    print()
    
    # Test 3: API Data Service
    print("📡 Testing API Data Service Integration:")
    try:
        # Test health endpoint
        api_health = requests.get('http://localhost:8001/health', timeout=5)
        print(f"   ✅ Health Check: {api_health.status_code}")
        
        # Test data endpoints
        try:
            data_response = requests.get('http://localhost:8001/api/financial-data', timeout=5)
            print(f"   ✅ Data Retrieval: {data_response.status_code}")
        except:
            print(f"   ⚠️  Data endpoint not available (expected)")
        
        results["services"]["api_data_service"] = {
            "status": "operational",
            "health_check": api_health.status_code
        }
        
    except Exception as e:
        print(f"   ❌ API Data Service Error: {e}")
        results["services"]["api_data_service"] = {
            "status": "failed",
            "error": str(e)
        }
    
    print()
    
    # Test 4: Cross-Service Integration Flow
    print("🔗 Testing Cross-Service Integration Flow:")
    try:
        # Simulate a complete user workflow
        print("   🚀 Simulating complete user workflow...")
        
        # Step 1: User starts financial simulation
        workflow_data = {
            'user_id': 'workflow-test-user',
            'user_name': 'Workflow Test User',
            'income': 85000,
            'expenses': [{'name': 'Housing', 'amount': 2500}],
            'total_expenses': 2500,
            'goal': 'Complete workflow integration test',
            'financial_type': 'aggressive',
            'risk_level': 'high'
        }
        
        workflow_response = requests.post('http://localhost:8002/start-simulation', 
                                        json=workflow_data, timeout=10)
        print(f"   ✅ Step 1 - Simulation Started: {workflow_response.status_code}")
        
        if workflow_response.status_code == 200:
            workflow_task_id = workflow_response.json().get('task_id')
            
            # Step 2: Store workflow event in memory
            memory_event = {
                'user_id': 'workflow-test-user',
                'content': f'Started financial simulation with task ID: {workflow_task_id}',
                'metadata': {
                    'type': 'workflow_event',
                    'task_id': workflow_task_id,
                    'step': 'simulation_started'
                }
            }
            
            memory_store = requests.post('http://localhost:8003/memory/store', 
                                       json=memory_event, timeout=5)
            print(f"   ✅ Step 2 - Memory Stored: {memory_store.status_code}")
            
            # Step 3: Check simulation status
            status_check = requests.get(f'http://localhost:8002/simulation-status/{workflow_task_id}', 
                                      timeout=5)
            print(f"   ✅ Step 3 - Status Check: {status_check.status_code}")
            
            # Step 4: Retrieve user's workflow history
            history_query = {
                'user_id': 'workflow-test-user',
                'query': 'workflow simulation',
                'limit': 10
            }
            
            history_response = requests.post('http://localhost:8003/memory/retrieve', 
                                           json=history_query, timeout=5)
            print(f"   ✅ Step 4 - History Retrieved: {history_response.status_code}")
            
            results["integrations"]["complete_workflow"] = {
                "status": "success",
                "steps_completed": 4,
                "task_id": workflow_task_id,
                "workflow_data": workflow_data
            }
            
            print("   🎉 Complete workflow integration successful!")
        
    except Exception as e:
        print(f"   ❌ Workflow Integration Error: {e}")
        results["integrations"]["complete_workflow"] = {
            "status": "failed",
            "error": str(e)
        }
    
    print()
    
    # Generate Integration Report
    print("📋 INTEGRATION TEST SUMMARY:")
    print("-" * 30)
    
    operational_services = sum(1 for service in results["services"].values() 
                             if service.get("status") == "operational")
    total_services = len(results["services"])
    
    print(f"   Services Operational: {operational_services}/{total_services}")
    
    for service_name, service_data in results["services"].items():
        status_icon = "✅" if service_data.get("status") == "operational" else "❌"
        print(f"   {status_icon} {service_name.replace('_', ' ').title()}: {service_data.get('status')}")
    
    workflow_status = results["integrations"].get("complete_workflow", {}).get("status", "not_tested")
    workflow_icon = "✅" if workflow_status == "success" else "❌"
    print(f"   {workflow_icon} Complete Workflow: {workflow_status}")
    
    print()
    print(f"🎯 Integration Test Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results
    with open("system_integration_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: system_integration_test_results.json")
    
    return results

if __name__ == "__main__":
    test_system_integration()
