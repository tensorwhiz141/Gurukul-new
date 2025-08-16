#!/usr/bin/env python3
"""
Demo Financial Simulation
This script starts a real financial simulation using the correct API format
"""

import requests
import json
import time
import sys
import uuid

# Server configuration
SERVER_URL = "http://localhost:8002"

def start_financial_simulation():
    """Start a financial simulation with the correct API format"""
    print("🚀 Starting Financial Agent Simulation...")
    print("💰 Using Groq AI and MongoDB Atlas for real financial analysis")
    
    # Create simulation data in the exact format expected by the API
    simulation_data = {
        "user_id": f"demo_user_{str(uuid.uuid4())[:8]}",
        "user_name": "Demo Financial User",
        "income": 75000.0,  # Annual income
        "expenses": [
            {"name": "rent", "amount": 1500.0},
            {"name": "groceries", "amount": 600.0},
            {"name": "utilities", "amount": 200.0},
            {"name": "transportation", "amount": 400.0},
            {"name": "healthcare", "amount": 300.0},
            {"name": "entertainment", "amount": 250.0}
        ],
        "total_expenses": 3250.0,  # Monthly total
        "goal": "Build wealth for retirement and emergency fund",
        "financial_type": "Conservative Investor",
        "risk_level": "moderate"
    }
    
    print(f"👤 User: {simulation_data['user_name']}")
    print(f"💰 Monthly Income: ${simulation_data['income']/12:,.2f}")
    print(f"💸 Monthly Expenses: ${simulation_data['total_expenses']:,.2f}")
    print(f"💵 Monthly Savings: ${(simulation_data['income']/12) - simulation_data['total_expenses']:,.2f}")
    print(f"🎯 Goal: {simulation_data['goal']}")
    print(f"📊 Risk Level: {simulation_data['risk_level']}")
    
    try:
        print("\n📡 Sending simulation request to backend...")
        response = requests.post(
            f"{SERVER_URL}/start-simulation",
            json=simulation_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            print(f"✅ Simulation started successfully!")
            print(f"📋 Task ID: {task_id}")
            print(f"📊 Status: {result.get('status')}")
            print(f"📝 Message: {result.get('message')}")
            
            return task_id
        else:
            print(f"❌ Failed to start simulation: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def monitor_simulation_progress(task_id):
    """Monitor the progress of a running simulation"""
    print(f"\n🔍 Monitoring simulation progress for task: {task_id}")
    print("⏳ This may take several minutes as the AI agents analyze your financial situation...")
    
    max_attempts = 120  # Maximum 10 minutes of monitoring
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Check simulation status
            response = requests.get(
                f"{SERVER_URL}/simulation-status/{task_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                status_data = response.json()
                task_status = status_data.get("task_status", "unknown")
                task_details = status_data.get("task_details", {})
                
                print(f"📊 Status: {task_status} | Attempt: {attempt + 1}/{max_attempts}")
                
                if task_status == "completed":
                    print("🎉 Simulation completed successfully!")
                    return True
                elif task_status == "failed":
                    print("❌ Simulation failed!")
                    error = task_details.get("error", "Unknown error")
                    print(f"📄 Error: {error}")
                    return False
                elif task_status == "running":
                    print("⚙️ AI agents are working on your financial analysis...")
                elif task_status == "queued":
                    print("📋 Simulation is queued and will start shortly...")
                    
            else:
                print(f"⚠️ Status check failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error during status check: {e}")
        except Exception as e:
            print(f"⚠️ Error during status check: {e}")
        
        # Wait before next check
        time.sleep(5)
        attempt += 1
    
    print("⏰ Monitoring timeout reached")
    return False

def get_simulation_results(task_id):
    """Get the final results of the simulation"""
    print(f"\n📊 Retrieving simulation results for task: {task_id}")
    
    try:
        response = requests.get(
            f"{SERVER_URL}/simulation-results/{task_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ Results retrieved successfully!")
            
            # Display key results
            data = results.get("data", {})
            user_id = results.get("user_id", "N/A")
            source = results.get("source", "unknown")
            
            print(f"\n📈 FINANCIAL SIMULATION RESULTS:")
            print(f"👤 User ID: {user_id}")
            print(f"💾 Data Source: {source}")
            print(f"📋 Task Status: {results.get('task_status', 'N/A')}")
            
            # Show cashflow results
            cashflow_data = data.get("simulated_cashflow", [])
            if cashflow_data:
                print(f"\n💰 CASHFLOW ANALYSIS ({len(cashflow_data)} months):")
                for month_data in cashflow_data:
                    month = month_data.get("month", "N/A")
                    income = month_data.get("income", {})
                    expenses = month_data.get("expenses", {})
                    balance = month_data.get("balance", {})
                    
                    print(f"  Month {month}:")
                    print(f"    💵 Income: ${income.get('total', 0):,.2f}")
                    print(f"    💸 Expenses: ${expenses.get('total', 0):,.2f}")
                    print(f"    💰 Balance Change: ${balance.get('change', 0):,.2f}")
            
            # Show other agent results
            for agent_type in ["discipline_report", "goal_status", "behavior_tracker", "karmic_tracker", "financial_strategy"]:
                agent_data = data.get(agent_type, [])
                if agent_data:
                    print(f"\n📊 {agent_type.upper().replace('_', ' ')} ({len(agent_data)} entries):")
                    for entry in agent_data[:2]:  # Show first 2 entries
                        month = entry.get("month", "N/A")
                        print(f"  Month {month}: {str(entry)[:100]}...")
            
            # Show monthly reflections
            reflections = data.get("monthly_reflections", [])
            if reflections:
                print(f"\n🤔 MONTHLY REFLECTIONS ({len(reflections)} entries):")
                for reflection in reflections:
                    month = reflection.get("month", "N/A")
                    print(f"  Month {month}: Available")
            
            return results
        else:
            print(f"❌ Failed to get results: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def main():
    """Main function to run the financial agent simulation"""
    print("🏦 Financial Agent Simulation Demo")
    print("=" * 60)
    print("🤖 This demo will start a real financial simulation using:")
    print("   • Groq AI (Llama3-70B) for financial analysis")
    print("   • MongoDB Atlas for data storage")
    print("   • Multiple AI agents for comprehensive analysis")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{SERVER_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running and accessible")
        else:
            print("❌ Backend server is not responding correctly")
            return 1
    except:
        print("❌ Cannot connect to backend server")
        print(f"📡 Make sure the server is running on {SERVER_URL}")
        return 1
    
    # Start the simulation
    task_id = start_financial_simulation()
    if not task_id:
        print("❌ Failed to start simulation")
        return 1
    
    # Monitor progress
    print("\n" + "=" * 60)
    success = monitor_simulation_progress(task_id)
    
    if success:
        # Get results
        print("\n" + "=" * 60)
        results = get_simulation_results(task_id)
        
        if results:
            print("\n🎉 Financial Agent Simulation Completed Successfully!")
            print("📊 Check the MongoDB Atlas dashboard to see stored data")
            print("🔍 View AgentOps dashboard for detailed agent interactions")
            print("💡 The AI has analyzed your financial situation using multiple agents")
            return 0
    
    print("\n⚠️ Simulation did not complete successfully")
    print("💡 Check the server logs for more details")
    return 1

if __name__ == "__main__":
    sys.exit(main())
