#!/usr/bin/env python3
"""
Start Financial Agent Simulation
This script demonstrates how to start a financial simulation using the backend API
"""

import requests
import json
import time
import sys

# Server configuration
SERVER_URL = "http://localhost:8002"

def start_financial_simulation():
    """Start a financial simulation with sample data"""
    print("🚀 Starting Financial Agent Simulation...")
    
    # Sample financial simulation data
    simulation_data = {
        "user_id": "demo_user_001",
        "initial_capital": 100000,  # $100,000 starting capital
        "simulation_months": 12,    # 12-month simulation
        "risk_tolerance": "moderate",
        "investment_goals": [
            "wealth_building",
            "retirement_planning"
        ],
        "financial_profile": {
            "age": 35,
            "income": 75000,
            "expenses": 45000,
            "debt": 25000,
            "savings_rate": 0.20
        },
        "market_conditions": {
            "economic_outlook": "stable",
            "inflation_rate": 0.03,
            "interest_rates": 0.05
        },
        "preferences": {
            "sectors": ["technology", "healthcare", "finance"],
            "esg_focus": True,
            "international_exposure": 0.30
        }
    }
    
    try:
        # Start the simulation
        print("📊 Sending simulation request to backend...")
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
            print(f"👤 User ID: {result.get('user_id')}")
            print(f"💰 Initial Capital: ${simulation_data['initial_capital']:,}")
            print(f"📅 Duration: {simulation_data['simulation_months']} months")
            
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
    
    max_attempts = 60  # Maximum 5 minutes of monitoring
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
                status = status_data.get("status", "unknown")
                progress = status_data.get("progress", 0)
                current_month = status_data.get("current_month", 0)
                
                print(f"📊 Status: {status} | Progress: {progress}% | Month: {current_month}")
                
                if status == "completed":
                    print("🎉 Simulation completed successfully!")
                    return True
                elif status == "failed":
                    print("❌ Simulation failed!")
                    error = status_data.get("error", "Unknown error")
                    print(f"📄 Error: {error}")
                    return False
                elif status == "running":
                    print("⏳ Simulation is running...")
                    
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
            if "summary" in results:
                summary = results["summary"]
                print(f"\n📈 SIMULATION SUMMARY:")
                print(f"💰 Final Portfolio Value: ${summary.get('final_value', 'N/A'):,}")
                print(f"📊 Total Return: {summary.get('total_return', 'N/A')}%")
                print(f"📅 Simulation Period: {summary.get('months', 'N/A')} months")
                print(f"⚖️ Risk Level: {summary.get('risk_level', 'N/A')}")
            
            if "monthly_results" in results:
                monthly_data = results["monthly_results"]
                print(f"\n📅 MONTHLY BREAKDOWN:")
                for month_data in monthly_data[-3:]:  # Show last 3 months
                    month = month_data.get("month", "N/A")
                    value = month_data.get("portfolio_value", "N/A")
                    return_pct = month_data.get("monthly_return", "N/A")
                    print(f"Month {month}: ${value:,} (Return: {return_pct}%)")
            
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
    print("🏦 Financial Agent Simulation Starter")
    print("=" * 50)
    
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
    print("\n" + "=" * 50)
    success = monitor_simulation_progress(task_id)
    
    if success:
        # Get results
        print("\n" + "=" * 50)
        results = get_simulation_results(task_id)
        
        if results:
            print("\n🎉 Financial Agent Simulation Completed Successfully!")
            print("📊 Check the MongoDB Atlas dashboard to see stored data")
            print("🔍 View AgentOps dashboard for detailed agent interactions")
            return 0
    
    print("\n⚠️ Simulation did not complete successfully")
    return 1

if __name__ == "__main__":
    sys.exit(main())
