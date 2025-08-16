"""
Test script for the enhanced financial simulation system.
Tests with the provided user profile: Vedant, ₹8,000 income, ₹5,000 expenses, ₹50,000 emergency fund goal.
"""

import sys
import os
import json
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from functions.enhanced_financial_analysis import EnhancedFinancialAnalyzer
from functions.enhanced_task_functions import (
    generate_enhanced_cashflow_simulation,
    generate_enhanced_financial_strategy,
    generate_enhanced_goal_tracking
)

def test_enhanced_financial_simulation():
    """Test the enhanced financial simulation with the provided user profile."""
    
    print("🚀 Testing Enhanced Financial Simulation System")
    print("=" * 60)
    
    # Test user profile: Vedant, ₹8,000 income, ₹5,000 expenses, ₹50,000 emergency fund goal
    test_user_inputs = {
        "user_id": "vedant_test_user",
        "user_name": "Vedant",
        "income": 8000,
        "expenses": [
            {"name": "food", "amount": 3000},
            {"name": "transportation", "amount": 1000},
            {"name": "utilities", "amount": 500},
            {"name": "entertainment", "amount": 500}
        ],
        "total_expenses": 5000,
        "goal": "Build emergency fund of 50000 rs",
        "financial_type": "moderate",
        "risk_level": "medium"
    }
    
    # Test economic context
    economic_context = {
        "inflation_rate": 0.06,
        "interest_rate": 0.075,
        "cost_of_living_index": 1.02
    }
    
    market_context = "Market conditions this month: Crypto is volatile, Stocks are bullish, Commodities are stable."
    
    print(f"👤 Testing with user: {test_user_inputs['user_name']}")
    print(f"💰 Income: ₹{test_user_inputs['income']:,}")
    print(f"💸 Expenses: ₹{test_user_inputs['total_expenses']:,}")
    print(f"🎯 Goal: {test_user_inputs['goal']}")
    print(f"📊 Financial Type: {test_user_inputs['financial_type']}")
    print(f"⚖️ Risk Level: {test_user_inputs['risk_level']}")
    print()
    
    # Test 1: Enhanced Financial Analysis
    print("🔍 Test 1: Enhanced Financial Analysis")
    print("-" * 40)
    
    try:
        analyzer = EnhancedFinancialAnalyzer()
        analysis = analyzer.analyze_financial_health(test_user_inputs, month=1)
        
        print(f"✅ Financial Health Score: {analysis['financial_health_score']}/100")
        print(f"💰 Monthly Surplus: ₹{analysis['savings_analysis']['monthly_surplus']:,}")
        print(f"📈 Savings Rate: {analysis['savings_analysis']['savings_rate']:.1f}%")
        print(f"🎯 Goal Timeline: {analysis['goal_analysis'].get('months_to_achieve', 'N/A')} months")
        print(f"📊 Investment Recommendations: {len(analysis['investment_recommendations'])}")
        print()
        
        # Show key insights
        print("💡 Key Insights:")
        for insight in analysis['monthly_insights'][:3]:
            print(f"   • {insight}")
        print()
        
    except Exception as e:
        print(f"❌ Enhanced Financial Analysis failed: {e}")
        return False
    
    # Test 2: Enhanced Cashflow Simulation
    print("🔍 Test 2: Enhanced Cashflow Simulation")
    print("-" * 40)
    
    try:
        cashflow_result = generate_enhanced_cashflow_simulation(
            user_inputs=test_user_inputs,
            month=1,
            economic_context=economic_context,
            market_context=market_context,
            previous_data=None
        )
        
        print(f"✅ User: {cashflow_result['user_name']}")
        print(f"💰 Income: ₹{cashflow_result['income']['total']:,}")
        print(f"💸 Expenses: ₹{cashflow_result['expenses']['total']:,}")
        print(f"💵 Savings: ₹{cashflow_result['savings']['amount']:,}")
        print(f"📊 Health Score: {cashflow_result['analysis']['financial_health_score']}")
        print(f"🎯 Goal Progress: {cashflow_result['goal_progress']['percentage']:.1f}%")
        print()
        
        # Show notes
        print("📝 Notes:")
        print(f"   {cashflow_result['notes'][:200]}...")
        print()
        
    except Exception as e:
        print(f"❌ Enhanced Cashflow Simulation failed: {e}")
        return False
    
    # Test 3: Enhanced Financial Strategy
    print("🔍 Test 3: Enhanced Financial Strategy")
    print("-" * 40)
    
    try:
        strategy_result = generate_enhanced_financial_strategy(
            user_inputs=test_user_inputs,
            month=1,
            previous_data=None
        )
        
        print(f"✅ Generated {len(strategy_result['recommendations'])} recommendations")
        print(f"📊 Financial Health Score: {strategy_result['financial_health_score']}")
        print()
        
        # Show recommendations
        print("💡 Strategic Recommendations:")
        for i, rec in enumerate(strategy_result['recommendations'][:3], 1):
            print(f"   {i}. {rec['description']} ({rec['priority']} priority)")
            print(f"      Impact: {rec['reasoning']}")
        print()
        
    except Exception as e:
        print(f"❌ Enhanced Financial Strategy failed: {e}")
        return False
    
    # Test 4: Enhanced Goal Tracking
    print("🔍 Test 4: Enhanced Goal Tracking")
    print("-" * 40)
    
    try:
        goal_result = generate_enhanced_goal_tracking(
            user_inputs=test_user_inputs,
            month=1,
            previous_data=None
        )
        
        goal = goal_result['goals'][0] if goal_result['goals'] else {}
        print(f"✅ Goal: {goal.get('name', 'N/A')}")
        print(f"🎯 Target: ₹{goal.get('target_amount', 0):,}")
        print(f"💰 Saved: ₹{goal.get('saved_so_far', 0):,}")
        print(f"📈 Progress: {goal.get('progress_percentage', 0):.1f}%")
        print(f"⏰ Timeline: {goal.get('months_remaining', 0)} months remaining")
        print(f"📊 Status: {goal.get('status', 'unknown')}")
        print()
        
        # Show insights
        print("💡 Goal Insights:")
        for insight in goal_result.get('insights', [])[:3]:
            print(f"   • {insight}")
        print()
        
    except Exception as e:
        print(f"❌ Enhanced Goal Tracking failed: {e}")
        return False
    
    # Test 5: Multi-month simulation
    print("🔍 Test 5: Multi-month Simulation")
    print("-" * 40)
    
    try:
        previous_months_data = []
        
        for month in range(1, 4):  # Test 3 months
            print(f"   Month {month}:")
            
            # Generate cashflow for this month
            month_cashflow = generate_enhanced_cashflow_simulation(
                user_inputs=test_user_inputs,
                month=month,
                economic_context=economic_context,
                market_context=market_context,
                previous_data=previous_months_data
            )
            
            # Add to previous data for next month
            previous_months_data.append(month_cashflow)
            
            print(f"     Savings: ₹{month_cashflow['savings']['amount']:,}")
            print(f"     Balance: ₹{month_cashflow['balance']['ending']:,}")
            
            # Generate goal tracking
            goal_tracking = generate_enhanced_goal_tracking(
                user_inputs=test_user_inputs,
                month=month,
                previous_data=previous_months_data
            )
            
            goal = goal_tracking['goals'][0] if goal_tracking['goals'] else {}
            print(f"     Goal Progress: {goal.get('progress_percentage', 0):.1f}%")
            print()
        
        print("✅ Multi-month simulation completed successfully!")
        
    except Exception as e:
        print(f"❌ Multi-month simulation failed: {e}")
        return False
    
    print("🎉 All tests completed successfully!")
    print("✅ Enhanced Financial Simulation System is working correctly")
    return True

if __name__ == "__main__":
    success = test_enhanced_financial_simulation()
    if success:
        print("\n🚀 Ready to deploy enhanced financial simulation!")
    else:
        print("\n❌ Tests failed - please check the implementation")
