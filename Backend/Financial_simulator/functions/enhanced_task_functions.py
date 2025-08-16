"""
Enhanced Task Functions for Financial Simulation
Uses the EnhancedFinancialAnalyzer to generate meaningful financial insights.
"""

import os
import json
from typing import Dict, List, Any, Optional
from collections import Counter
from .enhanced_financial_analysis import EnhancedFinancialAnalyzer

# Initialize the enhanced analyzer
financial_analyzer = EnhancedFinancialAnalyzer()

def build_enhanced_cashflow_context(month_number: int, user_id: str, user_inputs: Dict[str, Any], previous_data: Optional[List[Dict]] = None) -> str:
    """
    Build enhanced cashflow context using the financial analyzer.
    
    Args:
        month_number: Current month number
        user_id: User identifier
        user_inputs: User financial profile data
        previous_data: Previous months' data for comparison
        
    Returns:
        Comprehensive financial analysis context string
    """
    try:
        # Generate comprehensive analysis
        analysis = financial_analyzer.generate_comprehensive_monthly_summary(
            user_inputs, month_number, previous_data
        )
        
        # Format the analysis into a readable context
        context_parts = []
        
        # Financial Health Overview
        context_parts.append(f"=== FINANCIAL HEALTH ANALYSIS - MONTH {month_number} ===")
        context_parts.append(f"Financial Health Score: {analysis['financial_health_score']}/100")
        context_parts.append("")
        
        # Income Analysis
        income_analysis = analysis["income_analysis"]
        context_parts.append("INCOME ANALYSIS:")
        context_parts.append(f"- Monthly Income: ₹{income_analysis['monthly_income']:,.0f}")
        context_parts.append(f"- Annual Projection: ₹{income_analysis['annual_projection']:,.0f}")
        context_parts.append("")
        
        # Expense Analysis
        expense_analysis = analysis["expense_analysis"]
        context_parts.append("EXPENSE ANALYSIS:")
        context_parts.append(f"- Total Expenses: ₹{expense_analysis['total_expenses']:,.0f}")
        context_parts.append(f"- Essential Expenses: ₹{expense_analysis['essential_expenses']:,.0f} ({expense_analysis['essential_ratio']:.1f}%)")
        context_parts.append(f"- Non-Essential Expenses: ₹{expense_analysis['non_essential_expenses']:,.0f}")
        context_parts.append(f"- Expense Efficiency: {expense_analysis['expense_efficiency']}")
        context_parts.append("")
        
        # Savings Analysis
        savings_analysis = analysis["savings_analysis"]
        context_parts.append("SAVINGS ANALYSIS:")
        context_parts.append(f"- Monthly Surplus: ₹{savings_analysis['monthly_surplus']:,.0f}")
        context_parts.append(f"- Savings Rate: {savings_analysis['savings_rate']:.1f}%")
        context_parts.append(f"- Target Savings Rate: {savings_analysis['target_savings_rate']:.1f}%")
        context_parts.append(f"- Performance: {savings_analysis['savings_performance']}")
        context_parts.append("")
        
        # Goal Analysis
        goal_analysis = analysis["goal_analysis"]
        context_parts.append("GOAL ANALYSIS:")
        context_parts.append(f"- Goal: {goal_analysis.get('goal_description', 'Not specified')}")
        if goal_analysis.get("target_amount", 0) > 0:
            context_parts.append(f"- Target Amount: ₹{goal_analysis['target_amount']:,.0f}")
            context_parts.append(f"- Monthly Contribution: ₹{goal_analysis.get('monthly_contribution', 0):,.0f}")
            context_parts.append(f"- Timeline: {goal_analysis.get('months_to_achieve', 0)} months ({goal_analysis.get('years_to_achieve', 0)} years)")
            context_parts.append(f"- Feasibility: {goal_analysis.get('feasibility', 'unknown')}")
        context_parts.append("")
        
        # Investment Recommendations
        investment_recs = analysis["investment_recommendations"]
        if investment_recs:
            context_parts.append("INVESTMENT RECOMMENDATIONS:")
            for rec in investment_recs[:3]:  # Top 3 recommendations
                context_parts.append(f"- {rec['type'].title()}: {rec['recommendation']}")
                if 'monthly_allocation' in rec:
                    context_parts.append(f"  Monthly Allocation: ₹{rec['monthly_allocation']:,.0f}")
            context_parts.append("")
        
        # Monthly Insights
        insights = analysis["monthly_insights"]
        if insights:
            context_parts.append("KEY INSIGHTS:")
            for insight in insights:
                context_parts.append(f"- {insight}")
            context_parts.append("")
        
        # Progress Tracking (if available)
        if "progress_tracking" in analysis:
            progress = analysis["progress_tracking"]
            context_parts.append("PROGRESS TRACKING:")
            context_parts.append(f"- Savings Trend: {progress.get('savings_trend', 'unknown')}")
            if progress.get('savings_change', 0) != 0:
                context_parts.append(f"- Savings Change: ₹{progress['savings_change']:,.0f}")
            context_parts.append(f"- Consistency Score: {progress.get('consistency_score', 50)}/100")
            context_parts.append("")
        
        # Actionable Recommendations
        if "actionable_recommendations" in analysis:
            action_recs = analysis["actionable_recommendations"]
            context_parts.append("ACTIONABLE RECOMMENDATIONS:")
            for rec in action_recs:
                context_parts.append(f"- {rec['action']} ({rec['priority']} priority)")
                context_parts.append(f"  Expected Impact: {rec['expected_impact']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        print(f"Error in build_enhanced_cashflow_context: {e}")
        return f"Error generating enhanced context for month {month_number}: {str(e)}"

def generate_enhanced_cashflow_simulation(user_inputs: Dict[str, Any], month: int, economic_context: Dict, market_context: str, previous_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Generate enhanced cashflow simulation with meaningful insights.

    Args:
        user_inputs: User financial profile
        month: Current month number
        economic_context: Economic environment data
        market_context: Market conditions
        previous_data: Previous months' data

    Returns:
        Enhanced cashflow simulation result
    """
    try:
        # Get comprehensive analysis
        analysis = financial_analyzer.generate_comprehensive_monthly_summary(
            user_inputs, month, previous_data
        )

        # Extract basic data
        user_name = user_inputs.get("user_name", "User")
        income = float(user_inputs.get("income", 0))
        total_expenses = float(user_inputs.get("total_expenses", 0))
        monthly_surplus = income - total_expenses

        # Calculate starting balance from previous data
        starting_balance = 0
        if previous_data and len(previous_data) > 0:
            last_month = previous_data[-1]
            if isinstance(last_month.get("balance"), dict):
                starting_balance = last_month["balance"].get("ending", 0)
            else:
                # Try to calculate from savings
                total_saved = sum(
                    data.get("savings", {}).get("amount", 0) if isinstance(data.get("savings"), dict)
                    else data.get("savings", 0)
                    for data in previous_data
                )
                starting_balance = total_saved

        ending_balance = starting_balance + monthly_surplus

        # Generate expense breakdown based on user inputs
        expenses_list = user_inputs.get("expenses", [])
        expense_breakdown = {
            "housing": 0, "utilities": 0, "groceries": 0, "transportation": 0,
            "healthcare": 0, "entertainment": 0, "dining_out": 0, "subscriptions": 0, "other": 0
        }

        # Map user expenses to categories
        for expense in expenses_list:
            name = expense.get("name", "").lower()
            amount = float(expense.get("amount", 0))

            if "food" in name or "grocery" in name:
                expense_breakdown["groceries"] = amount
            elif "house" in name or "rent" in name:
                expense_breakdown["housing"] = amount
            elif "transport" in name or "travel" in name:
                expense_breakdown["transportation"] = amount
            elif "health" in name or "medical" in name:
                expense_breakdown["healthcare"] = amount
            elif "entertainment" in name or "fun" in name:
                expense_breakdown["entertainment"] = amount
            else:
                expense_breakdown["other"] += amount

        # Ensure total matches
        calculated_total = sum(expense_breakdown.values())
        if calculated_total != total_expenses and total_expenses > 0:
            # Adjust 'other' category to match total
            expense_breakdown["other"] += (total_expenses - calculated_total)

        # Generate comprehensive notes based on analysis
        notes_parts = []

        # Add key insights
        for insight in analysis["monthly_insights"][:2]:  # Top 2 insights
            notes_parts.append(insight)

        # Add specific recommendations
        if "actionable_recommendations" in analysis:
            top_rec = analysis["actionable_recommendations"][0] if analysis["actionable_recommendations"] else None
            if top_rec:
                notes_parts.append(f"Recommendation: {top_rec['action']} - {top_rec['expected_impact']}")

        # Add goal progress note
        goal_analysis = analysis["goal_analysis"]
        if goal_analysis.get("target_amount", 0) > 0:
            months_to_goal = goal_analysis.get("months_to_achieve", 0)
            if months_to_goal > 0:
                notes_parts.append(f"Goal Progress: At current savings rate, you'll achieve your ₹{goal_analysis['target_amount']:,.0f} goal in {months_to_goal} months.")

        notes = " ".join(notes_parts)

        # Calculate goal progress percentage
        goal_progress_percentage = 0
        if goal_analysis.get("target_amount", 0) > 0:
            current_saved = ending_balance if ending_balance > 0 else 0
            goal_progress_percentage = min((current_saved / goal_analysis["target_amount"]) * 100, 100)

        # Generate goal progress message
        goal_progress_message = "Starting your financial journey!"
        if goal_progress_percentage > 0:
            if goal_progress_percentage >= 100:
                goal_progress_message = "Congratulations! You've achieved your financial goal!"
            elif goal_progress_percentage >= 75:
                goal_progress_message = "Excellent progress! You're almost there!"
            elif goal_progress_percentage >= 50:
                goal_progress_message = "Great progress! You're halfway to your goal!"
            elif goal_progress_percentage >= 25:
                goal_progress_message = "Good progress! Keep up the momentum!"
            else:
                goal_progress_message = "You're making progress towards your goal!"

        return {
            "user_name": user_name,
            "month": month,
            "income": {
                "salary": income,
                "investments": 0,
                "other": 0,
                "total": income
            },
            "expenses": {
                **expense_breakdown,
                "total": total_expenses
            },
            "savings": {
                "amount": monthly_surplus,
                "percentage_of_income": (monthly_surplus / income * 100) if income > 0 else 0,
                "target_met": monthly_surplus >= (income * 0.2)  # 20% target
            },
            "balance": {
                "starting": starting_balance,
                "ending": ending_balance,
                "change": monthly_surplus
            },
            "analysis": {
                "spending_categories": {
                    "essential": analysis["expense_analysis"]["essential_expenses"],
                    "non_essential": analysis["expense_analysis"]["non_essential_expenses"],
                    "ratio": analysis["expense_analysis"]["essential_ratio"] / 100
                },
                "savings_rate": f"{analysis['savings_analysis']['savings_rate']:.1f}%",
                "cash_flow": "positive" if monthly_surplus > 0 else "negative",
                "financial_health_score": analysis["financial_health_score"]
            },
            "goal_progress": {
                "percentage": goal_progress_percentage,
                "message": goal_progress_message,
                "target_amount": goal_analysis.get("target_amount", 0),
                "months_remaining": goal_analysis.get("months_to_achieve", 0)
            },
            "notes": notes,
            "recommendations": analysis.get("actionable_recommendations", [])[:3]  # Top 3 recommendations
        }

    except Exception as e:
        print(f"Error in generate_enhanced_cashflow_simulation: {e}")
        # Return fallback data
        return {
            "user_name": user_inputs.get("user_name", "User"),
            "month": month,
            "income": {"salary": user_inputs.get("income", 0), "total": user_inputs.get("income", 0)},
            "expenses": {"total": user_inputs.get("total_expenses", 0)},
            "savings": {"amount": user_inputs.get("income", 0) - user_inputs.get("total_expenses", 0)},
            "notes": f"Error generating enhanced analysis: {str(e)}",
            "error": str(e)
        }

def generate_enhanced_financial_strategy(user_inputs: Dict[str, Any], month: int, previous_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Generate enhanced financial strategy recommendations.

    Args:
        user_inputs: User financial profile
        month: Current month number
        previous_data: Previous months' data

    Returns:
        Enhanced financial strategy with specific recommendations
    """
    try:
        # Get comprehensive analysis
        analysis = financial_analyzer.generate_comprehensive_monthly_summary(
            user_inputs, month, previous_data
        )

        # Extract actionable recommendations
        recommendations = analysis.get("actionable_recommendations", [])

        # Format recommendations for the expected output structure
        formatted_recommendations = []
        for rec in recommendations:
            formatted_rec = {
                "type": rec["category"],
                "description": rec["action"],
                "reasoning": rec["expected_impact"],
                "priority": rec["priority"],
                "specific_steps": rec.get("specific_steps", [])
            }
            formatted_recommendations.append(formatted_rec)

        # Add investment-specific recommendations
        investment_recs = analysis.get("investment_recommendations", [])
        for inv_rec in investment_recs[:2]:  # Top 2 investment recommendations
            formatted_recommendations.append({
                "type": "investment",
                "description": inv_rec["recommendation"],
                "reasoning": f"Expected return: {inv_rec.get('expected_return', 'Variable')}",
                "priority": inv_rec["priority"],
                "monthly_allocation": inv_rec.get("monthly_allocation", 0)
            })

        return {
            "month": month,
            "recommendations": formatted_recommendations[:5],  # Limit to top 5
            "financial_health_score": analysis["financial_health_score"],
            "key_insights": analysis["monthly_insights"][:3],  # Top 3 insights
            "goal_status": {
                "target_amount": analysis["goal_analysis"].get("target_amount", 0),
                "months_to_achieve": analysis["goal_analysis"].get("months_to_achieve", 0),
                "feasibility": analysis["goal_analysis"].get("feasibility", "unknown")
            }
        }

    except Exception as e:
        print(f"Error in generate_enhanced_financial_strategy: {e}")
        return {
            "month": month,
            "recommendations": [{
                "type": "error",
                "description": "Unable to generate strategy recommendations",
                "reasoning": str(e)
            }],
            "error": str(e)
        }

def generate_enhanced_goal_tracking(user_inputs: Dict[str, Any], month: int, previous_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Generate enhanced goal tracking with detailed progress analysis.

    Args:
        user_inputs: User financial profile
        month: Current month number
        previous_data: Previous months' data

    Returns:
        Enhanced goal tracking result
    """
    try:
        # Get comprehensive analysis
        analysis = financial_analyzer.generate_comprehensive_monthly_summary(
            user_inputs, month, previous_data
        )

        goal_analysis = analysis["goal_analysis"]
        savings_analysis = analysis["savings_analysis"]

        # Calculate cumulative savings from previous data
        cumulative_savings = 0
        if previous_data:
            for data in previous_data:
                savings = data.get("savings", 0)
                if isinstance(savings, dict):
                    cumulative_savings += savings.get("amount", 0)
                elif isinstance(savings, (int, float)):
                    cumulative_savings += savings

        # Add current month savings
        current_savings = savings_analysis["monthly_surplus"]
        cumulative_savings += current_savings

        # Extract goal information
        goal_description = goal_analysis.get("goal_description", "Financial Goal")
        target_amount = goal_analysis.get("target_amount", 0)

        # Calculate progress metrics
        progress_percentage = 0
        status = "not_started"
        expected_by_now = 0

        if target_amount > 0:
            progress_percentage = min((cumulative_savings / target_amount) * 100, 100)

            # Calculate expected progress (assuming linear progress over timeline)
            months_to_achieve = goal_analysis.get("months_to_achieve", 36)  # Default 3 years
            expected_by_now = (target_amount / months_to_achieve) * month

            if progress_percentage >= 100:
                status = "achieved"
            elif cumulative_savings >= expected_by_now * 0.9:  # Within 10% of expected
                status = "on_track"
            elif cumulative_savings >= expected_by_now * 0.7:  # Within 30% of expected
                status = "slightly_behind"
            else:
                status = "behind"

        # Generate adjustment suggestions
        adjustment_suggestion = ""
        if status == "behind":
            shortfall = expected_by_now - cumulative_savings
            adjustment_suggestion = f"Increase monthly savings by ₹{shortfall / (months_to_achieve - month):,.0f} to get back on track"
        elif status == "on_track":
            adjustment_suggestion = "Continue current savings rate to achieve goal on time"
        elif status == "achieved":
            adjustment_suggestion = "Goal achieved! Consider setting a new financial goal"
        else:
            adjustment_suggestion = "Maintain consistent savings to stay on track"

        # Create goals array (supporting multiple goals in future)
        goals = [{
            "name": goal_description,
            "target_amount": target_amount,
            "saved_so_far": cumulative_savings,
            "expected_by_now": expected_by_now,
            "status": status,
            "priority": 1,  # High priority for main goal
            "adjustment_suggestion": adjustment_suggestion,
            "progress_percentage": progress_percentage,
            "months_remaining": max(0, goal_analysis.get("months_to_achieve", 0) - month)
        }]

        # Calculate summary statistics
        on_track_goals = sum(1 for goal in goals if goal["status"] in ["on_track", "achieved"])
        behind_goals = sum(1 for goal in goals if goal["status"] in ["behind", "slightly_behind"])

        return {
            "month": month,
            "goals": goals,
            "summary": {
                "on_track_goals": on_track_goals,
                "behind_goals": behind_goals,
                "total_saved": cumulative_savings,
                "total_required_by_now": expected_by_now,
                "overall_progress": progress_percentage
            },
            "insights": [
                f"You've saved ₹{cumulative_savings:,.0f} towards your goal of ₹{target_amount:,.0f}",
                f"Current progress: {progress_percentage:.1f}% of target amount",
                f"Monthly savings rate: {savings_analysis['savings_rate']:.1f}%"
            ],
            "recommendations": [
                adjustment_suggestion,
                "Review and adjust goal timeline if needed",
                "Consider automating savings to ensure consistency"
            ]
        }

    except Exception as e:
        print(f"Error in generate_enhanced_goal_tracking: {e}")
        return {
            "month": month,
            "goals": [{
                "name": user_inputs.get("goal", "Financial Goal"),
                "target_amount": 0,
                "saved_so_far": 0,
                "expected_by_now": 0,
                "status": "unknown",
                "priority": 1,
                "adjustment_suggestion": "Unable to calculate progress"
            }],
            "summary": {
                "on_track_goals": 0,
                "behind_goals": 0,
                "total_saved": 0,
                "total_required_by_now": 0
            },
            "error": str(e)
        }
