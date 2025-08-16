"""
Enhanced Financial Analysis Module
Provides meaningful financial insights and analysis based on user profile data.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math

class EnhancedFinancialAnalyzer:
    """Enhanced financial analyzer that provides meaningful insights."""
    
    def __init__(self):
        self.risk_profiles = {
            "conservative": {"max_risk_allocation": 0.3, "emergency_fund_months": 6},
            "moderate": {"max_risk_allocation": 0.6, "emergency_fund_months": 4},
            "aggressive": {"max_risk_allocation": 0.8, "emergency_fund_months": 3}
        }
        
        self.financial_types = {
            "conservative": {"savings_rate_target": 0.25, "investment_preference": "low_risk"},
            "moderate": {"savings_rate_target": 0.20, "investment_preference": "balanced"},
            "risky": {"savings_rate_target": 0.15, "investment_preference": "high_growth"}
        }
    
    def analyze_financial_health(self, user_inputs: Dict[str, Any], month: int = 1) -> Dict[str, Any]:
        """
        Comprehensive financial health analysis.
        
        Args:
            user_inputs: User financial profile data
            month: Current month number
            
        Returns:
            Detailed financial health analysis
        """
        income = float(user_inputs.get("income", 0))
        expenses_list = user_inputs.get("expenses", [])
        total_expenses = float(user_inputs.get("total_expenses", 0))
        goal = user_inputs.get("goal", "")
        financial_type = user_inputs.get("financial_type", "moderate").lower()
        risk_level = user_inputs.get("risk_level", "medium").lower()
        
        # Calculate basic metrics
        monthly_surplus = income - total_expenses
        savings_rate = (monthly_surplus / income) * 100 if income > 0 else 0
        
        # Expense breakdown analysis
        expense_breakdown = self._analyze_expense_breakdown(expenses_list, total_expenses)
        
        # Goal analysis
        goal_analysis = self._analyze_financial_goal(goal, monthly_surplus, income)
        
        # Risk assessment
        risk_assessment = self._assess_financial_risk(income, total_expenses, financial_type, risk_level)
        
        # Investment recommendations
        investment_recommendations = self._generate_investment_recommendations(
            monthly_surplus, financial_type, risk_level, income
        )
        
        # Budget optimization suggestions
        budget_suggestions = self._generate_budget_suggestions(
            expense_breakdown, monthly_surplus, income, goal_analysis
        )
        
        return {
            "month": month,
            "financial_health_score": self._calculate_financial_health_score(
                savings_rate, expense_breakdown, goal_analysis
            ),
            "income_analysis": {
                "monthly_income": income,
                "annual_projection": income * 12,
                "income_stability": "stable"  # Could be enhanced with historical data
            },
            "expense_analysis": expense_breakdown,
            "savings_analysis": {
                "monthly_surplus": monthly_surplus,
                "savings_rate": round(savings_rate, 2),
                "target_savings_rate": self.financial_types.get(financial_type, {}).get("savings_rate_target", 0.20) * 100,
                "savings_performance": "above_target" if savings_rate >= 20 else "below_target"
            },
            "goal_analysis": goal_analysis,
            "risk_assessment": risk_assessment,
            "investment_recommendations": investment_recommendations,
            "budget_suggestions": budget_suggestions,
            "monthly_insights": self._generate_monthly_insights(
                income, total_expenses, monthly_surplus, goal_analysis, month
            )
        }
    
    def _analyze_expense_breakdown(self, expenses_list: List[Dict], total_expenses: float) -> Dict[str, Any]:
        """Analyze expense breakdown and categorization."""
        essential_categories = ["food", "housing", "utilities", "transportation", "healthcare"]
        non_essential_categories = ["entertainment", "dining_out", "subscriptions", "shopping"]
        
        essential_expenses = 0
        non_essential_expenses = 0
        expense_details = {}
        
        for expense in expenses_list:
            name = expense.get("name", "").lower()
            amount = float(expense.get("amount", 0))
            expense_details[name] = amount
            
            if any(cat in name for cat in essential_categories):
                essential_expenses += amount
            else:
                non_essential_expenses += amount
        
        # If we can't categorize properly, use a reasonable split
        if essential_expenses == 0 and non_essential_expenses == 0:
            essential_expenses = total_expenses * 0.7  # Assume 70% essential
            non_essential_expenses = total_expenses * 0.3  # Assume 30% non-essential
        
        return {
            "total_expenses": total_expenses,
            "essential_expenses": essential_expenses,
            "non_essential_expenses": non_essential_expenses,
            "essential_ratio": (essential_expenses / total_expenses) * 100 if total_expenses > 0 else 0,
            "expense_details": expense_details,
            "expense_efficiency": "good" if essential_expenses / total_expenses > 0.6 else "needs_improvement"
        }
    
    def _analyze_financial_goal(self, goal: str, monthly_surplus: float, income: float) -> Dict[str, Any]:
        """Analyze financial goal and provide timeline projections."""
        # Extract goal amount from goal string (basic parsing)
        goal_amount = self._extract_goal_amount(goal)
        
        if goal_amount > 0 and monthly_surplus > 0:
            months_to_goal = math.ceil(goal_amount / monthly_surplus)
            years_to_goal = months_to_goal / 12
            
            # Calculate if goal is realistic
            goal_feasibility = "realistic" if months_to_goal <= 60 else "challenging"
            
            return {
                "goal_description": goal,
                "target_amount": goal_amount,
                "monthly_contribution": monthly_surplus,
                "months_to_achieve": months_to_goal,
                "years_to_achieve": round(years_to_goal, 1),
                "feasibility": goal_feasibility,
                "progress_percentage": 0,  # Will be updated with historical data
                "recommended_monthly_savings": max(monthly_surplus, goal_amount / 36)  # 3-year target
            }
        
        return {
            "goal_description": goal,
            "target_amount": goal_amount,
            "analysis": "insufficient_surplus" if monthly_surplus <= 0 else "goal_amount_unclear",
            "recommendation": "increase_income_or_reduce_expenses" if monthly_surplus <= 0 else "clarify_goal_amount"
        }
    
    def _extract_goal_amount(self, goal: str) -> float:
        """Extract numerical goal amount from goal string."""
        import re
        
        # Common patterns for Indian currency
        patterns = [
            r'(\d+(?:,\d+)*)\s*(?:crore|cr)',  # X crore
            r'(\d+(?:,\d+)*)\s*(?:lakh|lac)',  # X lakh
            r'(\d+(?:,\d+)*)\s*(?:thousand|k)',  # X thousand
            r'₹\s*(\d+(?:,\d+)*)',  # ₹X
            r'(\d+(?:,\d+)*)\s*(?:rs|rupees)',  # X rs/rupees
        ]
        
        goal_lower = goal.lower()
        
        for pattern in patterns:
            match = re.search(pattern, goal_lower)
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                
                if 'crore' in goal_lower or 'cr' in goal_lower:
                    return amount * 10000000  # 1 crore = 10 million
                elif 'lakh' in goal_lower or 'lac' in goal_lower:
                    return amount * 100000  # 1 lakh = 100 thousand
                elif 'thousand' in goal_lower or 'k' in goal_lower:
                    return amount * 1000
                else:
                    return amount
        
        # Default fallback for common goals
        if 'emergency' in goal_lower:
            return 50000  # Default emergency fund
        elif 'house' in goal_lower or 'home' in goal_lower:
            return 2000000  # Default house down payment
        elif 'car' in goal_lower:
            return 500000  # Default car purchase
        
        return 0

    def _assess_financial_risk(self, income: float, expenses: float, financial_type: str, risk_level: str) -> Dict[str, Any]:
        """Assess financial risk based on user profile."""
        debt_to_income_ratio = 0  # Would need debt information
        emergency_fund_needed = expenses * self.risk_profiles.get(risk_level, {}).get("emergency_fund_months", 4)

        risk_factors = []
        if (income - expenses) / income < 0.1:  # Less than 10% savings rate
            risk_factors.append("low_savings_rate")
        if expenses / income > 0.8:  # High expense ratio
            risk_factors.append("high_expense_ratio")

        risk_score = len(risk_factors) * 25  # Simple scoring

        return {
            "risk_level": risk_level,
            "risk_score": min(risk_score, 100),
            "risk_factors": risk_factors,
            "emergency_fund_needed": emergency_fund_needed,
            "financial_stability": "stable" if risk_score < 50 else "at_risk",
            "recommendations": self._get_risk_mitigation_recommendations(risk_factors)
        }

    def _generate_investment_recommendations(self, surplus: float, financial_type: str, risk_level: str, income: float) -> List[Dict[str, Any]]:
        """Generate investment recommendations based on user profile."""
        if surplus <= 0:
            return [{
                "type": "emergency",
                "recommendation": "Focus on increasing income or reducing expenses before investing",
                "priority": "high"
            }]

        recommendations = []

        # Emergency fund first
        emergency_fund_target = (income * 0.6) * self.risk_profiles.get(risk_level, {}).get("emergency_fund_months", 4)
        recommendations.append({
            "type": "emergency_fund",
            "recommendation": f"Build emergency fund of ₹{emergency_fund_target:,.0f} ({self.risk_profiles.get(risk_level, {}).get('emergency_fund_months', 4)} months expenses)",
            "monthly_allocation": min(surplus * 0.4, emergency_fund_target / 12),
            "priority": "high"
        })

        # Investment recommendations based on risk profile
        remaining_surplus = surplus * 0.6

        if risk_level == "conservative" or financial_type == "conservative":
            recommendations.extend([
                {
                    "type": "fixed_deposits",
                    "recommendation": "Invest in Fixed Deposits or PPF for guaranteed returns",
                    "monthly_allocation": remaining_surplus * 0.6,
                    "expected_return": "7-8% annually",
                    "priority": "medium"
                },
                {
                    "type": "debt_funds",
                    "recommendation": "Consider debt mutual funds for better tax efficiency",
                    "monthly_allocation": remaining_surplus * 0.4,
                    "expected_return": "6-9% annually",
                    "priority": "medium"
                }
            ])
        elif risk_level == "moderate" or financial_type == "moderate":
            recommendations.extend([
                {
                    "type": "balanced_funds",
                    "recommendation": "Invest in balanced mutual funds (60% equity, 40% debt)",
                    "monthly_allocation": remaining_surplus * 0.5,
                    "expected_return": "10-12% annually",
                    "priority": "medium"
                },
                {
                    "type": "index_funds",
                    "recommendation": "Consider low-cost index funds for long-term growth",
                    "monthly_allocation": remaining_surplus * 0.3,
                    "expected_return": "11-13% annually",
                    "priority": "medium"
                },
                {
                    "type": "ppf_elss",
                    "recommendation": "Invest in PPF and ELSS for tax benefits",
                    "monthly_allocation": remaining_surplus * 0.2,
                    "expected_return": "8-12% annually",
                    "priority": "low"
                }
            ])
        else:  # aggressive/risky
            recommendations.extend([
                {
                    "type": "equity_funds",
                    "recommendation": "Invest in diversified equity mutual funds",
                    "monthly_allocation": remaining_surplus * 0.6,
                    "expected_return": "12-15% annually",
                    "priority": "medium"
                },
                {
                    "type": "small_cap_funds",
                    "recommendation": "Consider small-cap funds for higher growth potential",
                    "monthly_allocation": remaining_surplus * 0.3,
                    "expected_return": "15-18% annually (higher risk)",
                    "priority": "low"
                },
                {
                    "type": "direct_equity",
                    "recommendation": "Consider direct equity investment with proper research",
                    "monthly_allocation": remaining_surplus * 0.1,
                    "expected_return": "Variable (high risk)",
                    "priority": "low"
                }
            ])

        return recommendations

    def _generate_budget_suggestions(self, expense_breakdown: Dict, surplus: float, income: float, goal_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate budget optimization suggestions."""
        suggestions = []

        # Expense optimization
        if expense_breakdown["essential_ratio"] < 60:
            suggestions.append({
                "type": "expense_optimization",
                "suggestion": "Your essential expenses are well-controlled. Consider optimizing non-essential spending.",
                "potential_savings": expense_breakdown["non_essential_expenses"] * 0.2,
                "priority": "medium"
            })
        elif expense_breakdown["essential_ratio"] > 80:
            suggestions.append({
                "type": "essential_expense_review",
                "suggestion": "Review essential expenses for potential savings (housing, utilities, transportation).",
                "potential_savings": expense_breakdown["essential_expenses"] * 0.1,
                "priority": "high"
            })

        # Savings rate improvement
        current_savings_rate = (surplus / income) * 100 if income > 0 else 0
        if current_savings_rate < 20:
            suggestions.append({
                "type": "savings_rate_improvement",
                "suggestion": f"Increase savings rate from {current_savings_rate:.1f}% to 20% by reducing expenses by ₹{(income * 0.2 - surplus):,.0f}",
                "target_savings": income * 0.2,
                "priority": "high"
            })

        # Goal-specific suggestions
        if goal_analysis.get("feasibility") == "challenging":
            suggestions.append({
                "type": "goal_acceleration",
                "suggestion": "Consider increasing income or extending goal timeline for better feasibility",
                "alternative_timeline": goal_analysis.get("months_to_achieve", 0) * 0.75,
                "priority": "medium"
            })

        return suggestions

    def _get_risk_mitigation_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Get recommendations to mitigate identified risk factors."""
        recommendations = []

        if "low_savings_rate" in risk_factors:
            recommendations.append("Increase savings rate by reducing non-essential expenses")
        if "high_expense_ratio" in risk_factors:
            recommendations.append("Review and optimize major expense categories")

        return recommendations

    def _calculate_financial_health_score(self, savings_rate: float, expense_breakdown: Dict, goal_analysis: Dict) -> int:
        """Calculate overall financial health score (0-100)."""
        score = 0

        # Savings rate component (40 points max)
        if savings_rate >= 25:
            score += 40
        elif savings_rate >= 20:
            score += 35
        elif savings_rate >= 15:
            score += 25
        elif savings_rate >= 10:
            score += 15
        else:
            score += max(0, savings_rate)

        # Expense efficiency component (30 points max)
        essential_ratio = expense_breakdown.get("essential_ratio", 0)
        if essential_ratio >= 60 and essential_ratio <= 80:
            score += 30
        elif essential_ratio >= 50 and essential_ratio <= 90:
            score += 20
        else:
            score += 10

        # Goal feasibility component (30 points max)
        if goal_analysis.get("feasibility") == "realistic":
            score += 30
        elif goal_analysis.get("months_to_achieve", 0) > 0:
            score += 20
        else:
            score += 10

        return min(score, 100)

    def _generate_monthly_insights(self, income: float, expenses: float, surplus: float, goal_analysis: Dict, month: int) -> List[str]:
        """Generate monthly financial insights."""
        insights = []

        # Income vs expenses insight
        expense_ratio = (expenses / income) * 100 if income > 0 else 0
        if expense_ratio > 80:
            insights.append(f"Your expenses consume {expense_ratio:.1f}% of your income. Consider reducing non-essential spending.")
        elif expense_ratio < 60:
            insights.append(f"Excellent expense control! You're using only {expense_ratio:.1f}% of your income for expenses.")
        else:
            insights.append(f"Your expense ratio of {expense_ratio:.1f}% is reasonable, with room for optimization.")

        # Savings potential insight
        if surplus > 0:
            annual_savings = surplus * 12
            insights.append(f"At current savings rate, you could save ₹{annual_savings:,.0f} annually.")

            # Goal progress insight
            if goal_analysis.get("target_amount", 0) > 0:
                months_to_goal = goal_analysis.get("months_to_achieve", 0)
                if months_to_goal <= 24:
                    insights.append(f"You're on track to achieve your goal in {months_to_goal} months!")
                elif months_to_goal <= 60:
                    insights.append(f"Your goal is achievable in {months_to_goal} months with consistent savings.")
                else:
                    insights.append(f"Consider increasing savings or adjusting your goal timeline for better feasibility.")
        else:
            insights.append("Focus on creating a positive cash flow before setting investment goals.")

        # Month-specific insights
        if month == 1:
            insights.append("This is your first month - establish good financial habits early!")
        elif month <= 3:
            insights.append("You're building momentum - consistency is key to financial success.")
        elif month <= 6:
            insights.append("Review your progress and adjust strategies based on what's working.")

        return insights

    def generate_comprehensive_monthly_summary(self, user_inputs: Dict[str, Any], month: int, previous_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate a comprehensive monthly financial summary."""
        analysis = self.analyze_financial_health(user_inputs, month)

        # Add progress tracking if previous data is available
        if previous_data and len(previous_data) > 0:
            analysis["progress_tracking"] = self._analyze_progress(analysis, previous_data)

        # Generate actionable recommendations
        analysis["actionable_recommendations"] = self._generate_actionable_recommendations(analysis)

        # Add month-over-month comparison
        if previous_data:
            analysis["month_over_month"] = self._compare_with_previous_month(analysis, previous_data)

        return analysis

    def _analyze_progress(self, current_analysis: Dict, previous_data: List[Dict]) -> Dict[str, Any]:
        """Analyze progress compared to previous months."""
        if not previous_data:
            return {"status": "no_previous_data"}

        # Get the most recent month's data
        last_month = previous_data[-1] if previous_data else {}

        current_surplus = current_analysis["savings_analysis"]["monthly_surplus"]
        last_surplus = last_month.get("savings", {}).get("amount", 0) if isinstance(last_month.get("savings"), dict) else last_month.get("savings", 0)

        progress = {
            "savings_trend": "improving" if current_surplus > last_surplus else "declining" if current_surplus < last_surplus else "stable",
            "savings_change": current_surplus - last_surplus,
            "consistency_score": self._calculate_consistency_score(previous_data)
        }

        return progress

    def _calculate_consistency_score(self, previous_data: List[Dict]) -> int:
        """Calculate consistency score based on historical data."""
        if len(previous_data) < 2:
            return 50  # Neutral score for insufficient data

        # Analyze savings consistency
        savings_amounts = []
        for data in previous_data:
            savings = data.get("savings", 0)
            if isinstance(savings, dict):
                savings_amounts.append(savings.get("amount", 0))
            else:
                savings_amounts.append(savings)

        if not savings_amounts:
            return 50

        # Calculate coefficient of variation (lower is more consistent)
        mean_savings = sum(savings_amounts) / len(savings_amounts)
        if mean_savings == 0:
            return 25  # Low score for no savings

        variance = sum((x - mean_savings) ** 2 for x in savings_amounts) / len(savings_amounts)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_savings

        # Convert to score (0-100, where 100 is most consistent)
        consistency_score = max(0, 100 - (cv * 100))
        return int(consistency_score)

    def _generate_actionable_recommendations(self, analysis: Dict) -> List[Dict[str, Any]]:
        """Generate specific, actionable recommendations."""
        recommendations = []

        savings_analysis = analysis["savings_analysis"]
        expense_analysis = analysis["expense_analysis"]
        goal_analysis = analysis["goal_analysis"]

        # Savings recommendations
        if savings_analysis["savings_rate"] < 15:
            recommendations.append({
                "category": "savings",
                "action": "Increase savings rate",
                "specific_steps": [
                    f"Reduce non-essential expenses by ₹{expense_analysis['non_essential_expenses'] * 0.2:,.0f}",
                    "Set up automatic savings transfer",
                    "Track daily expenses for one week"
                ],
                "expected_impact": f"Could increase monthly savings by ₹{expense_analysis['non_essential_expenses'] * 0.2:,.0f}",
                "priority": "high"
            })

        # Goal-specific recommendations
        if goal_analysis.get("feasibility") == "challenging":
            recommendations.append({
                "category": "goal_planning",
                "action": "Optimize goal strategy",
                "specific_steps": [
                    "Consider extending timeline by 12 months",
                    "Explore additional income sources",
                    "Review and reduce target amount if possible"
                ],
                "expected_impact": "Make goal more achievable and reduce financial stress",
                "priority": "medium"
            })

        # Investment recommendations
        monthly_surplus = savings_analysis["monthly_surplus"]
        if monthly_surplus > 1000:
            recommendations.append({
                "category": "investment",
                "action": "Start systematic investment",
                "specific_steps": [
                    f"Invest ₹{monthly_surplus * 0.6:,.0f} in balanced mutual funds",
                    f"Keep ₹{monthly_surplus * 0.4:,.0f} in emergency fund",
                    "Set up SIP (Systematic Investment Plan)"
                ],
                "expected_impact": "Potential returns of 10-12% annually",
                "priority": "medium"
            })

        return recommendations

    def _compare_with_previous_month(self, current_analysis: Dict, previous_data: List[Dict]) -> Dict[str, Any]:
        """Compare current month with previous month."""
        if not previous_data:
            return {"status": "no_comparison_data"}

        last_month = previous_data[-1]
        current_surplus = current_analysis["savings_analysis"]["monthly_surplus"]

        # Extract previous month surplus
        last_surplus = 0
        if isinstance(last_month.get("savings"), dict):
            last_surplus = last_month["savings"].get("amount", 0)
        elif isinstance(last_month.get("savings"), (int, float)):
            last_surplus = last_month["savings"]

        comparison = {
            "surplus_change": current_surplus - last_surplus,
            "surplus_change_percentage": ((current_surplus - last_surplus) / last_surplus * 100) if last_surplus > 0 else 0,
            "trend": "improving" if current_surplus > last_surplus else "declining" if current_surplus < last_surplus else "stable"
        }

        return comparison
