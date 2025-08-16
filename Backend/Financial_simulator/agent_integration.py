"""
Agent Integration Module for Forecast Engine v2
Implements mock triggers for SalesAgent and MarketingAgent to demonstrate
integration with the prediction system.
"""

import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesAgent:
    """
    Mock Sales Agent that integrates with the Forecast Engine v2.
    Demonstrates how agents can adapt behavior based on predictions.
    """
    
    def __init__(self, agent_id: str = "sales_agent_001", api_base_url: str = "http://localhost:8002"):
        self.agent_id = agent_id
        self.api_base_url = api_base_url
        self.current_load = 0
        self.lead_qualification_threshold = 0.7  # Default threshold
        
    def process_lead(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a lead with forecast-based adaptation.
        
        Args:
            lead_data: Dictionary containing lead information
            
        Returns:
            Processing result with forecast-influenced decisions
        """
        logger.info(f"🎯 SalesAgent processing lead: {lead_data.get('lead_id', 'unknown')}")
        
        try:
            # Get forecast data to adapt behavior
            forecast_response = self._get_forecast_data()
            
            if forecast_response:
                # Adapt lead qualification based on predicted load
                predicted_load = self._extract_predicted_load(forecast_response)
                self._adapt_qualification_threshold(predicted_load)
            
            # Process the lead
            qualification_score = self._qualify_lead(lead_data)
            
            # Make decision based on adapted threshold
            if qualification_score >= self.lead_qualification_threshold:
                decision = "accept"
                priority = "high" if qualification_score > 0.8 else "medium"
            else:
                decision = "defer"
                priority = "low"
            
            result = {
                "agent_id": self.agent_id,
                "lead_id": lead_data.get('lead_id'),
                "decision": decision,
                "priority": priority,
                "qualification_score": qualification_score,
                "threshold_used": self.lead_qualification_threshold,
                "forecast_influenced": forecast_response is not None,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Lead processed: {decision} (score: {qualification_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing lead: {e}")
            return {
                "agent_id": self.agent_id,
                "lead_id": lead_data.get('lead_id'),
                "decision": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_forecast_data(self) -> Optional[Dict[str, Any]]:
        """Get forecast data from the prediction API."""
        try:
            response = requests.get(f"{self.api_base_url}/forecast?days=7", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Forecast API returned status {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Failed to get forecast data: {e}")
            return None
    
    def _extract_predicted_load(self, forecast_data: Dict[str, Any]) -> float:
        """Extract predicted daily load from forecast response."""
        try:
            content = forecast_data.get('content', {})
            forecast_data_content = content.get('forecast_data', {})
            daily_load = forecast_data_content.get('daily_agent_load', {})
            predictions = daily_load.get('predictions', [])
            
            if predictions:
                # Get average load for next 3 days
                next_3_days = predictions[:3]
                avg_load = sum(p['yhat'] for p in next_3_days) / len(next_3_days)
                return avg_load
            
            return 15.0  # Default load
        except Exception as e:
            logger.warning(f"Error extracting predicted load: {e}")
            return 15.0
    
    def _adapt_qualification_threshold(self, predicted_load: float):
        """Adapt lead qualification threshold based on predicted load."""
        if predicted_load > 25:
            # High predicted load - be more selective
            self.lead_qualification_threshold = 0.8
            logger.info(f"📈 High load predicted ({predicted_load:.1f}), raising threshold to 0.8")
        elif predicted_load < 10:
            # Low predicted load - be less selective
            self.lead_qualification_threshold = 0.5
            logger.info(f"📉 Low load predicted ({predicted_load:.1f}), lowering threshold to 0.5")
        else:
            # Normal load - use default threshold
            self.lead_qualification_threshold = 0.7
            logger.info(f"📊 Normal load predicted ({predicted_load:.1f}), using default threshold 0.7")
    
    def _qualify_lead(self, lead_data: Dict[str, Any]) -> float:
        """Simple lead qualification scoring."""
        score = 0.5  # Base score
        
        # Add points for various factors
        if lead_data.get('budget', 0) > 10000:
            score += 0.2
        if lead_data.get('urgency', 'low') == 'high':
            score += 0.2
        if lead_data.get('decision_maker', False):
            score += 0.1
        
        return min(1.0, score)


class MarketingAgent:
    """
    Mock Marketing Agent that integrates with the Forecast Engine v2.
    Demonstrates campaign intensity adaptation based on capacity predictions.
    """
    
    def __init__(self, agent_id: str = "marketing_agent_001", api_base_url: str = "http://localhost:8002"):
        self.agent_id = agent_id
        self.api_base_url = api_base_url
        self.campaign_intensity = 1.0  # Default intensity multiplier
        
    def plan_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan a marketing campaign with forecast-based intensity adjustment.
        
        Args:
            campaign_data: Dictionary containing campaign information
            
        Returns:
            Campaign plan with forecast-influenced adjustments
        """
        logger.info(f"📢 MarketingAgent planning campaign: {campaign_data.get('campaign_id', 'unknown')}")
        
        try:
            # Get forecast data to adapt campaign intensity
            forecast_response = self._get_forecast_data()
            
            if forecast_response:
                # Adapt campaign intensity based on predicted capacity
                self._adapt_campaign_intensity(forecast_response)
            
            # Calculate adjusted campaign parameters
            base_budget = campaign_data.get('budget', 10000)
            base_reach = campaign_data.get('target_reach', 1000)
            
            adjusted_budget = base_budget * self.campaign_intensity
            adjusted_reach = base_reach * self.campaign_intensity
            
            result = {
                "agent_id": self.agent_id,
                "campaign_id": campaign_data.get('campaign_id'),
                "original_budget": base_budget,
                "adjusted_budget": adjusted_budget,
                "original_reach": base_reach,
                "adjusted_reach": adjusted_reach,
                "intensity_multiplier": self.campaign_intensity,
                "forecast_influenced": forecast_response is not None,
                "recommendation": self._get_campaign_recommendation(),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Campaign planned with intensity {self.campaign_intensity:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error planning campaign: {e}")
            return {
                "agent_id": self.agent_id,
                "campaign_id": campaign_data.get('campaign_id'),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_forecast_data(self) -> Optional[Dict[str, Any]]:
        """Get forecast data from the prediction API."""
        try:
            response = requests.get(f"{self.api_base_url}/forecast?days=14", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Forecast API returned status {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Failed to get forecast data: {e}")
            return None
    
    def _adapt_campaign_intensity(self, forecast_data: Dict[str, Any]):
        """Adapt campaign intensity based on forecast data."""
        try:
            content = forecast_data.get('content', {})
            risk_level = content.get('risk', 'stable')
            
            if risk_level == 'critical':
                # Critical risk - reduce campaign intensity
                self.campaign_intensity = 0.5
                logger.info("🚨 Critical risk detected, reducing campaign intensity to 0.5")
            elif risk_level == 'warning':
                # Warning risk - moderate reduction
                self.campaign_intensity = 0.7
                logger.info("⚠️ Warning risk detected, reducing campaign intensity to 0.7")
            else:
                # Stable - normal or increased intensity
                self.campaign_intensity = 1.2
                logger.info("✅ Stable conditions, increasing campaign intensity to 1.2")
                
        except Exception as e:
            logger.warning(f"Error adapting campaign intensity: {e}")
            self.campaign_intensity = 1.0
    
    def _get_campaign_recommendation(self) -> str:
        """Get campaign recommendation based on current intensity."""
        if self.campaign_intensity >= 1.2:
            return "aggressive_campaign"
        elif self.campaign_intensity >= 0.8:
            return "standard_campaign"
        else:
            return "conservative_campaign"


class ReassignmentAgent:
    """
    Mock Reassignment Agent that responds to high-risk predictions.
    Demonstrates automatic task redistribution based on forecast alerts.
    """
    
    def __init__(self, agent_id: str = "reassignment_agent_001", api_base_url: str = "http://localhost:8002"):
        self.agent_id = agent_id
        self.api_base_url = api_base_url
    
    def evaluate_reassignment(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if a task should be reassigned based on predictions.
        
        Args:
            task_data: Dictionary containing task information
            
        Returns:
            Reassignment decision and recommendations
        """
        logger.info(f"🔄 ReassignmentAgent evaluating task: {task_data.get('task_id', 'unknown')}")
        
        try:
            # Get agent score for current assignee
            current_agent = task_data.get('assigned_agent', 'unknown')
            score_response = self._get_agent_score(current_agent)
            
            if score_response:
                agent_score = score_response.get('agent_score', {})
                should_reassign = agent_score.get('capacity_status') == 'overloaded'
                
                result = {
                    "agent_id": self.agent_id,
                    "task_id": task_data.get('task_id'),
                    "current_assignee": current_agent,
                    "should_reassign": should_reassign,
                    "reason": agent_score.get('recommendation', 'unknown'),
                    "agent_score": agent_score.get('score', 0.5),
                    "capacity_status": agent_score.get('capacity_status', 'unknown'),
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"✅ Reassignment evaluation: {'reassign' if should_reassign else 'keep'}")
                return result
            else:
                return {
                    "agent_id": self.agent_id,
                    "task_id": task_data.get('task_id'),
                    "should_reassign": False,
                    "reason": "forecast_data_unavailable",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Error evaluating reassignment: {e}")
            return {
                "agent_id": self.agent_id,
                "task_id": task_data.get('task_id'),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_agent_score(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent score from the prediction API."""
        try:
            payload = {"agent_id": agent_id, "current_load": 15}
            response = requests.post(f"{self.api_base_url}/score-agent", json=payload, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Agent scoring API returned status {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Failed to get agent score: {e}")
            return None


# Simulation workflow functions
def run_end_to_end_simulation():
    """
    Run end-to-end simulation: SalesAgent → PredictionAgent → ReassignmentAgent
    """
    logger.info("🚀 Starting end-to-end agent simulation workflow")
    
    # Initialize agents
    sales_agent = SalesAgent()
    marketing_agent = MarketingAgent()
    reassignment_agent = ReassignmentAgent()
    
    # Simulate workflow
    results = {
        "simulation_id": f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "workflow_steps": []
    }
    
    # Step 1: Sales agent processes lead
    lead_data = {
        "lead_id": "lead_001",
        "budget": 15000,
        "urgency": "high",
        "decision_maker": True
    }
    
    sales_result = sales_agent.process_lead(lead_data)
    results["workflow_steps"].append({
        "step": 1,
        "agent": "SalesAgent",
        "action": "process_lead",
        "result": sales_result
    })
    
    # Step 2: Marketing agent plans campaign
    campaign_data = {
        "campaign_id": "campaign_001",
        "budget": 20000,
        "target_reach": 5000
    }
    
    marketing_result = marketing_agent.plan_campaign(campaign_data)
    results["workflow_steps"].append({
        "step": 2,
        "agent": "MarketingAgent",
        "action": "plan_campaign",
        "result": marketing_result
    })
    
    # Step 3: Reassignment agent evaluates task
    task_data = {
        "task_id": "task_001",
        "assigned_agent": "agent_1",
        "priority": "high"
    }
    
    reassignment_result = reassignment_agent.evaluate_reassignment(task_data)
    results["workflow_steps"].append({
        "step": 3,
        "agent": "ReassignmentAgent",
        "action": "evaluate_reassignment",
        "result": reassignment_result
    })
    
    logger.info("✅ End-to-end simulation completed")
    return results
