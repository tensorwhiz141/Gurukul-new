"""
Domain-Specific Forecasting Integration
Combines Edumentor and Wellness prediction agents for comprehensive forecasting
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

try:
    from edumentor_prediction_agent import EdumentorPredictionAgent
    EDUMENTOR_AVAILABLE = True
except ImportError as e:
    EDUMENTOR_AVAILABLE = False
    logging.warning(f"Edumentor agent not available: {e}")

try:
    from wellness_prediction_agent import WellnessPredictionAgent
    WELLNESS_AVAILABLE = True
except ImportError as e:
    WELLNESS_AVAILABLE = False
    logging.warning(f"Wellness agent not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastRequest(BaseModel):
    user_id: str
    domain: str  # 'edumentor', 'wellness', or 'combined'
    forecast_days: int = 30
    metrics: Optional[List[str]] = None

class DomainSpecificForecasting:
    """Main class for domain-specific forecasting"""
    
    def __init__(self):
        self.edumentor_agent = None
        self.wellness_agent = None

        if EDUMENTOR_AVAILABLE:
            self.edumentor_agent = EdumentorPredictionAgent(use_prophet=True)
            logger.info("Edumentor agent initialized")
        else:
            logger.warning("Edumentor agent not available")

        if WELLNESS_AVAILABLE:
            self.wellness_agent = WellnessPredictionAgent(use_prophet=True)
            logger.info("Wellness agent initialized")
        else:
            logger.warning("Wellness agent not available")

        logger.info(f"Domain-specific forecasting initialized (Edumentor: {EDUMENTOR_AVAILABLE}, Wellness: {WELLNESS_AVAILABLE})")
    
    async def generate_edumentor_forecast(self, user_id: str, forecast_days: int = 30) -> Dict[str, Any]:
        """Generate educational forecasting"""
        if not self.edumentor_agent:
            raise HTTPException(status_code=503, detail="Edumentor agent not available")

        try:
            logger.info(f"Generating Edumentor forecast for user {user_id}")

            result = self.edumentor_agent.predict_learning_outcomes(
                student_id=user_id,
                forecast_days=forecast_days
            )
            
            # Add domain-specific metadata
            result['domain'] = 'edumentor'
            result['forecast_type'] = 'educational_metrics'
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Edumentor forecasting: {e}")
            raise HTTPException(status_code=500, detail=f"Edumentor forecasting failed: {str(e)}")
    
    async def generate_wellness_forecast(self, user_id: str, forecast_days: int = 30) -> Dict[str, Any]:
        """Generate wellness forecasting"""
        if not self.wellness_agent:
            raise HTTPException(status_code=503, detail="Wellness agent not available")

        try:
            logger.info(f"Generating Wellness forecast for user {user_id}")

            result = self.wellness_agent.predict_wellness_outcomes(
                user_id=user_id,
                forecast_days=forecast_days
            )
            
            # Add domain-specific metadata
            result['domain'] = 'wellness'
            result['forecast_type'] = 'wellness_metrics'
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Wellness forecasting: {e}")
            raise HTTPException(status_code=500, detail=f"Wellness forecasting failed: {str(e)}")
    
    async def generate_combined_forecast(self, user_id: str, forecast_days: int = 30) -> Dict[str, Any]:
        """Generate combined forecasting from both domains"""
        try:
            logger.info(f"Generating combined forecast for user {user_id}")
            
            # Run both forecasts concurrently
            edumentor_task = asyncio.create_task(
                self.generate_edumentor_forecast(user_id, forecast_days)
            )
            wellness_task = asyncio.create_task(
                self.generate_wellness_forecast(user_id, forecast_days)
            )
            
            edumentor_result, wellness_result = await asyncio.gather(
                edumentor_task, wellness_task, return_exceptions=True
            )
            
            # Handle potential errors
            if isinstance(edumentor_result, Exception):
                logger.error(f"Edumentor forecast failed: {edumentor_result}")
                edumentor_result = {"error": str(edumentor_result)}
            
            if isinstance(wellness_result, Exception):
                logger.error(f"Wellness forecast failed: {wellness_result}")
                wellness_result = {"error": str(wellness_result)}
            
            # Combine results
            combined_result = {
                'user_id': user_id,
                'domain': 'combined',
                'forecast_type': 'multi_domain_metrics',
                'forecast_period_days': forecast_days,
                'edumentor_forecast': edumentor_result,
                'wellness_forecast': wellness_result,
                'cross_domain_insights': self._generate_cross_domain_insights(
                    edumentor_result, wellness_result
                ),
                'holistic_recommendations': self._generate_holistic_recommendations(
                    edumentor_result, wellness_result
                ),
                'overall_risk_assessment': self._assess_overall_risk(
                    edumentor_result, wellness_result
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error in combined forecasting: {e}")
            raise HTTPException(status_code=500, detail=f"Combined forecasting failed: {str(e)}")
    
    def _generate_cross_domain_insights(self, edumentor_result: Dict, wellness_result: Dict) -> List[str]:
        """Generate insights that span both domains"""
        insights = []
        
        try:
            # Check for correlations between learning and wellness
            if 'predictions' in edumentor_result and 'predictions' in wellness_result:
                
                # Learning stress vs wellness stress
                learning_stress = edumentor_result['predictions'].get('learning_progress_risk', {}).get('mean_forecast', 0.5)
                wellness_stress = wellness_result['predictions'].get('stress_level_prediction', {}).get('mean_forecast', 0.5)
                
                if learning_stress > 0.7 and wellness_stress > 0.7:
                    insights.append("High stress detected in both learning and wellness domains - comprehensive stress management needed")
                
                # Engagement vs work-life balance
                engagement_risk = edumentor_result['predictions'].get('engagement_drop_risk', {}).get('mean_forecast', 0.5)
                work_life_balance = wellness_result['predictions'].get('work_life_balance_score', {}).get('mean_forecast', 60)
                
                if engagement_risk > 0.6 and work_life_balance < 50:
                    insights.append("Poor work-life balance may be affecting learning engagement")
                
                # Learning velocity vs burnout
                learning_velocity = edumentor_result['predictions'].get('learning_velocity', {}).get('mean_forecast', 0.8)
                burnout_risk = wellness_result['predictions'].get('burnout_risk', {}).get('mean_forecast', 0.5)
                
                if learning_velocity < 0.6 and burnout_risk > 0.7:
                    insights.append("Burnout risk may be impacting learning performance")
                
                # Financial health vs learning progress
                financial_health = wellness_result['predictions'].get('financial_health_score', {}).get('mean_forecast', 70)
                quiz_completion = edumentor_result['predictions'].get('quiz_completion_rate', {}).get('mean_forecast', 0.8)
                
                if financial_health < 50 and quiz_completion < 0.6:
                    insights.append("Financial stress may be affecting learning focus and completion rates")
            
            if not insights:
                insights.append("Cross-domain analysis shows balanced performance across learning and wellness metrics")
        
        except Exception as e:
            logger.error(f"Error generating cross-domain insights: {e}")
            insights.append("Unable to generate cross-domain insights due to data processing error")
        
        return insights
    
    def _generate_holistic_recommendations(self, edumentor_result: Dict, wellness_result: Dict) -> List[str]:
        """Generate holistic recommendations considering both domains"""
        recommendations = []
        
        try:
            # Combine recommendations from both domains
            edumentor_recs = edumentor_result.get('recommendations', [])
            wellness_recs = wellness_result.get('recommendations', [])
            
            # Add domain-specific prefixes
            for rec in edumentor_recs:
                recommendations.append(f"Learning: {rec}")
            
            for rec in wellness_recs:
                recommendations.append(f"Wellness: {rec}")
            
            # Add holistic recommendations
            if 'predictions' in edumentor_result and 'predictions' in wellness_result:
                
                # Integrated stress management
                learning_stress = edumentor_result['predictions'].get('learning_progress_risk', {}).get('mean_forecast', 0.5)
                wellness_stress = wellness_result['predictions'].get('stress_level_prediction', {}).get('mean_forecast', 0.5)
                
                if learning_stress > 0.6 and wellness_stress > 0.6:
                    recommendations.append("Holistic: Implement integrated stress management covering both academic and personal life")
                    recommendations.append("Holistic: Consider mindfulness practices that benefit both learning focus and overall wellness")
                
                # Time management integration
                study_frequency = edumentor_result['predictions'].get('study_session_frequency', {}).get('mean_forecast', 2.5)
                work_life_balance = wellness_result['predictions'].get('work_life_balance_score', {}).get('mean_forecast', 60)
                
                if study_frequency > 5 and work_life_balance < 50:
                    recommendations.append("Holistic: Balance intensive study periods with adequate rest and personal time")
                
                # Financial wellness and learning
                financial_health = wellness_result['predictions'].get('financial_health_score', {}).get('mean_forecast', 70)
                learning_velocity = edumentor_result['predictions'].get('learning_velocity', {}).get('mean_forecast', 0.8)
                
                if financial_health < 60 and learning_velocity < 0.7:
                    recommendations.append("Holistic: Address financial concerns to reduce stress and improve learning focus")
                    recommendations.append("Holistic: Consider financial literacy education as part of learning curriculum")
        
        except Exception as e:
            logger.error(f"Error generating holistic recommendations: {e}")
            recommendations.append("Unable to generate holistic recommendations due to processing error")
        
        return recommendations
    
    def _assess_overall_risk(self, edumentor_result: Dict, wellness_result: Dict) -> Dict[str, Any]:
        """Assess overall risk across both domains"""
        try:
            edumentor_risks = edumentor_result.get('risk_assessment', {})
            wellness_risks = wellness_result.get('risk_assessment', {})
            
            # Count high risks across domains
            total_high_risks = 0
            total_medium_risks = 0
            
            for risk_level in edumentor_risks.values():
                if risk_level == 'HIGH':
                    total_high_risks += 1
                elif risk_level == 'MEDIUM':
                    total_medium_risks += 1
            
            for risk_level in wellness_risks.values():
                if risk_level == 'HIGH':
                    total_high_risks += 1
                elif risk_level == 'MEDIUM':
                    total_medium_risks += 1
            
            # Determine overall risk level
            if total_high_risks >= 4:
                overall_risk = 'CRITICAL'
            elif total_high_risks >= 2:
                overall_risk = 'HIGH'
            elif total_high_risks >= 1 or total_medium_risks >= 4:
                overall_risk = 'MEDIUM'
            else:
                overall_risk = 'LOW'
            
            return {
                'overall_risk_level': overall_risk,
                'total_high_risks': total_high_risks,
                'total_medium_risks': total_medium_risks,
                'edumentor_risks': edumentor_risks,
                'wellness_risks': wellness_risks,
                'intervention_priority': self._determine_intervention_priority(overall_risk)
            }
        
        except Exception as e:
            logger.error(f"Error assessing overall risk: {e}")
            return {
                'overall_risk_level': 'UNKNOWN',
                'error': str(e)
            }
    
    def _determine_intervention_priority(self, overall_risk: str) -> List[str]:
        """Determine intervention priorities based on overall risk"""
        if overall_risk == 'CRITICAL':
            return [
                "Immediate comprehensive assessment required",
                "Coordinate between educational and wellness support teams",
                "Consider temporary reduction in academic load",
                "Implement crisis intervention protocols"
            ]
        elif overall_risk == 'HIGH':
            return [
                "Schedule comprehensive support meeting within 48 hours",
                "Implement targeted interventions in high-risk areas",
                "Increase monitoring frequency",
                "Coordinate support across domains"
            ]
        elif overall_risk == 'MEDIUM':
            return [
                "Schedule check-in within one week",
                "Implement preventive measures",
                "Monitor trends closely",
                "Provide targeted resources"
            ]
        else:
            return [
                "Continue regular monitoring",
                "Maintain current support level",
                "Focus on prevention and wellness maintenance"
            ]

# Initialize the forecasting system
domain_forecasting = DomainSpecificForecasting()

# FastAPI integration functions
async def forecast_edumentor(request: ForecastRequest):
    """API endpoint for Edumentor forecasting"""
    return await domain_forecasting.generate_edumentor_forecast(
        user_id=request.user_id,
        forecast_days=request.forecast_days
    )

async def forecast_wellness(request: ForecastRequest):
    """API endpoint for Wellness forecasting"""
    return await domain_forecasting.generate_wellness_forecast(
        user_id=request.user_id,
        forecast_days=request.forecast_days
    )

async def forecast_combined(request: ForecastRequest):
    """API endpoint for combined forecasting"""
    return await domain_forecasting.generate_combined_forecast(
        user_id=request.user_id,
        forecast_days=request.forecast_days
    )
