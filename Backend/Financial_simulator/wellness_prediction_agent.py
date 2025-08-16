"""
Wellness Bot forecasting for financial health and burnout risk
Advanced wellness metrics prediction with Prophet and ARIMA models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class WellnessPredictionAgent:
    """Specialized prediction agent for wellness metrics"""
    
    def __init__(self, use_prophet: bool = True):
        self.use_prophet = use_prophet and PROPHET_AVAILABLE
        self.wellness_metrics = [
            'financial_health_score',
            'burnout_risk',
            'spending_volatility',
            'stress_level_prediction',
            'work_life_balance_score',
            'emotional_wellness_index',
            'financial_anxiety_level',
            'sleep_quality_impact'
        ]
        
        # Wellness-specific thresholds
        self.risk_thresholds = {
            'financial_health_score': 60,  # Below 60 is concerning
            'burnout_risk': 0.7,  # Above 70% is high risk
            'spending_volatility': 0.6,  # Above 60% volatility is unstable
            'stress_level_prediction': 0.7,  # Above 70% stress needs intervention
            'work_life_balance_score': 50,  # Below 50 is poor balance
            'emotional_wellness_index': 60,  # Below 60 needs support
            'financial_anxiety_level': 0.6  # Above 60% anxiety is concerning
        }
    
    def load_wellness_data(self) -> Dict[str, pd.DataFrame]:
        """Load wellness historical data with realistic patterns"""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        n_days = len(dates)
        
        # Wellness patterns
        day_of_week = np.array([d.weekday() for d in dates])
        week_of_year = np.array([d.isocalendar()[1] for d in dates])
        month_of_year = np.array([d.month for d in dates])
        
        # Work stress patterns (higher on weekdays, seasonal variations)
        work_stress = np.where(day_of_week < 5, 1.2, 0.8)  # Higher on weekdays
        seasonal_stress = 1 + 0.3 * np.sin(2 * np.pi * (month_of_year - 1) / 12)  # Holiday stress
        
        # Financial health score (0-100)
        base_health = 70
        market_volatility = 15 * np.sin(2 * np.pi * week_of_year / 26)  # Bi-annual cycles
        personal_events = 10 * np.sin(2 * np.pi * week_of_year / 52)  # Annual events
        
        financial_health = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                base_health + market_volatility + personal_events +
                np.random.normal(0, 8, n_days), 0, 100
            )
        })
        
        # Burnout risk (0-1) - correlated with work stress
        burnout_base = 0.4
        burnout_risk = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                burnout_base + 0.2 * work_stress + 
                0.15 * np.sin(2 * np.pi * week_of_year / 52) +  # Annual burnout cycle
                np.random.normal(0, 0.08, n_days), 0, 1
            )
        })
        
        # Spending volatility (0-1) - higher during holidays and stress periods
        holiday_spending = np.where(
            (month_of_year == 12) | (month_of_year == 1) |  # Holiday season
            (month_of_year == 7) | (month_of_year == 8),    # Summer vacation
            1.4, 1.0
        )
        
        spending_volatility = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                0.3 + 0.2 * holiday_spending + 
                0.15 * burnout_risk['y'] +
                0.1 * np.sin(2 * np.pi * day_of_week / 7) +  # Weekly patterns
                np.random.normal(0, 0.1, n_days), 0, 1
            )
        })
        
        # Stress level prediction (0-1)
        stress_level = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                0.4 + 0.3 * work_stress * seasonal_stress +
                0.2 * burnout_risk['y'] +
                0.1 * spending_volatility['y'] +
                np.random.normal(0, 0.07, n_days), 0, 1
            )
        })
        
        # Work-life balance score (0-100)
        weekend_boost = np.where((day_of_week == 5) | (day_of_week == 6), 20, 0)
        
        work_life_balance = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                60 - 30 * burnout_risk['y'] + weekend_boost +
                10 * np.sin(2 * np.pi * week_of_year / 52) +  # Seasonal variation
                np.random.normal(0, 8, n_days), 0, 100
            )
        })
        
        # Emotional wellness index (0-100)
        emotional_wellness = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                70 + 0.3 * financial_health['y'] - 40 * stress_level['y'] +
                0.2 * work_life_balance['y'] +
                np.random.normal(0, 6, n_days), 0, 100
            )
        })
        
        # Financial anxiety level (0-1)
        financial_anxiety = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                0.5 - 0.006 * financial_health['y'] + 0.4 * spending_volatility['y'] +
                0.3 * stress_level['y'] +
                np.random.normal(0, 0.08, n_days), 0, 1
            )
        })
        
        # Sleep quality impact (0-1, where 1 is poor sleep)
        sleep_quality_impact = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                0.3 + 0.4 * stress_level['y'] + 0.3 * burnout_risk['y'] +
                0.1 * np.sin(2 * np.pi * day_of_week / 7) +  # Weekly sleep patterns
                np.random.normal(0, 0.06, n_days), 0, 1
            )
        })
        
        return {
            'financial_health_score': financial_health,
            'burnout_risk': burnout_risk,
            'spending_volatility': spending_volatility,
            'stress_level_prediction': stress_level,
            'work_life_balance_score': work_life_balance,
            'emotional_wellness_index': emotional_wellness,
            'financial_anxiety_level': financial_anxiety,
            'sleep_quality_impact': sleep_quality_impact
        }
    
    def predict_wellness_outcomes(self, user_id: str, forecast_days: int = 30) -> Dict[str, Any]:
        """Predict wellness outcomes for a specific user"""
        try:
            data = self.load_wellness_data()
            predictions = {}
            
            for metric, df in data.items():
                if self.use_prophet:
                    forecast = self._prophet_forecast(df, forecast_days)
                elif STATSMODELS_AVAILABLE:
                    forecast = self._arima_forecast(df, forecast_days)
                else:
                    forecast = self._simple_forecast(df, forecast_days)
                
                predictions[metric] = forecast
            
            return {
                'user_id': user_id,
                'forecast_period_days': forecast_days,
                'predictions': predictions,
                'insights': self._generate_wellness_insights(predictions),
                'risk_assessment': self._assess_wellness_risks(predictions),
                'recommendations': self._generate_wellness_recommendations(predictions),
                'intervention_priority': self._calculate_intervention_priority(predictions),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error in wellness outcome prediction: {e}")
            return {'error': str(e), 'user_id': user_id}
    
    def _prophet_forecast(self, df: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Generate forecast using Prophet"""
        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(df)
            
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            return {
                'forecast_values': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records'),
                'mean_forecast': forecast['yhat'].tail(periods).mean(),
                'trend': 'increasing' if forecast['yhat'].tail(periods).iloc[-1] > forecast['yhat'].tail(periods).iloc[0] else 'decreasing',
                'confidence_interval': {
                    'lower': forecast['yhat_lower'].tail(periods).mean(),
                    'upper': forecast['yhat_upper'].tail(periods).mean()
                },
                'model_used': 'prophet'
            }
        except Exception as e:
            return {'error': str(e), 'model_used': 'prophet_failed'}
    
    def _arima_forecast(self, df: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Generate forecast using ARIMA"""
        try:
            model = ARIMA(df['y'], order=(1, 1, 1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=periods)
            
            return {
                'forecast_values': forecast.tolist(),
                'mean_forecast': forecast.mean(),
                'trend': 'increasing' if forecast.iloc[-1] > forecast.iloc[0] else 'decreasing',
                'model_used': 'arima'
            }
        except Exception as e:
            return {'error': str(e), 'model_used': 'arima_failed'}
    
    def _simple_forecast(self, df: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Simple moving average forecast"""
        recent_mean = df['y'].tail(7).mean()
        trend = df['y'].tail(14).diff().mean()
        
        forecast_values = [recent_mean + trend * i for i in range(1, periods + 1)]
        
        return {
            'forecast_values': forecast_values,
            'mean_forecast': np.mean(forecast_values),
            'trend': 'increasing' if trend > 0 else 'decreasing',
            'model_used': 'simple_moving_average'
        }
    
    def _generate_wellness_insights(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate wellness insights from predictions"""
        insights = []
        
        try:
            # Financial health insights
            financial_health = predictions.get('financial_health_score', {}).get('mean_forecast', 70)
            if financial_health < 50:
                insights.append("Financial health score critically low - immediate financial counseling recommended")
            elif financial_health < 60:
                insights.append("Financial health declining - consider budget review and financial planning")
            
            # Burnout risk insights
            burnout_risk = predictions.get('burnout_risk', {}).get('mean_forecast', 0.5)
            if burnout_risk > 0.8:
                insights.append("Severe burnout risk detected - immediate intervention required")
            elif burnout_risk > 0.7:
                insights.append("High burnout risk - consider workload reduction and stress management")
            
            # Stress level insights
            stress_level = predictions.get('stress_level_prediction', {}).get('mean_forecast', 0.5)
            if stress_level > 0.7:
                insights.append("High stress levels predicted - implement stress reduction techniques")
            
            # Work-life balance insights
            work_life_balance = predictions.get('work_life_balance_score', {}).get('mean_forecast', 60)
            if work_life_balance < 40:
                insights.append("Poor work-life balance - prioritize personal time and boundaries")
            
            # Financial anxiety insights
            financial_anxiety = predictions.get('financial_anxiety_level', {}).get('mean_forecast', 0.5)
            if financial_anxiety > 0.7:
                insights.append("High financial anxiety - consider financial therapy or counseling")
            
        except Exception as e:
            insights.append("Unable to generate detailed wellness insights")
        
        return insights
    
    def _assess_wellness_risks(self, predictions: Dict[str, Any]) -> Dict[str, str]:
        """Assess wellness risks based on predictions"""
        risks = {}
        
        for metric, threshold in self.risk_thresholds.items():
            if metric in predictions:
                forecast_value = predictions[metric].get('mean_forecast', 0.5)
                
                if metric in ['burnout_risk', 'spending_volatility', 'stress_level_prediction', 'financial_anxiety_level', 'sleep_quality_impact']:
                    # Higher values are worse for these metrics
                    if forecast_value > threshold:
                        risks[metric] = 'HIGH'
                    elif forecast_value > threshold * 0.7:
                        risks[metric] = 'MEDIUM'
                    else:
                        risks[metric] = 'LOW'
                else:
                    # Lower values are worse for these metrics
                    if forecast_value < threshold:
                        risks[metric] = 'HIGH'
                    elif forecast_value < threshold * 1.3:
                        risks[metric] = 'MEDIUM'
                    else:
                        risks[metric] = 'LOW'
        
        return risks
    
    def _generate_wellness_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate wellness recommendations"""
        recommendations = []
        
        # Financial health recommendations
        financial_health = predictions.get('financial_health_score', {}).get('mean_forecast', 70)
        if financial_health < 60:
            recommendations.append("Schedule financial planning session with advisor")
            recommendations.append("Create emergency fund if not already established")
            recommendations.append("Review and optimize monthly budget")
        
        # Burnout recommendations
        burnout_risk = predictions.get('burnout_risk', {}).get('mean_forecast', 0.5)
        if burnout_risk > 0.7:
            recommendations.append("Take immediate time off for mental health recovery")
            recommendations.append("Discuss workload reduction with supervisor")
            recommendations.append("Implement daily mindfulness or meditation practice")
        
        # Stress management recommendations
        stress_level = predictions.get('stress_level_prediction', {}).get('mean_forecast', 0.5)
        if stress_level > 0.6:
            recommendations.append("Practice stress reduction techniques (deep breathing, yoga)")
            recommendations.append("Establish regular exercise routine")
            recommendations.append("Consider professional stress management counseling")
        
        # Work-life balance recommendations
        work_life_balance = predictions.get('work_life_balance_score', {}).get('mean_forecast', 60)
        if work_life_balance < 50:
            recommendations.append("Set clear boundaries between work and personal time")
            recommendations.append("Schedule regular personal activities and hobbies")
            recommendations.append("Practice saying 'no' to non-essential work commitments")
        
        # Financial anxiety recommendations
        financial_anxiety = predictions.get('financial_anxiety_level', {}).get('mean_forecast', 0.5)
        if financial_anxiety > 0.6:
            recommendations.append("Practice financial mindfulness and gratitude exercises")
            recommendations.append("Limit checking financial accounts to specific times")
            recommendations.append("Consider financial therapy or support groups")
        
        if not recommendations:
            recommendations.append("Current wellness trajectory is positive - maintain healthy habits")
            recommendations.append("Continue regular self-care and wellness practices")
        
        return recommendations
    
    def _calculate_intervention_priority(self, predictions: Dict[str, Any]) -> str:
        """Calculate intervention priority based on risk levels"""
        high_risk_count = 0
        medium_risk_count = 0
        
        risks = self._assess_wellness_risks(predictions)
        
        for risk_level in risks.values():
            if risk_level == 'HIGH':
                high_risk_count += 1
            elif risk_level == 'MEDIUM':
                medium_risk_count += 1
        
        if high_risk_count >= 3:
            return 'CRITICAL'
        elif high_risk_count >= 1:
            return 'HIGH'
        elif medium_risk_count >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'
