"""
Edumentor-specific forecasting for learning progress and quiz delays
Advanced educational metrics prediction with Prophet and ARIMA models
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
    logging.warning("Prophet not available, using ARIMA fallback")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available")

try:
    from .prediction_agent import PredictionAgent
    BASE_AGENT_AVAILABLE = True
except ImportError:
    # Fallback for standalone usage
    BASE_AGENT_AVAILABLE = False
    class PredictionAgent:
        def __init__(self, use_prophet=True):
            self.use_prophet = use_prophet

class EdumentorPredictionAgent(PredictionAgent):
    """Specialized prediction agent for educational metrics"""
    
    def __init__(self, use_prophet: bool = True):
        super().__init__(use_prophet)
        self.learning_metrics = [
            'quiz_completion_rate',
            'learning_progress_risk', 
            'concept_mastery_decline',
            'engagement_drop_risk',
            'study_session_frequency',
            'knowledge_retention_score',
            'learning_velocity',
            'difficulty_adaptation_rate'
        ]
        
        # Educational-specific thresholds
        self.risk_thresholds = {
            'quiz_completion_rate': 0.6,  # Below 60% is concerning
            'learning_progress_risk': 0.7,  # Above 70% risk is high
            'concept_mastery_decline': 0.5,  # Above 50% decline is significant
            'engagement_drop_risk': 0.6,  # Above 60% risk needs intervention
            'knowledge_retention_score': 0.7,  # Below 70% needs review
            'learning_velocity': 0.5  # Below 50% of expected pace
        }
    
    def load_edumentor_data(self) -> Dict[str, pd.DataFrame]:
        """Load educational historical data with realistic patterns"""
        # Generate 1 year of daily data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        n_days = len(dates)
        
        # Create time-based features for educational patterns
        day_of_week = np.array([d.weekday() for d in dates])  # 0=Monday, 6=Sunday
        week_of_year = np.array([d.isocalendar()[1] for d in dates])
        
        # Weekend effect (lower activity on weekends)
        weekend_effect = np.where((day_of_week == 5) | (day_of_week == 6), 0.7, 1.0)
        
        # Semester patterns (lower activity during breaks)
        semester_pattern = np.where(
            (week_of_year < 3) | (week_of_year > 50) |  # Winter break
            ((week_of_year > 13) & (week_of_year < 17)) |  # Spring break
            ((week_of_year > 22) & (week_of_year < 35)),  # Summer break
            0.6, 1.0
        )
        
        # Quiz completion rate (0-1) with educational patterns
        base_completion = 0.8
        seasonal_variation = 0.15 * np.sin(2 * np.pi * week_of_year / 52)
        weekly_pattern = 0.1 * np.sin(2 * np.pi * day_of_week / 7)
        noise = np.random.normal(0, 0.08, n_days)
        
        quiz_completion = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                base_completion + seasonal_variation + weekly_pattern + noise,
                0, 1
            ) * weekend_effect * semester_pattern
        })
        
        # Learning progress risk (0-1) - inverse of completion rate
        progress_risk = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                1 - quiz_completion['y'] + np.random.normal(0, 0.05, n_days),
                0, 1
            )
        })
        
        # Concept mastery decline (0-1) - follows forgetting curve
        forgetting_curve = 0.3 + 0.4 * np.exp(-np.arange(n_days) / 30)
        concept_mastery = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                forgetting_curve + 0.2 * np.sin(2 * np.pi * np.arange(n_days) / 14) +
                np.random.normal(0, 0.06, n_days),
                0, 1
            )
        })
        
        # Engagement drop risk (0-1) - correlated with completion rate
        engagement_risk = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                0.4 + 0.3 * (1 - quiz_completion['y']) + 
                0.2 * np.sin(2 * np.pi * day_of_week / 7) +
                np.random.normal(0, 0.07, n_days),
                0, 1
            )
        })
        
        # Study session frequency (sessions per day)
        study_frequency = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                2.5 + 1.5 * np.sin(2 * np.pi * day_of_week / 7) +
                0.8 * np.sin(2 * np.pi * week_of_year / 52) +
                np.random.normal(0, 0.3, n_days),
                0, 8
            ) * weekend_effect * semester_pattern
        })
        
        # Knowledge retention score (0-1)
        retention_score = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                0.75 + 0.2 * quiz_completion['y'] - 0.1 * concept_mastery['y'] +
                np.random.normal(0, 0.05, n_days),
                0, 1
            )
        })
        
        # Learning velocity (relative to expected pace)
        learning_velocity = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                0.8 + 0.3 * quiz_completion['y'] + 
                0.2 * np.sin(2 * np.pi * week_of_year / 52) +
                np.random.normal(0, 0.1, n_days),
                0, 2
            )
        })
        
        # Difficulty adaptation rate (0-1)
        adaptation_rate = pd.DataFrame({
            'ds': dates,
            'y': np.clip(
                0.6 + 0.2 * learning_velocity['y'] + 
                0.15 * np.sin(2 * np.pi * np.arange(n_days) / 21) +
                np.random.normal(0, 0.08, n_days),
                0, 1
            )
        })
        
        return {
            'quiz_completion_rate': quiz_completion,
            'learning_progress_risk': progress_risk,
            'concept_mastery_decline': concept_mastery,
            'engagement_drop_risk': engagement_risk,
            'study_session_frequency': study_frequency,
            'knowledge_retention_score': retention_score,
            'learning_velocity': learning_velocity,
            'difficulty_adaptation_rate': adaptation_rate
        }
    
    def predict_learning_outcomes(self, student_id: str, forecast_days: int = 30) -> Dict[str, Any]:
        """Predict learning outcomes for a specific student"""
        try:
            data = self.load_edumentor_data()
            predictions = {}
            
            for metric, df in data.items():
                if self.use_prophet and PROPHET_AVAILABLE:
                    forecast = self._prophet_forecast(df, forecast_days)
                elif STATSMODELS_AVAILABLE:
                    forecast = self._arima_forecast(df, forecast_days)
                else:
                    forecast = self._simple_forecast(df, forecast_days)
                
                predictions[metric] = forecast
            
            # Generate educational insights
            insights = self._generate_educational_insights(predictions)
            
            return {
                'student_id': student_id,
                'forecast_period_days': forecast_days,
                'predictions': predictions,
                'insights': insights,
                'risk_assessment': self._assess_learning_risks(predictions),
                'recommendations': self._generate_learning_recommendations(predictions),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error in learning outcome prediction: {e}")
            return {
                'error': str(e),
                'student_id': student_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_educational_insights(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate educational insights from predictions"""
        insights = []
        
        try:
            # Quiz completion trend
            quiz_trend = predictions.get('quiz_completion_rate', {}).get('trend', 'stable')
            if quiz_trend == 'decreasing':
                insights.append("Quiz completion rate is declining - consider intervention strategies")
            elif quiz_trend == 'increasing':
                insights.append("Quiz completion rate is improving - current approach is effective")
            
            # Learning progress risk
            progress_risk = predictions.get('learning_progress_risk', {}).get('mean_forecast', 0.5)
            if progress_risk > self.risk_thresholds['learning_progress_risk']:
                insights.append("High learning progress risk detected - immediate support recommended")
            
            # Engagement patterns
            engagement_risk = predictions.get('engagement_drop_risk', {}).get('mean_forecast', 0.5)
            if engagement_risk > self.risk_thresholds['engagement_drop_risk']:
                insights.append("Student engagement at risk - consider gamification or content variety")
            
            # Knowledge retention
            retention = predictions.get('knowledge_retention_score', {}).get('mean_forecast', 0.7)
            if retention < self.risk_thresholds['knowledge_retention_score']:
                insights.append("Knowledge retention below optimal - implement spaced repetition")
            
        except Exception as e:
            logging.error(f"Error generating educational insights: {e}")
            insights.append("Unable to generate detailed insights due to data processing error")
        
        return insights
    
    def _assess_learning_risks(self, predictions: Dict[str, Any]) -> Dict[str, str]:
        """Assess learning risks based on predictions"""
        risks = {}
        
        try:
            for metric, threshold in self.risk_thresholds.items():
                if metric in predictions:
                    forecast_value = predictions[metric].get('mean_forecast', 0.5)
                    
                    if metric in ['learning_progress_risk', 'concept_mastery_decline', 'engagement_drop_risk']:
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
        
        except Exception as e:
            logging.error(f"Error assessing learning risks: {e}")
            risks['assessment_error'] = 'Unable to assess risks'
        
        return risks
    
    def _generate_learning_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations based on predictions"""
        recommendations = []
        
        try:
            # Quiz completion recommendations
            quiz_rate = predictions.get('quiz_completion_rate', {}).get('mean_forecast', 0.8)
            if quiz_rate < 0.7:
                recommendations.append("Implement adaptive quiz difficulty to improve completion rates")
                recommendations.append("Provide immediate feedback and hints during quizzes")
            
            # Learning velocity recommendations
            velocity = predictions.get('learning_velocity', {}).get('mean_forecast', 0.8)
            if velocity < 0.6:
                recommendations.append("Break down complex concepts into smaller, manageable chunks")
                recommendations.append("Increase interactive elements and hands-on practice")
            
            # Engagement recommendations
            engagement_risk = predictions.get('engagement_drop_risk', {}).get('mean_forecast', 0.5)
            if engagement_risk > 0.6:
                recommendations.append("Introduce gamification elements and achievement badges")
                recommendations.append("Vary content delivery methods (video, interactive, text)")
                recommendations.append("Implement peer learning and discussion forums")
            
            # Retention recommendations
            retention = predictions.get('knowledge_retention_score', {}).get('mean_forecast', 0.7)
            if retention < 0.7:
                recommendations.append("Implement spaced repetition for key concepts")
                recommendations.append("Add periodic review sessions and knowledge checks")
                recommendations.append("Create concept maps and visual learning aids")
            
            if not recommendations:
                recommendations.append("Current learning trajectory is positive - maintain current approach")
                recommendations.append("Consider advanced challenges to accelerate learning")
        
        except Exception as e:
            logging.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to processing error")
        
        return recommendations

    def _prophet_forecast(self, df: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Generate forecast using Prophet"""
        try:
            from prophet import Prophet
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
                'model_used': 'prophet'
            }
        except Exception as e:
            return {'error': str(e), 'model_used': 'prophet_failed'}

    def _arima_forecast(self, df: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Generate forecast using ARIMA"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
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
