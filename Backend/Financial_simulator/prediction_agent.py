"""
Forecast Engine v2 - Advanced Time Series Forecasting with Prophet
Core prediction agent for delay risk, escalation likelihood, and daily agent load forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from pathlib import Path

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionAgent:
    """
    Advanced forecasting agent using Prophet (preferred) or ARIMA for time series prediction.
    Generates predictions for:
    - Delay risk per task
    - Escalation likelihood
    - Daily agent load forecasting
    """
    
    def __init__(self, use_prophet: bool = True):
        """
        Initialize the prediction agent.
        
        Args:
            use_prophet: Whether to use Prophet (True) or ARIMA (False) for forecasting
        """
        self.use_prophet = use_prophet and PROPHET_AVAILABLE
        self.models = {}
        self.historical_data = {}
        self.predictions = {}
        
        if not PROPHET_AVAILABLE and use_prophet:
            logger.warning("Prophet not available, falling back to ARIMA")
            self.use_prophet = False
    
    def load_task_history_data(self, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """
        Load and process task history data from JSON files.
        
        Args:
            data_dir: Directory containing historical data files
            
        Returns:
            Dictionary of processed DataFrames for different metrics
        """
        data_path = Path(data_dir)
        processed_data = {}
        
        try:
            # Load user history files
            history_files = list(data_path.glob("*_person_history.json"))
            
            if not history_files:
                logger.warning(f"No history files found in {data_dir}")
                return self._generate_mock_data()
            
            all_data = []
            for file_path in history_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        elif isinstance(data, dict):
                            all_data.append(data)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            
            if not all_data:
                return self._generate_mock_data()
            
            # Process data for forecasting
            processed_data = self._process_historical_data(all_data)
            
        except Exception as e:
            logger.error(f"Error loading task history data: {e}")
            processed_data = self._generate_mock_data()
        
        self.historical_data = processed_data
        return processed_data
    
    def _generate_mock_data(self) -> Dict[str, pd.DataFrame]:
        """Generate mock historical data for testing purposes."""
        logger.info("Generating mock historical data for forecasting")
        
        # Generate 90 days of mock data
        dates = pd.date_range(start=datetime.now() - timedelta(days=90), 
                             end=datetime.now(), freq='D')
        
        np.random.seed(42)  # For reproducible results
        
        # Mock delay risk data (0-1 scale)
        delay_risk = pd.DataFrame({
            'ds': dates,
            'y': np.random.beta(2, 5, len(dates)) + 0.1 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
        })
        
        # Mock escalation likelihood (0-1 scale)
        escalation_risk = pd.DataFrame({
            'ds': dates,
            'y': np.random.beta(1.5, 8, len(dates)) + 0.05 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30)
        })
        
        # Mock daily agent load (number of tasks)
        agent_load = pd.DataFrame({
            'ds': dates,
            'y': np.random.poisson(15, len(dates)) + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
        })
        
        return {
            'delay_risk': delay_risk,
            'escalation_likelihood': escalation_risk,
            'daily_agent_load': agent_load
        }
    
    def _process_historical_data(self, raw_data: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Process raw historical data into time series format.
        
        Args:
            raw_data: List of historical records
            
        Returns:
            Dictionary of processed time series DataFrames
        """
        # Extract relevant metrics from historical data
        processed = {
            'delay_risk': [],
            'escalation_likelihood': [],
            'daily_agent_load': []
        }
        
        for record in raw_data:
            try:
                # Extract timestamp
                timestamp = record.get('timestamp', datetime.now().isoformat())
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)
                
                # Extract delay risk indicators
                delay_indicators = record.get('discipline_score', 0.8)
                if isinstance(delay_indicators, (int, float)):
                    delay_risk = max(0, min(1, 1 - delay_indicators))  # Invert discipline score
                else:
                    delay_risk = 0.3  # Default moderate risk
                
                # Extract escalation likelihood
                escalation_score = record.get('behavior_score', 0.7)
                if isinstance(escalation_score, (int, float)):
                    escalation_likelihood = max(0, min(1, 1 - escalation_score))
                else:
                    escalation_likelihood = 0.2  # Default low escalation
                
                # Extract agent load (approximate from financial complexity)
                financial_complexity = record.get('total_expenses', 1000)
                if isinstance(financial_complexity, (int, float)):
                    agent_load = min(50, max(5, financial_complexity / 100))  # Scale to reasonable range
                else:
                    agent_load = 15  # Default load
                
                processed['delay_risk'].append({'ds': timestamp, 'y': delay_risk})
                processed['escalation_likelihood'].append({'ds': timestamp, 'y': escalation_likelihood})
                processed['daily_agent_load'].append({'ds': timestamp, 'y': agent_load})
                
            except Exception as e:
                logger.error(f"Error processing record: {e}")
                continue
        
        # Convert to DataFrames
        result = {}
        for key, data_list in processed.items():
            if data_list:
                df = pd.DataFrame(data_list)
                df['ds'] = pd.to_datetime(df['ds'])
                df = df.sort_values('ds').reset_index(drop=True)
                result[key] = df
            else:
                logger.warning(f"No data available for {key}, using mock data")
                result[key] = self._generate_mock_data()[key]
        
        return result
    
    def train_prophet_model(self, data: pd.DataFrame, metric_name: str) -> Prophet:
        """
        Train a Prophet model for the given time series data.
        
        Args:
            data: DataFrame with 'ds' (date) and 'y' (value) columns
            metric_name: Name of the metric being forecasted
            
        Returns:
            Trained Prophet model
        """
        try:
            # Configure Prophet model based on metric type
            if metric_name in ['delay_risk', 'escalation_likelihood']:
                # For probability metrics, use logistic growth
                model = Prophet(
                    growth='logistic',
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    seasonality_mode='additive'
                )
                # Set capacity for logistic growth
                data['cap'] = 1.0
                data['floor'] = 0.0
            else:
                # For count metrics like agent load
                model = Prophet(
                    growth='linear',
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    seasonality_mode='multiplicative'
                )
            
            # Fit the model
            model.fit(data)
            logger.info(f"Prophet model trained successfully for {metric_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error training Prophet model for {metric_name}: {e}")
            raise
    
    def train_arima_model(self, data: pd.DataFrame, metric_name: str) -> ARIMA:
        """
        Train an ARIMA model for the given time series data.
        
        Args:
            data: DataFrame with time series data
            metric_name: Name of the metric being forecasted
            
        Returns:
            Fitted ARIMA model
        """
        try:
            # Use simple ARIMA(1,1,1) configuration
            # In production, you might want to use auto_arima for optimal parameters
            model = ARIMA(data['y'], order=(1, 1, 1))
            fitted_model = model.fit()
            logger.info(f"ARIMA model trained successfully for {metric_name}")
            return fitted_model
            
        except Exception as e:
            logger.error(f"Error training ARIMA model for {metric_name}: {e}")
            # Fallback to simpler model
            try:
                model = ARIMA(data['y'], order=(0, 1, 0))
                fitted_model = model.fit()
                logger.info(f"Fallback ARIMA model trained for {metric_name}")
                return fitted_model
            except Exception as e2:
                logger.error(f"Fallback ARIMA model also failed for {metric_name}: {e2}")
                raise

    def generate_forecasts(self, forecast_days: int = 30) -> Dict[str, Dict]:
        """
        Generate forecasts for all metrics using trained models.

        Args:
            forecast_days: Number of days to forecast ahead

        Returns:
            Dictionary containing forecasts for each metric
        """
        if not self.historical_data:
            logger.warning("No historical data loaded, loading default data")
            self.load_task_history_data()

        forecasts = {}

        for metric_name, data in self.historical_data.items():
            try:
                if self.use_prophet:
                    forecast = self._generate_prophet_forecast(data, metric_name, forecast_days)
                else:
                    forecast = self._generate_arima_forecast(data, metric_name, forecast_days)

                forecasts[metric_name] = forecast
                logger.info(f"Generated forecast for {metric_name}")

            except Exception as e:
                logger.error(f"Error generating forecast for {metric_name}: {e}")
                # Generate fallback forecast
                forecasts[metric_name] = self._generate_fallback_forecast(data, metric_name, forecast_days)

        self.predictions = forecasts
        return forecasts

    def _generate_prophet_forecast(self, data: pd.DataFrame, metric_name: str, forecast_days: int) -> Dict:
        """Generate forecast using Prophet model."""
        model = self.train_prophet_model(data, metric_name)

        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_days)

        # Add capacity for logistic growth models
        if metric_name in ['delay_risk', 'escalation_likelihood']:
            future['cap'] = 1.0
            future['floor'] = 0.0

        # Generate forecast
        forecast = model.predict(future)

        # Extract forecast data
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)

        return {
            'model_type': 'prophet',
            'metric': metric_name,
            'forecast_days': forecast_days,
            'predictions': forecast_data.to_dict('records'),
            'confidence_intervals': True,
            'last_historical_value': float(data['y'].iloc[-1]),
            'forecast_trend': self._calculate_trend(forecast_data['yhat'].values)
        }

    def _generate_arima_forecast(self, data: pd.DataFrame, metric_name: str, forecast_days: int) -> Dict:
        """Generate forecast using ARIMA model."""
        model = self.train_arima_model(data, metric_name)

        # Generate forecast
        forecast_result = model.forecast(steps=forecast_days)
        confidence_intervals = model.get_forecast(steps=forecast_days).conf_int()

        # Create forecast dates
        last_date = data['ds'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')

        # Prepare forecast data
        predictions = []
        for i, date in enumerate(forecast_dates):
            predictions.append({
                'ds': date.isoformat(),
                'yhat': float(forecast_result.iloc[i]),
                'yhat_lower': float(confidence_intervals.iloc[i, 0]),
                'yhat_upper': float(confidence_intervals.iloc[i, 1])
            })

        return {
            'model_type': 'arima',
            'metric': metric_name,
            'forecast_days': forecast_days,
            'predictions': predictions,
            'confidence_intervals': True,
            'last_historical_value': float(data['y'].iloc[-1]),
            'forecast_trend': self._calculate_trend([p['yhat'] for p in predictions])
        }

    def _generate_fallback_forecast(self, data: pd.DataFrame, metric_name: str, forecast_days: int) -> Dict:
        """Generate simple fallback forecast using moving average."""
        logger.warning(f"Using fallback forecast for {metric_name}")

        # Simple moving average forecast
        window_size = min(7, len(data))
        recent_avg = data['y'].tail(window_size).mean()
        recent_std = data['y'].tail(window_size).std()

        # Create forecast dates
        last_date = data['ds'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')

        predictions = []
        for date in forecast_dates:
            predictions.append({
                'ds': date.isoformat(),
                'yhat': float(recent_avg),
                'yhat_lower': float(recent_avg - 1.96 * recent_std),
                'yhat_upper': float(recent_avg + 1.96 * recent_std)
            })

        return {
            'model_type': 'fallback_moving_average',
            'metric': metric_name,
            'forecast_days': forecast_days,
            'predictions': predictions,
            'confidence_intervals': True,
            'last_historical_value': float(data['y'].iloc[-1]),
            'forecast_trend': 'stable'
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from forecast values."""
        if len(values) < 2:
            return 'stable'

        start_val = values[0]
        end_val = values[-1]
        change_pct = (end_val - start_val) / start_val if start_val != 0 else 0

        if change_pct > 0.1:
            return 'increasing'
        elif change_pct < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def get_risk_assessment(self, metric_name: str) -> Dict[str, Any]:
        """
        Get risk assessment for a specific metric.

        Args:
            metric_name: Name of the metric to assess

        Returns:
            Risk assessment dictionary
        """
        if metric_name not in self.predictions:
            logger.warning(f"No predictions available for {metric_name}")
            return {'risk': 'unknown', 'confidence': 0.0, 'recommendation': 'insufficient_data'}

        forecast = self.predictions[metric_name]
        predictions = forecast['predictions']

        if not predictions:
            return {'risk': 'unknown', 'confidence': 0.0, 'recommendation': 'no_predictions'}

        # Calculate average predicted value for next 7 days
        next_week_values = [p['yhat'] for p in predictions[:7]]
        avg_prediction = np.mean(next_week_values)

        # Determine risk level
        if metric_name in ['delay_risk', 'escalation_likelihood']:
            if avg_prediction > 0.7:
                risk_level = 'high'
                recommendation = 'immediate_action_required'
            elif avg_prediction > 0.4:
                risk_level = 'medium'
                recommendation = 'monitor_closely'
            else:
                risk_level = 'low'
                recommendation = 'continue_monitoring'
        else:  # agent_load
            if avg_prediction > 30:
                risk_level = 'high'
                recommendation = 'scale_up_resources'
            elif avg_prediction > 20:
                risk_level = 'medium'
                recommendation = 'prepare_for_increased_load'
            else:
                risk_level = 'low'
                recommendation = 'current_capacity_sufficient'

        # Calculate confidence based on prediction interval width
        confidence_widths = [(p['yhat_upper'] - p['yhat_lower']) for p in predictions[:7]]
        avg_width = np.mean(confidence_widths)
        confidence = max(0.1, min(1.0, 1.0 - avg_width))  # Narrower intervals = higher confidence

        return {
            'risk': risk_level,
            'confidence': float(confidence),
            'recommendation': recommendation,
            'predicted_value': float(avg_prediction),
            'trend': forecast['forecast_trend'],
            'model_type': forecast['model_type']
        }

    def get_agent_score(self, agent_id: str, current_load: int = None) -> Dict[str, Any]:
        """
        Score an agent based on current load and predicted capacity.

        Args:
            agent_id: Identifier for the agent
            current_load: Current task load for the agent

        Returns:
            Agent scoring dictionary
        """
        # Get daily agent load prediction
        if 'daily_agent_load' not in self.predictions:
            self.generate_forecasts()

        load_forecast = self.predictions.get('daily_agent_load', {})
        predictions = load_forecast.get('predictions', [])

        if not predictions:
            return {
                'agent_id': agent_id,
                'score': 0.5,
                'capacity_status': 'unknown',
                'recommendation': 'insufficient_data'
            }

        # Get predicted load for tomorrow
        tomorrow_prediction = predictions[0]['yhat'] if predictions else 15
        current_load = current_load or 10  # Default current load

        # Calculate capacity score (0-1, higher is better)
        optimal_load = 20  # Optimal tasks per agent per day
        if current_load <= optimal_load * 0.7:
            capacity_score = 1.0  # Under-utilized
        elif current_load <= optimal_load:
            capacity_score = 0.8  # Well-utilized
        elif current_load <= optimal_load * 1.3:
            capacity_score = 0.5  # Over-utilized
        else:
            capacity_score = 0.2  # Severely over-utilized

        # Adjust score based on predicted load
        if tomorrow_prediction > optimal_load * 1.2:
            capacity_score *= 0.7  # Reduce score if high load predicted

        # Determine capacity status
        if capacity_score >= 0.8:
            status = 'available'
            recommendation = 'can_accept_new_tasks'
        elif capacity_score >= 0.5:
            status = 'moderate'
            recommendation = 'monitor_load'
        else:
            status = 'overloaded'
            recommendation = 'reassign_tasks'

        return {
            'agent_id': agent_id,
            'score': float(capacity_score),
            'capacity_status': status,
            'recommendation': recommendation,
            'current_load': current_load,
            'predicted_load': float(tomorrow_prediction),
            'optimal_load': optimal_load
        }

    def should_reassign_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if a task should be reassigned based on predictions.

        Args:
            task_data: Dictionary containing task information

        Returns:
            Reassignment recommendation
        """
        # Get risk assessments
        delay_risk = self.get_risk_assessment('delay_risk')
        escalation_risk = self.get_risk_assessment('escalation_likelihood')

        # Calculate overall risk score
        delay_weight = 0.6
        escalation_weight = 0.4

        overall_risk_score = (
            delay_risk['predicted_value'] * delay_weight +
            escalation_risk['predicted_value'] * escalation_weight
        )

        # Determine if reassignment is needed
        if overall_risk_score > 0.7:
            should_reassign = True
            urgency = 'immediate'
            reason = 'high_risk_detected'
        elif overall_risk_score > 0.5:
            should_reassign = True
            urgency = 'within_24h'
            reason = 'moderate_risk_detected'
        else:
            should_reassign = False
            urgency = 'none'
            reason = 'risk_within_acceptable_limits'

        return {
            'should_reassign': should_reassign,
            'urgency': urgency,
            'reason': reason,
            'overall_risk_score': float(overall_risk_score),
            'delay_risk': delay_risk,
            'escalation_risk': escalation_risk,
            'confidence': min(delay_risk['confidence'], escalation_risk['confidence'])
        }

    def get_daily_forecast_summary(self) -> Dict[str, Any]:
        """
        Get a summary of daily forecasts for dashboard consumption.

        Returns:
            Summary dictionary with key metrics
        """
        if not self.predictions:
            self.generate_forecasts()

        summary = {
            'forecast_date': datetime.now().isoformat(),
            'metrics': {},
            'overall_status': 'stable',
            'recommendations': []
        }

        risk_levels = []

        for metric_name in ['delay_risk', 'escalation_likelihood', 'daily_agent_load']:
            if metric_name in self.predictions:
                risk_assessment = self.get_risk_assessment(metric_name)
                summary['metrics'][metric_name] = risk_assessment

                if risk_assessment['risk'] == 'high':
                    risk_levels.append(3)
                elif risk_assessment['risk'] == 'medium':
                    risk_levels.append(2)
                else:
                    risk_levels.append(1)

                # Add recommendations
                if risk_assessment['recommendation'] not in ['continue_monitoring', 'current_capacity_sufficient']:
                    summary['recommendations'].append({
                        'metric': metric_name,
                        'action': risk_assessment['recommendation'],
                        'priority': risk_assessment['risk']
                    })

        # Determine overall status
        if risk_levels:
            max_risk = max(risk_levels)
            if max_risk >= 3:
                summary['overall_status'] = 'critical'
            elif max_risk >= 2:
                summary['overall_status'] = 'warning'
            else:
                summary['overall_status'] = 'stable'

        return summary


# Utility functions for integration
def create_prediction_agent(use_prophet: bool = True) -> PredictionAgent:
    """Create and initialize a prediction agent."""
    return PredictionAgent(use_prophet=use_prophet)


def run_forecast_simulation(agent: PredictionAgent, simulation_days: int = 30) -> Dict[str, Any]:
    """
    Run a complete forecast simulation.

    Args:
        agent: Initialized PredictionAgent
        simulation_days: Number of days to simulate

    Returns:
        Complete simulation results
    """
    logger.info(f"Starting forecast simulation for {simulation_days} days")

    # Load historical data
    agent.load_task_history_data()

    # Generate forecasts
    forecasts = agent.generate_forecasts(forecast_days=simulation_days)

    # Get daily summary
    daily_summary = agent.get_daily_forecast_summary()

    # Simulate agent scoring
    agent_scores = []
    for i in range(5):  # Score 5 mock agents
        agent_id = f"agent_{i+1}"
        current_load = np.random.randint(5, 25)  # Random current load
        score = agent.get_agent_score(agent_id, current_load)
        agent_scores.append(score)

    # Simulate task reassignment decisions
    mock_task = {
        'task_id': 'task_001',
        'assigned_agent': 'agent_1',
        'priority': 'medium',
        'complexity': 'standard'
    }
    reassignment_decision = agent.should_reassign_task(mock_task)

    return {
        'simulation_date': datetime.now().isoformat(),
        'simulation_days': simulation_days,
        'forecasts': forecasts,
        'daily_summary': daily_summary,
        'agent_scores': agent_scores,
        'reassignment_example': reassignment_decision,
        'model_type': 'prophet' if agent.use_prophet else 'arima'
    }
