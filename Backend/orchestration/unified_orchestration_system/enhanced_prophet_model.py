"""
Enhanced Prophet Model for Advanced Time Series Forecasting
Implements Prophet with logistic growth, seasonality detection, and capacity constraints
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime, timedelta
import json

# Suppress Prophet warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('prophet').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class EnhancedProphetModel:
    """
    Enhanced Prophet model with advanced configuration for different metric types
    """
    
    def __init__(self, metric_type: str = "general"):
        """
        Initialize Enhanced Prophet Model
        
        Args:
            metric_type: Type of metric ('probability', 'load', 'general')
        """
        self.metric_type = metric_type
        self.model = None
        self.forecast = None
        self.performance_metrics = {}
        self.is_fitted = False
        
        # Configuration based on metric type
        self.config = self._get_config_for_metric_type(metric_type)
        
    def _get_config_for_metric_type(self, metric_type: str) -> Dict[str, Any]:
        """
        Get Prophet configuration based on metric type
        
        Args:
            metric_type: Type of metric
            
        Returns:
            Configuration dictionary
        """
        configs = {
            'probability': {
                'growth': 'logistic',
                'cap': 1.0,
                'floor': 0.0,
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'mcmc_samples': 0,
                'interval_width': 0.80,
                'uncertainty_samples': 1000
            },
            'load': {
                'growth': 'linear',
                'seasonality_mode': 'additive',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': True,
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'mcmc_samples': 0,
                'interval_width': 0.80,
                'uncertainty_samples': 1000
            },
            'general': {
                'growth': 'linear',
                'seasonality_mode': 'additive',
                'yearly_seasonality': 'auto',
                'weekly_seasonality': 'auto',
                'daily_seasonality': 'auto',
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'mcmc_samples': 0,
                'interval_width': 0.80,
                'uncertainty_samples': 1000
            }
        }
        
        return configs.get(metric_type, configs['general'])
    
    def prepare_data(self, data: pd.DataFrame, date_col: str = 'ds', value_col: str = 'y') -> pd.DataFrame:
        """
        Prepare data for Prophet model
        
        Args:
            data: Input dataframe
            date_col: Date column name
            value_col: Value column name
            
        Returns:
            Prepared dataframe
        """
        # Ensure proper column names
        df = data.copy()
        if date_col != 'ds':
            df = df.rename(columns={date_col: 'ds'})
        if value_col != 'y':
            df = df.rename(columns={value_col: 'y'})
        
        # Ensure datetime format
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Sort by date
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Handle missing values
        df['y'] = df['y'].fillna(df['y'].median())
        
        # Add capacity constraints for logistic growth
        if self.config['growth'] == 'logistic':
            if 'cap' not in df.columns:
                df['cap'] = self.config['cap']
            if 'floor' not in df.columns and 'floor' in self.config:
                df['floor'] = self.config['floor']
        
        logger.info(f"Prepared data: {len(df)} records from {df['ds'].min()} to {df['ds'].max()}")
        return df
    
    def add_custom_seasonalities(self, model: Prophet) -> Prophet:
        """
        Add custom seasonalities based on data characteristics
        
        Args:
            model: Prophet model instance
            
        Returns:
            Model with added seasonalities
        """
        try:
            # Add monthly seasonality
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5,
                prior_scale=10.0
            )
            
            # Add quarterly seasonality for business metrics
            if self.metric_type in ['load', 'general']:
                model.add_seasonality(
                    name='quarterly',
                    period=91.25,
                    fourier_order=3,
                    prior_scale=5.0
                )
            
            logger.info("Added custom seasonalities: monthly, quarterly")
            
        except Exception as e:
            logger.warning(f"Could not add custom seasonalities: {e}")
        
        return model
    
    def fit(self, data: pd.DataFrame, date_col: str = 'ds', value_col: str = 'y') -> 'EnhancedProphetModel':
        """
        Fit the Prophet model to data
        
        Args:
            data: Training data
            date_col: Date column name
            value_col: Value column name
            
        Returns:
            Self for method chaining
        """
        try:
            # Prepare data
            df = self.prepare_data(data, date_col, value_col)
            
            # Validate data
            if len(df) < 10:
                raise ValueError(f"Insufficient data points: {len(df)}. Need at least 10 points.")
            
            # Initialize Prophet model
            self.model = Prophet(
                growth=self.config['growth'],
                seasonality_mode=self.config['seasonality_mode'],
                yearly_seasonality=self.config['yearly_seasonality'],
                weekly_seasonality=self.config['weekly_seasonality'],
                daily_seasonality=self.config['daily_seasonality'],
                changepoint_prior_scale=self.config['changepoint_prior_scale'],
                seasonality_prior_scale=self.config['seasonality_prior_scale'],
                holidays_prior_scale=self.config['holidays_prior_scale'],
                mcmc_samples=self.config['mcmc_samples'],
                interval_width=self.config['interval_width'],
                uncertainty_samples=self.config['uncertainty_samples']
            )
            
            # Add custom seasonalities
            self.model = self.add_custom_seasonalities(self.model)
            
            # Fit the model
            logger.info(f"Fitting Prophet model for {self.metric_type} metric...")
            self.model.fit(df)
            
            self.is_fitted = True
            logger.info("Prophet model fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            raise
    
    def predict(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """
        Generate predictions
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of predictions ('D' for daily, 'H' for hourly)
            
        Returns:
            Forecast dataframe
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            
            # Add capacity constraints for logistic growth
            if self.config['growth'] == 'logistic':
                future['cap'] = self.config['cap']
                if 'floor' in self.config:
                    future['floor'] = self.config['floor']
            
            # Generate forecast
            self.forecast = self.model.predict(future)
            
            logger.info(f"Generated forecast for {periods} periods")
            return self.forecast
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise
    
    def get_forecast_summary(self, periods: int = 30) -> Dict[str, Any]:
        """
        Get forecast summary with key metrics
        
        Args:
            periods: Number of periods to include in summary
            
        Returns:
            Forecast summary dictionary
        """
        if self.forecast is None:
            self.predict(periods)
        
        # Get future predictions only
        future_forecast = self.forecast.tail(periods)
        
        summary = {
            'metric_type': self.metric_type,
            'forecast_periods': periods,
            'forecast_start': future_forecast['ds'].min().isoformat(),
            'forecast_end': future_forecast['ds'].max().isoformat(),
            'predictions': {
                'mean': float(future_forecast['yhat'].mean()),
                'min': float(future_forecast['yhat'].min()),
                'max': float(future_forecast['yhat'].max()),
                'trend': 'increasing' if future_forecast['yhat'].iloc[-1] > future_forecast['yhat'].iloc[0] else 'decreasing'
            },
            'confidence_intervals': {
                'lower_bound_mean': float(future_forecast['yhat_lower'].mean()),
                'upper_bound_mean': float(future_forecast['yhat_upper'].mean())
            },
            'components': {
                'trend_contribution': float(future_forecast['trend'].mean()) if 'trend' in future_forecast.columns else None,
                'seasonal_contribution': float(future_forecast['seasonal'].mean()) if 'seasonal' in future_forecast.columns else None
            }
        }
        
        return summary

    def cross_validate_model(self, data: pd.DataFrame, initial: str = '30 days',
                           period: str = '7 days', horizon: str = '14 days') -> Dict[str, float]:
        """
        Perform cross-validation to evaluate model performance

        Args:
            data: Training data
            initial: Initial training period
            period: Period between cutoffs
            horizon: Forecast horizon

        Returns:
            Performance metrics dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")

        try:
            logger.info("Performing cross-validation...")

            # Perform cross-validation
            df_cv = cross_validation(
                self.model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="processes"
            )

            # Calculate performance metrics
            df_p = performance_metrics(df_cv)

            # Extract key metrics
            self.performance_metrics = {
                'mae': float(df_p['mae'].mean()),
                'mape': float(df_p['mape'].mean()),
                'rmse': float(df_p['rmse'].mean()),
                'coverage': float(df_p['coverage'].mean()) if 'coverage' in df_p.columns else None
            }

            logger.info(f"Cross-validation completed. MAE: {self.performance_metrics['mae']:.4f}")
            return self.performance_metrics

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            # Return default metrics if cross-validation fails
            return {'mae': float('inf'), 'mape': float('inf'), 'rmse': float('inf'), 'coverage': None}

    def detect_anomalies(self, threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the forecast

        Args:
            threshold: Confidence threshold for anomaly detection

        Returns:
            List of anomaly dictionaries
        """
        if self.forecast is None:
            raise ValueError("Must generate forecast before detecting anomalies")

        anomalies = []

        try:
            # Calculate prediction intervals
            forecast_df = self.forecast.copy()

            # Identify points outside confidence intervals
            forecast_df['anomaly_score'] = np.where(
                (forecast_df['yhat'] < forecast_df['yhat_lower']) |
                (forecast_df['yhat'] > forecast_df['yhat_upper']),
                1, 0
            )

            # Find anomalous periods
            anomaly_points = forecast_df[forecast_df['anomaly_score'] == 1]

            for _, row in anomaly_points.iterrows():
                anomalies.append({
                    'date': row['ds'].isoformat(),
                    'predicted_value': float(row['yhat']),
                    'lower_bound': float(row['yhat_lower']),
                    'upper_bound': float(row['yhat_upper']),
                    'severity': 'high' if abs(row['yhat'] - row['yhat_lower']) > abs(row['yhat_upper'] - row['yhat_lower']) * 0.5 else 'medium'
                })

            logger.info(f"Detected {len(anomalies)} potential anomalies")

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")

        return anomalies

    def get_model_components(self) -> Dict[str, pd.DataFrame]:
        """
        Get model components (trend, seasonality, etc.)

        Returns:
            Dictionary of component dataframes
        """
        if self.forecast is None:
            raise ValueError("Must generate forecast before extracting components")

        components = {}

        try:
            # Get Prophet components
            prophet_components = self.model.predict_components(self.forecast[['ds']])

            components['trend'] = prophet_components[['ds', 'trend']]

            # Add seasonality components
            for col in prophet_components.columns:
                if col not in ['ds', 'trend'] and not col.startswith('extra_regressors'):
                    components[col] = prophet_components[['ds', col]]

            logger.info(f"Extracted {len(components)} model components")

        except Exception as e:
            logger.error(f"Error extracting model components: {e}")

        return components

    def export_model_config(self) -> Dict[str, Any]:
        """
        Export model configuration for reproducibility

        Returns:
            Model configuration dictionary
        """
        config_export = {
            'metric_type': self.metric_type,
            'prophet_config': self.config,
            'is_fitted': self.is_fitted,
            'performance_metrics': self.performance_metrics,
            'model_version': '1.0',
            'created_at': datetime.now().isoformat()
        }

        return config_export
