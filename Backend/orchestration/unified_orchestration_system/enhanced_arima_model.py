"""
Enhanced ARIMA Model for Advanced Time Series Forecasting
Implements ARIMA with automatic parameter selection and stationarity testing
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime, timedelta
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedARIMAModel:
    """
    Enhanced ARIMA model with automatic parameter selection and robust error handling
    """
    
    def __init__(self, metric_type: str = "general"):
        """
        Initialize Enhanced ARIMA Model
        
        Args:
            metric_type: Type of metric ('probability', 'load', 'general')
        """
        self.metric_type = metric_type
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.performance_metrics = {}
        self.is_fitted = False
        self.original_data = None
        self.differenced_data = None
        
        # Parameter search ranges based on metric type
        self.param_ranges = self._get_param_ranges_for_metric_type(metric_type)
        
    def _get_param_ranges_for_metric_type(self, metric_type: str) -> Dict[str, List]:
        """
        Get ARIMA parameter search ranges based on metric type
        
        Args:
            metric_type: Type of metric
            
        Returns:
            Parameter ranges dictionary
        """
        ranges = {
            'probability': {
                'p_range': [0, 1, 2, 3],
                'd_range': [0, 1],
                'q_range': [0, 1, 2, 3],
                'max_params': 6,
                'seasonal': False
            },
            'load': {
                'p_range': [0, 1, 2, 3, 4],
                'd_range': [0, 1, 2],
                'q_range': [0, 1, 2, 3, 4],
                'max_params': 8,
                'seasonal': True
            },
            'general': {
                'p_range': [0, 1, 2, 3],
                'd_range': [0, 1, 2],
                'q_range': [0, 1, 2, 3],
                'max_params': 6,
                'seasonal': False
            }
        }
        
        return ranges.get(metric_type, ranges['general'])
    
    def test_stationarity(self, data: pd.Series, significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Test stationarity using Augmented Dickey-Fuller and KPSS tests
        
        Args:
            data: Time series data
            significance_level: Significance level for tests
            
        Returns:
            Stationarity test results
        """
        results = {}
        
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(data.dropna())
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < significance_level
            }
            
            # KPSS test
            kpss_result = kpss(data.dropna(), regression='c')
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > significance_level
            }
            
            # Overall stationarity assessment
            results['is_stationary'] = (
                results['adf']['is_stationary'] and 
                results['kpss']['is_stationary']
            )
            
            logger.info(f"Stationarity test - ADF p-value: {results['adf']['p_value']:.4f}, "
                       f"KPSS p-value: {results['kpss']['p_value']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in stationarity testing: {e}")
            results = {
                'adf': {'is_stationary': False},
                'kpss': {'is_stationary': False},
                'is_stationary': False
            }
        
        return results
    
    def determine_differencing(self, data: pd.Series, max_d: int = 2) -> Tuple[pd.Series, int]:
        """
        Determine optimal differencing order
        
        Args:
            data: Time series data
            max_d: Maximum differencing order to test
            
        Returns:
            Tuple of (differenced_data, optimal_d)
        """
        current_data = data.copy()
        d = 0
        
        for i in range(max_d + 1):
            stationarity_result = self.test_stationarity(current_data)
            
            if stationarity_result['is_stationary']:
                logger.info(f"Data is stationary with d={d}")
                return current_data, d
            
            if i < max_d:
                current_data = current_data.diff().dropna()
                d += 1
        
        logger.warning(f"Data may not be stationary even with d={d}")
        return current_data, d
    
    def grid_search_parameters(self, data: pd.Series) -> Tuple[int, int, int]:
        """
        Grid search for optimal ARIMA parameters
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of optimal (p, d, q) parameters
        """
        best_aic = float('inf')
        best_params = (1, 1, 1)  # Default fallback
        
        # Determine differencing order
        _, optimal_d = self.determine_differencing(data)
        
        # Grid search over p and q
        p_range = self.param_ranges['p_range']
        q_range = self.param_ranges['q_range']
        
        logger.info(f"Grid searching ARIMA parameters with d={optimal_d}")
        
        for p, q in itertools.product(p_range, q_range):
            # Skip if total parameters exceed limit
            if p + optimal_d + q > self.param_ranges['max_params']:
                continue
                
            try:
                model = ARIMA(data, order=(p, optimal_d, q))
                fitted_model = model.fit()
                
                aic = fitted_model.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, optimal_d, q)
                    
            except Exception as e:
                logger.debug(f"Failed to fit ARIMA({p},{optimal_d},{q}): {e}")
                continue
        
        logger.info(f"Best ARIMA parameters: {best_params} with AIC: {best_aic:.2f}")
        return best_params
    
    def fit(self, data: pd.DataFrame, date_col: str = 'ds', value_col: str = 'y') -> 'EnhancedARIMAModel':
        """
        Fit the ARIMA model to data
        
        Args:
            data: Training data
            date_col: Date column name
            value_col: Value column name
            
        Returns:
            Self for method chaining
        """
        try:
            # Prepare data
            df = data.copy()
            if date_col != 'ds':
                df = df.rename(columns={date_col: 'ds'})
            if value_col != 'y':
                df = df.rename(columns={value_col: 'y'})
            
            # Ensure datetime format and sort
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.sort_values('ds').reset_index(drop=True)
            
            # Handle missing values
            df['y'] = df['y'].fillna(df['y'].median())
            
            # Validate data
            if len(df) < 10:
                raise ValueError(f"Insufficient data points: {len(df)}. Need at least 10 points.")
            
            # Store original data
            self.original_data = df['y'].copy()
            
            # Find optimal parameters
            self.best_params = self.grid_search_parameters(self.original_data)
            
            # Fit the model with best parameters
            logger.info(f"Fitting ARIMA{self.best_params} model...")
            self.model = ARIMA(self.original_data, order=self.best_params)
            self.fitted_model = self.model.fit()
            
            self.is_fitted = True
            logger.info("ARIMA model fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            # Try fallback parameters
            try:
                logger.info("Attempting fallback ARIMA(1,1,1) model...")
                self.best_params = (1, 1, 1)
                self.model = ARIMA(self.original_data, order=self.best_params)
                self.fitted_model = self.model.fit()
                self.is_fitted = True
                logger.info("Fallback ARIMA model fitted successfully")
                return self
            except Exception as e2:
                logger.error(f"Fallback ARIMA model also failed: {e2}")
                raise
    
    def predict(self, periods: int = 30) -> pd.DataFrame:
        """
        Generate predictions
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Forecast dataframe
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Generate forecast
            forecast_result = self.fitted_model.forecast(steps=periods)
            confidence_intervals = self.fitted_model.get_forecast(steps=periods).conf_int()
            
            # Create forecast dataframe
            last_date = pd.to_datetime(self.original_data.index[-1]) if hasattr(self.original_data.index, 'dtype') else datetime.now()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_result,
                'yhat_lower': confidence_intervals.iloc[:, 0],
                'yhat_upper': confidence_intervals.iloc[:, 1]
            })
            
            logger.info(f"Generated ARIMA forecast for {periods} periods")
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating ARIMA predictions: {e}")
            raise

    def calculate_performance_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics for model evaluation

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Performance metrics dictionary
        """
        try:
            # Align series
            min_length = min(len(actual), len(predicted))
            actual_aligned = actual.iloc[-min_length:].values
            predicted_aligned = predicted.iloc[-min_length:].values

            # Calculate metrics
            mae = np.mean(np.abs(actual_aligned - predicted_aligned))
            mse = np.mean((actual_aligned - predicted_aligned) ** 2)
            rmse = np.sqrt(mse)

            # MAPE (handle division by zero)
            mape = np.mean(np.abs((actual_aligned - predicted_aligned) / np.where(actual_aligned != 0, actual_aligned, 1))) * 100

            self.performance_metrics = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'aic': float(self.fitted_model.aic) if self.fitted_model else None,
                'bic': float(self.fitted_model.bic) if self.fitted_model else None
            }

            logger.info(f"Performance metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            return self.performance_metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf')}

    def diagnose_residuals(self) -> Dict[str, Any]:
        """
        Perform residual diagnostics

        Returns:
            Diagnostic results dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before residual diagnostics")

        diagnostics = {}

        try:
            residuals = self.fitted_model.resid

            # Ljung-Box test for autocorrelation
            ljung_box_result = acorr_ljungbox(residuals, lags=10, return_df=True)
            diagnostics['ljung_box'] = {
                'statistic': float(ljung_box_result['lb_stat'].iloc[-1]),
                'p_value': float(ljung_box_result['lb_pvalue'].iloc[-1]),
                'no_autocorrelation': float(ljung_box_result['lb_pvalue'].iloc[-1]) > 0.05
            }

            # Residual statistics
            diagnostics['residual_stats'] = {
                'mean': float(residuals.mean()),
                'std': float(residuals.std()),
                'skewness': float(residuals.skew()) if hasattr(residuals, 'skew') else None,
                'kurtosis': float(residuals.kurtosis()) if hasattr(residuals, 'kurtosis') else None
            }

            logger.info("Residual diagnostics completed")

        except Exception as e:
            logger.error(f"Error in residual diagnostics: {e}")
            diagnostics = {'error': str(e)}

        return diagnostics

    def get_forecast_summary(self, periods: int = 30) -> Dict[str, Any]:
        """
        Get forecast summary with key metrics

        Args:
            periods: Number of periods to forecast

        Returns:
            Forecast summary dictionary
        """
        try:
            forecast_df = self.predict(periods)

            summary = {
                'metric_type': self.metric_type,
                'model_params': self.best_params,
                'forecast_periods': periods,
                'forecast_start': forecast_df['ds'].min().isoformat(),
                'forecast_end': forecast_df['ds'].max().isoformat(),
                'predictions': {
                    'mean': float(forecast_df['yhat'].mean()),
                    'min': float(forecast_df['yhat'].min()),
                    'max': float(forecast_df['yhat'].max()),
                    'trend': 'increasing' if forecast_df['yhat'].iloc[-1] > forecast_df['yhat'].iloc[0] else 'decreasing'
                },
                'confidence_intervals': {
                    'lower_bound_mean': float(forecast_df['yhat_lower'].mean()),
                    'upper_bound_mean': float(forecast_df['yhat_upper'].mean())
                },
                'model_diagnostics': self.diagnose_residuals(),
                'performance_metrics': self.performance_metrics
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating forecast summary: {e}")
            return {'error': str(e)}

    def export_model_config(self) -> Dict[str, Any]:
        """
        Export model configuration for reproducibility

        Returns:
            Model configuration dictionary
        """
        config_export = {
            'metric_type': self.metric_type,
            'best_params': self.best_params,
            'param_ranges': self.param_ranges,
            'is_fitted': self.is_fitted,
            'performance_metrics': self.performance_metrics,
            'model_version': '1.0',
            'created_at': datetime.now().isoformat()
        }

        return config_export

    def get_fallback_configurations(self) -> List[Tuple[int, int, int]]:
        """
        Get fallback ARIMA configurations for robust error handling

        Returns:
            List of fallback parameter tuples
        """
        fallback_configs = [
            (1, 1, 1),  # Simple ARIMA
            (0, 1, 1),  # Moving average with differencing
            (1, 1, 0),  # Autoregressive with differencing
            (2, 1, 2),  # More complex but stable
            (0, 1, 0),  # Random walk
            (1, 0, 1),  # ARMA without differencing
        ]

        return fallback_configs
