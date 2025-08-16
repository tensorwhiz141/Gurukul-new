"""
Smart Model Selection Logic
Implements intelligent model selection with Prophet primary, ARIMA fallback, and performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import json
import warnings

# Import our custom models
from enhanced_prophet_model import EnhancedProphetModel
from enhanced_arima_model import EnhancedARIMAModel
from model_performance_evaluator import ModelPerformanceEvaluator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SmartModelSelector:
    """
    Intelligent model selection system with automatic fallback and performance tracking
    """
    
    def __init__(self, metric_type: str = "general"):
        """
        Initialize Smart Model Selector
        
        Args:
            metric_type: Type of metric ('probability', 'load', 'general')
        """
        self.metric_type = metric_type
        self.performance_evaluator = ModelPerformanceEvaluator()
        self.model_history = {}
        self.selected_model = None
        self.selection_reason = ""
        self.performance_threshold = {
            'mae': 0.2,
            'rmse': 0.3,
            'mape': 20.0
        }
        
    def select_best_model(self, data: pd.DataFrame, date_col: str = 'ds', 
                         value_col: str = 'y', force_evaluation: bool = False) -> Dict[str, Any]:
        """
        Select the best model based on data characteristics and performance
        
        Args:
            data: Input time series data
            date_col: Date column name
            value_col: Value column name
            force_evaluation: Force full evaluation even with sufficient data
            
        Returns:
            Model selection results dictionary
        """
        try:
            logger.info(f"Starting smart model selection for {self.metric_type} metric...")
            
            # Prepare data
            df = self._prepare_data(data, date_col, value_col)
            
            # Check data requirements
            data_assessment = self._assess_data_quality(df)
            
            # Determine selection strategy
            if len(df) < 10:
                return self._handle_insufficient_data(df, data_assessment)
            elif len(df) < 20 and not force_evaluation:
                return self._quick_selection(df, data_assessment)
            else:
                return self._full_evaluation_selection(df, data_assessment)
                
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            return self._fallback_selection(str(e))
    
    def _prepare_data(self, data: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """Prepare and validate data for model selection"""
        df = data.copy()
        
        # Ensure proper column names
        if date_col != 'ds':
            df = df.rename(columns={date_col: 'ds'})
        if value_col != 'y':
            df = df.rename(columns={value_col: 'y'})
        
        # Ensure datetime format and sort
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Handle missing values
        df['y'] = df['y'].fillna(df['y'].median())
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ds']).reset_index(drop=True)
        
        logger.info(f"Prepared data: {len(df)} records from {df['ds'].min()} to {df['ds'].max()}")
        return df
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and characteristics"""
        assessment = {
            'data_points': len(data),
            'date_range_days': (data['ds'].max() - data['ds'].min()).days,
            'missing_values': data['y'].isna().sum(),
            'zero_values': (data['y'] == 0).sum(),
            'negative_values': (data['y'] < 0).sum(),
            'variance': float(data['y'].var()),
            'mean': float(data['y'].mean()),
            'std': float(data['y'].std()),
            'trend_direction': 'increasing' if data['y'].iloc[-1] > data['y'].iloc[0] else 'decreasing',
            'seasonality_detected': self._detect_seasonality(data['y'])
        }
        
        # Data quality score (0-1)
        quality_score = 1.0
        if assessment['missing_values'] > len(data) * 0.1:
            quality_score -= 0.2
        if assessment['variance'] == 0:
            quality_score -= 0.3
        if assessment['data_points'] < 30:
            quality_score -= 0.2
        
        assessment['quality_score'] = max(0.0, quality_score)
        
        logger.info(f"Data assessment - Points: {assessment['data_points']}, "
                   f"Quality: {assessment['quality_score']:.2f}, "
                   f"Seasonality: {assessment['seasonality_detected']}")
        
        return assessment
    
    def _detect_seasonality(self, series: pd.Series) -> bool:
        """Simple seasonality detection"""
        try:
            if len(series) < 14:  # Need at least 2 weeks for weekly seasonality
                return False
            
            # Check for weekly pattern (7-day cycle)
            autocorr_7 = series.autocorr(lag=7) if len(series) > 7 else 0
            
            # Check for monthly pattern (30-day cycle)
            autocorr_30 = series.autocorr(lag=30) if len(series) > 30 else 0
            
            # Consider seasonality if autocorrelation > 0.3
            return abs(autocorr_7) > 0.3 or abs(autocorr_30) > 0.3
            
        except Exception:
            return False
    
    def _handle_insufficient_data(self, data: pd.DataFrame, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cases with insufficient data"""
        logger.warning(f"Insufficient data ({len(data)} points) for advanced modeling")
        
        # Use simple linear regression or moving average
        self.selected_model = "simple_forecast"
        self.selection_reason = f"Insufficient data ({len(data)} points) for Prophet/ARIMA models"
        
        return {
            'selected_model': self.selected_model,
            'selection_reason': self.selection_reason,
            'data_assessment': assessment,
            'forecast_method': 'linear_trend',
            'confidence': 'low',
            'timestamp': datetime.now().isoformat()
        }
    
    def _quick_selection(self, data: pd.DataFrame, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Quick selection for moderate data sizes"""
        logger.info("Performing quick model selection...")
        
        # Default to Prophet for moderate data sizes
        try:
            prophet_model = EnhancedProphetModel(self.metric_type)
            prophet_model.fit(data)
            
            self.selected_model = prophet_model
            self.selection_reason = f"Quick selection: Prophet chosen for {len(data)} data points"
            
            return {
                'selected_model': 'prophet',
                'model_object': prophet_model,
                'selection_reason': self.selection_reason,
                'data_assessment': assessment,
                'confidence': 'medium',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Prophet quick selection failed: {e}, falling back to ARIMA")
            
            try:
                arima_model = EnhancedARIMAModel(self.metric_type)
                arima_model.fit(data)
                
                self.selected_model = arima_model
                self.selection_reason = f"Quick selection: ARIMA fallback after Prophet failure"
                
                return {
                    'selected_model': 'arima',
                    'model_object': arima_model,
                    'selection_reason': self.selection_reason,
                    'data_assessment': assessment,
                    'confidence': 'medium',
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e2:
                logger.error(f"Both models failed in quick selection: {e2}")
                return self._fallback_selection(f"Quick selection failed: {e2}")
    
    def _full_evaluation_selection(self, data: pd.DataFrame, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Full evaluation and comparison of models"""
        logger.info("Performing full model evaluation and selection...")
        
        # Split data for evaluation
        train_data, test_data = self.performance_evaluator.train_test_split(data)
        
        models_to_evaluate = {}
        evaluation_results = {}
        
        # Try Prophet model
        try:
            logger.info("Training Prophet model...")
            prophet_model = EnhancedProphetModel(self.metric_type)
            prophet_model.fit(train_data)
            models_to_evaluate['prophet'] = prophet_model
            
            # Evaluate Prophet
            prophet_eval = self.performance_evaluator.evaluate_model(
                prophet_model, train_data, test_data, 'prophet'
            )
            evaluation_results['prophet'] = prophet_eval
            
        except Exception as e:
            logger.warning(f"Prophet model training/evaluation failed: {e}")
        
        # Try ARIMA model
        try:
            logger.info("Training ARIMA model...")
            arima_model = EnhancedARIMAModel(self.metric_type)
            arima_model.fit(train_data)
            models_to_evaluate['arima'] = arima_model
            
            # Evaluate ARIMA
            arima_eval = self.performance_evaluator.evaluate_model(
                arima_model, train_data, test_data, 'arima'
            )
            evaluation_results['arima'] = arima_eval
            
        except Exception as e:
            logger.warning(f"ARIMA model training/evaluation failed: {e}")
        
        # Compare models and select best
        if evaluation_results:
            comparison_results = self.performance_evaluator.compare_models(evaluation_results)
            best_model_name = comparison_results.get('overall_best_model', 'prophet')
            
            if best_model_name in models_to_evaluate:
                self.selected_model = models_to_evaluate[best_model_name]
                self.selection_reason = f"Full evaluation: {best_model_name} selected based on performance comparison"
                
                return {
                    'selected_model': best_model_name,
                    'model_object': self.selected_model,
                    'selection_reason': self.selection_reason,
                    'data_assessment': assessment,
                    'evaluation_results': evaluation_results,
                    'comparison_results': comparison_results,
                    'confidence': 'high',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._fallback_selection("Best model not available in trained models")
        else:
            return self._fallback_selection("No models could be successfully trained and evaluated")
    
    def _fallback_selection(self, error_message: str) -> Dict[str, Any]:
        """Fallback selection when all else fails"""
        logger.error(f"Using fallback selection: {error_message}")
        
        self.selected_model = "simple_forecast"
        self.selection_reason = f"Fallback: {error_message}"
        
        return {
            'selected_model': 'simple_forecast',
            'selection_reason': self.selection_reason,
            'error': error_message,
            'confidence': 'very_low',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_model_recommendations(self, data_assessment: Dict[str, Any]) -> List[str]:
        """Get recommendations for improving model performance"""
        recommendations = []
        
        if data_assessment['data_points'] < 30:
            recommendations.append("Collect more data points for better model performance")
        
        if data_assessment['quality_score'] < 0.7:
            recommendations.append("Improve data quality by handling missing values and outliers")
        
        if data_assessment['variance'] == 0:
            recommendations.append("Data shows no variation - check data collection process")
        
        if not data_assessment['seasonality_detected'] and self.metric_type == 'load':
            recommendations.append("Consider external factors that might introduce seasonality")
        
        return recommendations
    
    def export_selection_history(self) -> Dict[str, Any]:
        """Export model selection history for analysis"""
        return {
            'metric_type': self.metric_type,
            'model_history': self.model_history,
            'current_selection': {
                'model': str(type(self.selected_model).__name__) if self.selected_model else None,
                'reason': self.selection_reason
            },
            'performance_thresholds': self.performance_threshold,
            'export_timestamp': datetime.now().isoformat()
        }
