"""
Model Performance Evaluation System
Implements 80/20 train-test split, accuracy metrics, and performance comparison framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelPerformanceEvaluator:
    """
    Comprehensive model performance evaluation system for time series forecasting
    """
    
    def __init__(self):
        """Initialize the performance evaluator"""
        self.evaluation_results = {}
        self.comparison_results = {}
        
    def train_test_split(self, data: pd.DataFrame, test_size: float = 0.2, 
                        date_col: str = 'ds', value_col: str = 'y') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets using temporal split
        
        Args:
            data: Input dataframe
            test_size: Proportion of data for testing (default: 0.2)
            date_col: Date column name
            value_col: Value column name
            
        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            # Ensure proper column names
            df = data.copy()
            if date_col != 'ds':
                df = df.rename(columns={date_col: 'ds'})
            if value_col != 'y':
                df = df.rename(columns={value_col: 'y'})
            
            # Ensure datetime format and sort
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.sort_values('ds').reset_index(drop=True)
            
            # Calculate split point
            split_point = int(len(df) * (1 - test_size))
            
            train_data = df.iloc[:split_point].copy()
            test_data = df.iloc[split_point:].copy()
            
            logger.info(f"Data split - Train: {len(train_data)} records, Test: {len(test_data)} records")
            logger.info(f"Train period: {train_data['ds'].min()} to {train_data['ds'].max()}")
            logger.info(f"Test period: {test_data['ds'].min()} to {test_data['ds'].max()}")
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error in train-test split: {e}")
            raise
    
    def calculate_accuracy_metrics(self, actual: Union[pd.Series, np.ndarray], 
                                 predicted: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Calculate comprehensive accuracy metrics
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of accuracy metrics
        """
        try:
            # Convert to numpy arrays for consistent handling
            if isinstance(actual, pd.Series):
                actual = actual.values
            if isinstance(predicted, pd.Series):
                predicted = predicted.values
            
            # Ensure same length
            min_length = min(len(actual), len(predicted))
            actual = actual[-min_length:]
            predicted = predicted[-min_length:]
            
            # Calculate metrics
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            
            # MAPE (Mean Absolute Percentage Error)
            # Handle division by zero
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
            else:
                mape = float('inf')
            
            # Additional metrics
            # Mean Error (bias)
            me = np.mean(predicted - actual)
            
            # Mean Absolute Scaled Error (MASE) - using naive forecast as baseline
            if len(actual) > 1:
                naive_forecast = actual[:-1]
                naive_mae = np.mean(np.abs(actual[1:] - naive_forecast))
                mase = mae / naive_mae if naive_mae != 0 else float('inf')
            else:
                mase = float('inf')
            
            # R-squared
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Symmetric Mean Absolute Percentage Error (SMAPE)
            smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100
            
            metrics = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'me': float(me),
                'mase': float(mase),
                'r2': float(r2),
                'smape': float(smape)
            }
            
            logger.info(f"Calculated metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf'),
                'me': float('inf'),
                'mase': float('inf'),
                'r2': 0.0,
                'smape': float('inf')
            }
    
    def evaluate_model(self, model, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a single model's performance
        
        Args:
            model: Fitted model object (Prophet or ARIMA)
            train_data: Training data
            test_data: Testing data
            model_name: Name of the model for identification
            
        Returns:
            Evaluation results dictionary
        """
        try:
            logger.info(f"Evaluating {model_name} model performance...")
            
            # Generate predictions for test period
            test_periods = len(test_data)
            predictions = model.predict(periods=test_periods)
            
            # Extract predicted values
            if 'yhat' in predictions.columns:
                predicted_values = predictions['yhat'].values
            else:
                predicted_values = predictions.values
            
            actual_values = test_data['y'].values
            
            # Calculate accuracy metrics
            metrics = self.calculate_accuracy_metrics(actual_values, predicted_values)
            
            # Additional model-specific information
            model_info = {}
            if hasattr(model, 'best_params'):
                model_info['parameters'] = model.best_params
            if hasattr(model, 'performance_metrics'):
                model_info['training_metrics'] = model.performance_metrics
            
            evaluation_result = {
                'model_name': model_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'test_period': {
                    'start': test_data['ds'].min().isoformat(),
                    'end': test_data['ds'].max().isoformat(),
                    'periods': test_periods
                },
                'accuracy_metrics': metrics,
                'model_info': model_info,
                'predictions_summary': {
                    'mean_prediction': float(np.mean(predicted_values)),
                    'std_prediction': float(np.std(predicted_values)),
                    'min_prediction': float(np.min(predicted_values)),
                    'max_prediction': float(np.max(predicted_values))
                },
                'actual_summary': {
                    'mean_actual': float(np.mean(actual_values)),
                    'std_actual': float(np.std(actual_values)),
                    'min_actual': float(np.min(actual_values)),
                    'max_actual': float(np.max(actual_values))
                }
            }
            
            # Store results
            self.evaluation_results[model_name] = evaluation_result
            
            logger.info(f"{model_name} evaluation completed - MAE: {metrics['mae']:.4f}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name} model: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'evaluation_timestamp': datetime.now().isoformat()
            }
    
    def compare_models(self, evaluation_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare multiple models and determine the best performer
        
        Args:
            evaluation_results: Dictionary of model evaluation results
            
        Returns:
            Comparison results dictionary
        """
        try:
            logger.info("Comparing model performances...")
            
            if not evaluation_results:
                raise ValueError("No evaluation results provided for comparison")
            
            # Extract metrics for comparison
            comparison_metrics = ['mae', 'rmse', 'mape', 'r2']
            model_scores = {}
            
            for model_name, results in evaluation_results.items():
                if 'accuracy_metrics' in results:
                    model_scores[model_name] = results['accuracy_metrics']
            
            if not model_scores:
                raise ValueError("No valid accuracy metrics found for comparison")
            
            # Determine best model for each metric
            best_models = {}
            for metric in comparison_metrics:
                if metric == 'r2':  # Higher is better for R-squared
                    best_model = max(model_scores.keys(), 
                                   key=lambda x: model_scores[x].get(metric, -float('inf')))
                else:  # Lower is better for error metrics
                    best_model = min(model_scores.keys(), 
                                   key=lambda x: model_scores[x].get(metric, float('inf')))
                best_models[metric] = best_model
            
            # Overall best model (based on MAE as primary metric)
            overall_best = best_models.get('mae', list(model_scores.keys())[0])
            
            # Calculate relative performance
            relative_performance = {}
            for model_name in model_scores.keys():
                relative_performance[model_name] = {}
                for metric in comparison_metrics:
                    if metric in model_scores[model_name]:
                        relative_performance[model_name][metric] = model_scores[model_name][metric]
            
            comparison_result = {
                'comparison_timestamp': datetime.now().isoformat(),
                'models_compared': list(model_scores.keys()),
                'best_models_by_metric': best_models,
                'overall_best_model': overall_best,
                'relative_performance': relative_performance,
                'performance_summary': self._generate_performance_summary(model_scores),
                'recommendations': self._generate_recommendations(model_scores, best_models)
            }
            
            self.comparison_results = comparison_result
            logger.info(f"Model comparison completed. Best overall model: {overall_best}")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {'error': str(e), 'comparison_timestamp': datetime.now().isoformat()}
    
    def _generate_performance_summary(self, model_scores: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate a summary of model performances"""
        summary = {}
        
        for model_name, scores in model_scores.items():
            mae = scores.get('mae', float('inf'))
            rmse = scores.get('rmse', float('inf'))
            mape = scores.get('mape', float('inf'))
            r2 = scores.get('r2', 0)
            
            # Performance rating based on MAE
            if mae < 0.1:
                rating = "Excellent"
            elif mae < 0.2:
                rating = "Good"
            elif mae < 0.5:
                rating = "Fair"
            else:
                rating = "Poor"
            
            summary[model_name] = {
                'performance_rating': rating,
                'key_metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2
                }
            }
        
        return summary
    
    def _generate_recommendations(self, model_scores: Dict[str, Dict], 
                                best_models: Dict[str, str]) -> List[str]:
        """Generate recommendations based on model performance"""
        recommendations = []
        
        # Check if any model is clearly superior
        mae_scores = {name: scores.get('mae', float('inf')) for name, scores in model_scores.items()}
        best_mae = min(mae_scores.values())
        
        if best_mae < 0.1:
            recommendations.append("Excellent forecasting accuracy achieved. Model is ready for production use.")
        elif best_mae < 0.2:
            recommendations.append("Good forecasting accuracy. Consider monitoring performance in production.")
        else:
            recommendations.append("Forecasting accuracy could be improved. Consider data preprocessing or feature engineering.")
        
        # Check model consistency
        if len(set(best_models.values())) == 1:
            recommendations.append(f"Model {list(best_models.values())[0]} consistently performs best across all metrics.")
        else:
            recommendations.append("Different models perform best for different metrics. Consider ensemble methods.")
        
        return recommendations
