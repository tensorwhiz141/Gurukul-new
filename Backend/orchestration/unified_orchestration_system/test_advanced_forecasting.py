#!/usr/bin/env python3
"""
Comprehensive Testing Framework for Advanced Forecasting System
Tests Prophet vs ARIMA performance and API endpoint functionality
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our forecasting modules
from enhanced_prophet_model import EnhancedProphetModel
from enhanced_arima_model import EnhancedARIMAModel
from model_performance_evaluator import ModelPerformanceEvaluator
from smart_model_selector import SmartModelSelector

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedForecastingTester:
    """
    Comprehensive testing framework for advanced forecasting capabilities
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8002"):
        """
        Initialize the testing framework
        
        Args:
            api_base_url: Base URL for API testing
        """
        self.api_base_url = api_base_url
        self.test_results = {}
        self.performance_evaluator = ModelPerformanceEvaluator()
        
    def generate_test_data(self, data_type: str = "general", periods: int = 100) -> pd.DataFrame:
        """
        Generate synthetic test data for different scenarios
        
        Args:
            data_type: Type of test data ('probability', 'load', 'general', 'seasonal')
            periods: Number of data points to generate
            
        Returns:
            Test dataframe
        """
        logger.info(f"Generating {data_type} test data with {periods} periods...")
        
        # Create date range
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        if data_type == 'probability':
            # Generate probability data (0-1 bounded with logistic growth)
            base_trend = np.logspace(-2, 0, periods) * 0.8
            noise = np.random.normal(0, 0.05, periods)
            values = np.clip(base_trend + noise, 0, 1)
            
        elif data_type == 'load':
            # Generate load data with seasonality
            trend = np.linspace(10, 50, periods)
            weekly_season = 10 * np.sin(2 * np.pi * np.arange(periods) / 7)
            monthly_season = 5 * np.sin(2 * np.pi * np.arange(periods) / 30)
            noise = np.random.normal(0, 3, periods)
            values = trend + weekly_season + monthly_season + noise
            values = np.maximum(values, 0)  # Ensure non-negative
            
        elif data_type == 'seasonal':
            # Generate data with strong seasonality
            trend = np.linspace(20, 80, periods)
            yearly_season = 15 * np.sin(2 * np.pi * np.arange(periods) / 365)
            weekly_season = 8 * np.sin(2 * np.pi * np.arange(periods) / 7)
            noise = np.random.normal(0, 2, periods)
            values = trend + yearly_season + weekly_season + noise
            
        else:  # general
            # Generate general time series data
            trend = np.linspace(100, 200, periods)
            noise = np.random.normal(0, 10, periods)
            values = trend + noise
        
        df = pd.DataFrame({
            'ds': dates,
            'y': values
        })
        
        logger.info(f"Generated test data - Mean: {values.mean():.2f}, Std: {values.std():.2f}")
        return df
    
    def test_prophet_model(self, test_data: pd.DataFrame, metric_type: str = "general") -> Dict[str, Any]:
        """
        Test Prophet model performance
        
        Args:
            test_data: Test dataset
            metric_type: Type of metric for model configuration
            
        Returns:
            Test results dictionary
        """
        logger.info(f"Testing Prophet model for {metric_type} metric...")
        
        try:
            # Split data
            train_data, test_data_split = self.performance_evaluator.train_test_split(test_data)
            
            # Initialize and fit Prophet model
            prophet_model = EnhancedProphetModel(metric_type)
            prophet_model.fit(train_data)
            
            # Generate predictions
            test_periods = len(test_data_split)
            predictions = prophet_model.predict(periods=test_periods)
            
            # Evaluate performance
            evaluation_result = self.performance_evaluator.evaluate_model(
                prophet_model, train_data, test_data_split, f'prophet_{metric_type}'
            )
            
            # Get forecast summary
            forecast_summary = prophet_model.get_forecast_summary(periods=30)
            
            # Cross-validation (if enough data)
            cv_metrics = {}
            if len(train_data) > 30:
                try:
                    cv_metrics = prophet_model.cross_validate_model(train_data)
                except Exception as e:
                    logger.warning(f"Cross-validation failed: {e}")
                    cv_metrics = {'error': str(e)}
            
            result = {
                'model_type': 'prophet',
                'metric_type': metric_type,
                'status': 'success',
                'evaluation': evaluation_result,
                'forecast_summary': forecast_summary,
                'cross_validation': cv_metrics,
                'training_time': 'completed',
                'test_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prophet test completed - MAE: {evaluation_result['accuracy_metrics']['mae']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Prophet model test failed: {e}")
            return {
                'model_type': 'prophet',
                'metric_type': metric_type,
                'status': 'failed',
                'error': str(e),
                'test_timestamp': datetime.now().isoformat()
            }
    
    def test_arima_model(self, test_data: pd.DataFrame, metric_type: str = "general") -> Dict[str, Any]:
        """
        Test ARIMA model performance
        
        Args:
            test_data: Test dataset
            metric_type: Type of metric for model configuration
            
        Returns:
            Test results dictionary
        """
        logger.info(f"Testing ARIMA model for {metric_type} metric...")
        
        try:
            # Split data
            train_data, test_data_split = self.performance_evaluator.train_test_split(test_data)
            
            # Initialize and fit ARIMA model
            arima_model = EnhancedARIMAModel(metric_type)
            arima_model.fit(train_data)
            
            # Generate predictions
            test_periods = len(test_data_split)
            predictions = arima_model.predict(periods=test_periods)
            
            # Evaluate performance
            evaluation_result = self.performance_evaluator.evaluate_model(
                arima_model, train_data, test_data_split, f'arima_{metric_type}'
            )
            
            # Get forecast summary
            forecast_summary = arima_model.get_forecast_summary(periods=30)
            
            # Get model diagnostics
            diagnostics = arima_model.diagnose_residuals()
            
            result = {
                'model_type': 'arima',
                'metric_type': metric_type,
                'status': 'success',
                'evaluation': evaluation_result,
                'forecast_summary': forecast_summary,
                'diagnostics': diagnostics,
                'best_params': arima_model.best_params,
                'test_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"ARIMA test completed - MAE: {evaluation_result['accuracy_metrics']['mae']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"ARIMA model test failed: {e}")
            return {
                'model_type': 'arima',
                'metric_type': metric_type,
                'status': 'failed',
                'error': str(e),
                'test_timestamp': datetime.now().isoformat()
            }
    
    def test_smart_model_selector(self, test_data: pd.DataFrame, metric_type: str = "general") -> Dict[str, Any]:
        """
        Test smart model selector
        
        Args:
            test_data: Test dataset
            metric_type: Type of metric for model configuration
            
        Returns:
            Test results dictionary
        """
        logger.info(f"Testing Smart Model Selector for {metric_type} metric...")
        
        try:
            selector = SmartModelSelector(metric_type)
            selection_result = selector.select_best_model(test_data)
            
            # Get recommendations
            if 'data_assessment' in selection_result:
                recommendations = selector.get_model_recommendations(selection_result['data_assessment'])
                selection_result['recommendations'] = recommendations
            
            # Export selection history
            selection_history = selector.export_selection_history()
            selection_result['selection_history'] = selection_history
            
            logger.info(f"Smart selector test completed - Selected: {selection_result['selected_model']}")
            return {
                'component': 'smart_model_selector',
                'metric_type': metric_type,
                'status': 'success',
                'result': selection_result,
                'test_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Smart model selector test failed: {e}")
            return {
                'component': 'smart_model_selector',
                'metric_type': metric_type,
                'status': 'failed',
                'error': str(e),
                'test_timestamp': datetime.now().isoformat()
            }
    
    def compare_model_performance(self, test_data: pd.DataFrame, metric_type: str = "general") -> Dict[str, Any]:
        """
        Compare Prophet vs ARIMA performance
        
        Args:
            test_data: Test dataset
            metric_type: Type of metric for model configuration
            
        Returns:
            Comparison results dictionary
        """
        logger.info(f"Comparing Prophet vs ARIMA performance for {metric_type} metric...")
        
        # Test both models
        prophet_result = self.test_prophet_model(test_data, metric_type)
        arima_result = self.test_arima_model(test_data, metric_type)
        
        # Extract evaluation results for comparison
        evaluation_results = {}
        if prophet_result['status'] == 'success':
            evaluation_results['prophet'] = prophet_result['evaluation']
        if arima_result['status'] == 'success':
            evaluation_results['arima'] = arima_result['evaluation']
        
        # Compare models
        comparison_result = {}
        if evaluation_results:
            comparison_result = self.performance_evaluator.compare_models(evaluation_results)
        
        return {
            'comparison_type': f'prophet_vs_arima_{metric_type}',
            'prophet_result': prophet_result,
            'arima_result': arima_result,
            'comparison': comparison_result,
            'test_timestamp': datetime.now().isoformat()
        }
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """
        Test API endpoints for forecasting functionality
        
        Returns:
            API test results dictionary
        """
        logger.info("Testing API endpoints...")
        
        api_tests = {}
        
        # Test health endpoint
        try:
            response = requests.get(f"{self.api_base_url}/", timeout=10)
            api_tests['health'] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'success': response.status_code == 200
            }
        except Exception as e:
            api_tests['health'] = {'error': str(e), 'success': False}
        
        # Test system status endpoint
        try:
            response = requests.get(f"{self.api_base_url}/system-status", timeout=10)
            api_tests['system_status'] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'success': response.status_code == 200
            }
            if response.status_code == 200:
                api_tests['system_status']['data'] = response.json()
        except Exception as e:
            api_tests['system_status'] = {'error': str(e), 'success': False}
        
        return {
            'api_base_url': self.api_base_url,
            'tests': api_tests,
            'test_timestamp': datetime.now().isoformat()
        }

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite

        Returns:
            Complete test results dictionary
        """
        logger.info("Starting comprehensive forecasting tests...")

        test_results = {
            'test_suite': 'advanced_forecasting',
            'start_time': datetime.now().isoformat(),
            'tests': {}
        }

        # Test different data types and scenarios
        test_scenarios = [
            ('probability', 'probability'),
            ('load', 'load'),
            ('general', 'general'),
            ('seasonal', 'general')
        ]

        for data_type, metric_type in test_scenarios:
            logger.info(f"Testing scenario: {data_type} data with {metric_type} metric...")

            # Generate test data
            test_data = self.generate_test_data(data_type, periods=80)

            # Test individual models
            prophet_result = self.test_prophet_model(test_data, metric_type)
            arima_result = self.test_arima_model(test_data, metric_type)

            # Test smart selector
            selector_result = self.test_smart_model_selector(test_data, metric_type)

            # Compare models
            comparison_result = self.compare_model_performance(test_data, metric_type)

            test_results['tests'][f'{data_type}_{metric_type}'] = {
                'data_type': data_type,
                'metric_type': metric_type,
                'prophet': prophet_result,
                'arima': arima_result,
                'smart_selector': selector_result,
                'comparison': comparison_result
            }

        # Test API endpoints
        api_test_results = self.test_api_endpoints()
        test_results['api_tests'] = api_test_results

        # Generate summary
        test_results['summary'] = self._generate_test_summary(test_results)
        test_results['end_time'] = datetime.now().isoformat()

        # Save results
        self._save_test_results(test_results)

        logger.info("Comprehensive tests completed!")
        return test_results

    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of test results"""
        summary = {
            'total_scenarios': len(test_results['tests']),
            'prophet_success_rate': 0,
            'arima_success_rate': 0,
            'selector_success_rate': 0,
            'api_success_rate': 0,
            'best_performing_model': {},
            'performance_metrics': {}
        }

        # Calculate success rates
        prophet_successes = 0
        arima_successes = 0
        selector_successes = 0

        for scenario_name, scenario_results in test_results['tests'].items():
            if scenario_results['prophet']['status'] == 'success':
                prophet_successes += 1
            if scenario_results['arima']['status'] == 'success':
                arima_successes += 1
            if scenario_results['smart_selector']['status'] == 'success':
                selector_successes += 1

        total_scenarios = len(test_results['tests'])
        if total_scenarios > 0:
            summary['prophet_success_rate'] = prophet_successes / total_scenarios
            summary['arima_success_rate'] = arima_successes / total_scenarios
            summary['selector_success_rate'] = selector_successes / total_scenarios

        # API success rate
        if 'api_tests' in test_results:
            api_tests = test_results['api_tests']['tests']
            api_successes = sum(1 for test in api_tests.values() if test.get('success', False))
            summary['api_success_rate'] = api_successes / len(api_tests) if api_tests else 0

        # Find best performing model overall
        model_performance = {'prophet': [], 'arima': []}

        for scenario_results in test_results['tests'].values():
            if scenario_results['prophet']['status'] == 'success':
                mae = scenario_results['prophet']['evaluation']['accuracy_metrics']['mae']
                model_performance['prophet'].append(mae)

            if scenario_results['arima']['status'] == 'success':
                mae = scenario_results['arima']['evaluation']['accuracy_metrics']['mae']
                model_performance['arima'].append(mae)

        # Calculate average performance
        for model, performances in model_performance.items():
            if performances:
                summary['performance_metrics'][model] = {
                    'average_mae': np.mean(performances),
                    'std_mae': np.std(performances),
                    'min_mae': np.min(performances),
                    'max_mae': np.max(performances)
                }

        # Determine best model
        if summary['performance_metrics']:
            best_model = min(summary['performance_metrics'].keys(),
                           key=lambda x: summary['performance_metrics'][x]['average_mae'])
            summary['best_performing_model'] = {
                'model': best_model,
                'average_mae': summary['performance_metrics'][best_model]['average_mae']
            }

        return summary

    def _save_test_results(self, test_results: Dict[str, Any]):
        """Save test results to file"""
        try:
            filename = f"advanced_forecasting_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            logger.info(f"Test results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

def main():
    """Main test execution function"""
    print("🚀 Advanced Forecasting System - Comprehensive Test Suite")
    print("=" * 70)

    # Initialize tester
    tester = AdvancedForecastingTester()

    try:
        # Run comprehensive tests
        results = tester.run_comprehensive_tests()

        # Print summary
        print("\n📊 Test Results Summary:")
        print("-" * 40)
        summary = results['summary']
        print(f"Total Scenarios Tested: {summary['total_scenarios']}")
        print(f"Prophet Success Rate: {summary['prophet_success_rate']:.1%}")
        print(f"ARIMA Success Rate: {summary['arima_success_rate']:.1%}")
        print(f"Smart Selector Success Rate: {summary['selector_success_rate']:.1%}")
        print(f"API Success Rate: {summary['api_success_rate']:.1%}")

        if summary['best_performing_model']:
            best = summary['best_performing_model']
            print(f"Best Performing Model: {best['model']} (MAE: {best['average_mae']:.4f})")

        print("\n✅ Testing completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        logger.error(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
