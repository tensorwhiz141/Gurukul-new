"""
Advanced Forecasting API Integration
Integrates Prophet and ARIMA models with the existing Gurukul orchestration system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import asyncio
from contextlib import asynccontextmanager

# Import our forecasting modules
from .enhanced_prophet_model import EnhancedProphetModel
from .enhanced_arima_model import EnhancedARIMAModel
from .model_performance_evaluator import ModelPerformanceEvaluator
from .smart_model_selector import SmartModelSelector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ForecastRequest(BaseModel):
    """Request model for forecasting"""
    data: List[Dict[str, Union[str, float]]] = Field(..., description="Time series data with 'date' and 'value' fields")
    metric_type: str = Field("general", description="Type of metric: 'probability', 'load', or 'general'")
    forecast_periods: int = Field(30, ge=1, le=365, description="Number of periods to forecast")
    model_preference: Optional[str] = Field(None, description="Preferred model: 'prophet', 'arima', or 'auto'")
    language: str = Field("en", description="Language for response")
    user_id: Optional[str] = Field(None, description="User ID for tracking")

class ForecastResponse(BaseModel):
    """Response model for forecasting"""
    status: str
    forecast_data: List[Dict[str, Any]]
    model_used: str
    accuracy_metrics: Dict[str, float]
    summary: Dict[str, Any]
    recommendations: List[str]
    timestamp: str
    language: str

class ModelComparisonRequest(BaseModel):
    """Request model for model comparison"""
    data: List[Dict[str, Union[str, float]]]
    metric_type: str = Field("general")
    language: str = Field("en")

class ModelComparisonResponse(BaseModel):
    """Response model for model comparison"""
    status: str
    comparison_results: Dict[str, Any]
    best_model: str
    performance_summary: Dict[str, Any]
    timestamp: str

class AdvancedForecastingAPI:
    """
    Advanced Forecasting API class for integration with Gurukul system
    """
    
    def __init__(self):
        """Initialize the forecasting API"""
        self.performance_evaluator = ModelPerformanceEvaluator()
        self.active_models = {}
        self.forecast_cache = {}
        
    def prepare_data_from_request(self, request_data: List[Dict[str, Union[str, float]]]) -> pd.DataFrame:
        """
        Prepare data from API request format
        
        Args:
            request_data: List of data points with date and value
            
        Returns:
            Prepared DataFrame
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(request_data)
            
            # Standardize column names
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'ds'})
            if 'value' in df.columns:
                df = df.rename(columns={'value': 'y'})
            
            # Ensure required columns exist
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("Data must contain 'date' and 'value' fields")
            
            # Convert date column
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            
            # Sort by date and remove duplicates
            df = df.sort_values('ds').drop_duplicates(subset=['ds']).reset_index(drop=True)
            
            # Handle missing values
            df['y'] = df['y'].fillna(df['y'].median())
            
            logger.info(f"Prepared data: {len(df)} records from {df['ds'].min()} to {df['ds'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise HTTPException(status_code=400, detail=f"Data preparation failed: {str(e)}")
    
    async def generate_forecast(self, request: ForecastRequest) -> ForecastResponse:
        """
        Generate forecast using advanced models
        
        Args:
            request: Forecast request
            
        Returns:
            Forecast response
        """
        try:
            logger.info(f"Generating forecast for {request.metric_type} metric, {request.forecast_periods} periods")
            
            # Prepare data
            df = self.prepare_data_from_request(request.data)
            
            # Validate data
            if len(df) < 10:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient data points: {len(df)}. Need at least 10 points."
                )
            
            # Select model
            if request.model_preference == "auto" or request.model_preference is None:
                selector = SmartModelSelector(request.metric_type)
                selection_result = selector.select_best_model(df)
                
                if selection_result['selected_model'] == 'prophet':
                    model = selection_result['model_object']
                elif selection_result['selected_model'] == 'arima':
                    model = selection_result['model_object']
                else:
                    # Fallback to simple forecast
                    return await self._generate_simple_forecast(request, df)
                
                model_used = selection_result['selected_model']
                
            elif request.model_preference == "prophet":
                model = EnhancedProphetModel(request.metric_type)
                model.fit(df)
                model_used = "prophet"
                
            elif request.model_preference == "arima":
                model = EnhancedARIMAModel(request.metric_type)
                model.fit(df)
                model_used = "arima"
                
            else:
                raise HTTPException(status_code=400, detail="Invalid model preference")
            
            # Generate forecast
            forecast_df = model.predict(periods=request.forecast_periods)
            
            # Calculate performance metrics if possible
            accuracy_metrics = {}
            if len(df) > 20:
                try:
                    train_data, test_data = self.performance_evaluator.train_test_split(df, test_size=0.2)
                    temp_model = type(model)(request.metric_type)
                    temp_model.fit(train_data)
                    temp_predictions = temp_model.predict(periods=len(test_data))
                    
                    if hasattr(temp_predictions, 'values'):
                        pred_values = temp_predictions['yhat'].values if 'yhat' in temp_predictions.columns else temp_predictions.values
                    else:
                        pred_values = temp_predictions
                    
                    accuracy_metrics = self.performance_evaluator.calculate_accuracy_metrics(
                        test_data['y'].values, pred_values
                    )
                except Exception as e:
                    logger.warning(f"Could not calculate accuracy metrics: {e}")
                    accuracy_metrics = {"note": "Accuracy metrics not available"}
            
            # Prepare forecast data for response
            forecast_data = []
            for _, row in forecast_df.iterrows():
                forecast_data.append({
                    "date": row['ds'].isoformat(),
                    "predicted_value": float(row['yhat']),
                    "lower_bound": float(row['yhat_lower']) if 'yhat_lower' in row else None,
                    "upper_bound": float(row['yhat_upper']) if 'yhat_upper' in row else None
                })
            
            # Generate summary
            summary = {
                "forecast_start": forecast_df['ds'].min().isoformat(),
                "forecast_end": forecast_df['ds'].max().isoformat(),
                "mean_prediction": float(forecast_df['yhat'].mean()),
                "trend": "increasing" if forecast_df['yhat'].iloc[-1] > forecast_df['yhat'].iloc[0] else "decreasing",
                "confidence_interval_width": float(forecast_df['yhat_upper'].mean() - forecast_df['yhat_lower'].mean()) if 'yhat_upper' in forecast_df.columns else None
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(df, forecast_df, accuracy_metrics, request.metric_type)
            
            # Create multilingual response
            response_content = self._create_multilingual_response(
                summary, recommendations, request.language
            )
            
            return ForecastResponse(
                status="success",
                forecast_data=forecast_data,
                model_used=model_used,
                accuracy_metrics=accuracy_metrics,
                summary=response_content["summary"],
                recommendations=response_content["recommendations"],
                timestamp=datetime.now().isoformat(),
                language=request.language
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")
    
    async def _generate_simple_forecast(self, request: ForecastRequest, df: pd.DataFrame) -> ForecastResponse:
        """Generate simple linear forecast as fallback"""
        try:
            # Simple linear trend forecast
            x = np.arange(len(df))
            y = df['y'].values
            
            # Fit linear regression
            coeffs = np.polyfit(x, y, 1)
            
            # Generate future predictions
            future_x = np.arange(len(df), len(df) + request.forecast_periods)
            future_y = np.polyval(coeffs, future_x)
            
            # Create future dates
            last_date = df['ds'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=request.forecast_periods,
                freq='D'
            )
            
            # Prepare forecast data
            forecast_data = []
            for i, (date, value) in enumerate(zip(future_dates, future_y)):
                forecast_data.append({
                    "date": date.isoformat(),
                    "predicted_value": float(value),
                    "lower_bound": None,
                    "upper_bound": None
                })
            
            summary = {
                "forecast_start": future_dates[0].isoformat(),
                "forecast_end": future_dates[-1].isoformat(),
                "mean_prediction": float(np.mean(future_y)),
                "trend": "increasing" if coeffs[0] > 0 else "decreasing",
                "confidence_interval_width": None
            }
            
            recommendations = [
                "Simple linear forecast used due to data limitations",
                "Consider collecting more data for advanced forecasting models",
                "Monitor actual values to improve future predictions"
            ]
            
            return ForecastResponse(
                status="success",
                forecast_data=forecast_data,
                model_used="simple_linear",
                accuracy_metrics={"note": "Simple forecast - accuracy metrics not available"},
                summary=summary,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat(),
                language=request.language
            )
            
        except Exception as e:
            logger.error(f"Simple forecast generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Simple forecast failed: {str(e)}")
    
    def _generate_recommendations(self, historical_data: pd.DataFrame, forecast_data: pd.DataFrame, 
                                accuracy_metrics: Dict[str, float], metric_type: str) -> List[str]:
        """Generate recommendations based on forecast results"""
        recommendations = []
        
        # Data quality recommendations
        if len(historical_data) < 30:
            recommendations.append("Consider collecting more historical data for improved accuracy")
        
        # Accuracy-based recommendations
        mae = accuracy_metrics.get('mae', float('inf'))
        if mae < 0.1:
            recommendations.append("Excellent forecast accuracy - model is reliable for decision making")
        elif mae < 0.2:
            recommendations.append("Good forecast accuracy - suitable for planning purposes")
        else:
            recommendations.append("Moderate forecast accuracy - use with caution for critical decisions")
        
        # Trend-based recommendations
        trend_direction = "increasing" if forecast_data['yhat'].iloc[-1] > forecast_data['yhat'].iloc[0] else "decreasing"
        
        if metric_type == "probability" and trend_direction == "increasing":
            recommendations.append("Risk probability is trending upward - consider preventive measures")
        elif metric_type == "load" and trend_direction == "increasing":
            recommendations.append("Load is expected to increase - plan for additional capacity")
        
        # Seasonality recommendations
        if len(historical_data) > 30:
            variance = historical_data['y'].var()
            if variance > historical_data['y'].mean():
                recommendations.append("High variability detected - monitor closely for unexpected changes")
        
        return recommendations

    def _create_multilingual_response(self, summary: Dict[str, Any], recommendations: List[str],
                                    language: str) -> Dict[str, Any]:
        """Create multilingual response content"""
        # For now, return English content
        # In a full implementation, this would translate based on language parameter
        return {
            "summary": summary,
            "recommendations": recommendations,
            "language": language
        }

    async def compare_models(self, request: ModelComparisonRequest) -> ModelComparisonResponse:
        """
        Compare Prophet vs ARIMA model performance

        Args:
            request: Model comparison request

        Returns:
            Model comparison response
        """
        try:
            logger.info(f"Comparing models for {request.metric_type} metric")

            # Prepare data
            df = self.prepare_data_from_request(request.data)

            # Validate data
            if len(df) < 20:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient data for comparison: {len(df)}. Need at least 20 points."
                )

            # Split data for evaluation
            train_data, test_data = self.performance_evaluator.train_test_split(df)

            evaluation_results = {}

            # Test Prophet model
            try:
                prophet_model = EnhancedProphetModel(request.metric_type)
                prophet_model.fit(train_data)
                prophet_eval = self.performance_evaluator.evaluate_model(
                    prophet_model, train_data, test_data, 'prophet'
                )
                evaluation_results['prophet'] = prophet_eval
            except Exception as e:
                logger.warning(f"Prophet evaluation failed: {e}")

            # Test ARIMA model
            try:
                arima_model = EnhancedARIMAModel(request.metric_type)
                arima_model.fit(train_data)
                arima_eval = self.performance_evaluator.evaluate_model(
                    arima_model, train_data, test_data, 'arima'
                )
                evaluation_results['arima'] = arima_eval
            except Exception as e:
                logger.warning(f"ARIMA evaluation failed: {e}")

            if not evaluation_results:
                raise HTTPException(status_code=500, detail="Both models failed to evaluate")

            # Compare models
            comparison_results = self.performance_evaluator.compare_models(evaluation_results)

            # Generate performance summary
            performance_summary = {}
            for model_name, eval_result in evaluation_results.items():
                if 'accuracy_metrics' in eval_result:
                    metrics = eval_result['accuracy_metrics']
                    performance_summary[model_name] = {
                        'mae': metrics.get('mae', float('inf')),
                        'rmse': metrics.get('rmse', float('inf')),
                        'mape': metrics.get('mape', float('inf')),
                        'r2': metrics.get('r2', 0)
                    }

            best_model = comparison_results.get('overall_best_model', 'unknown')

            return ModelComparisonResponse(
                status="success",
                comparison_results=comparison_results,
                best_model=best_model,
                performance_summary=performance_summary,
                timestamp=datetime.now().isoformat()
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

# Create FastAPI app instance
app = FastAPI(
    title="Advanced Forecasting API",
    description="Prophet and ARIMA-based forecasting for Gurukul system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API instance
forecasting_api = AdvancedForecastingAPI()

@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate time series forecast"""
    return await forecasting_api.generate_forecast(request)

@app.post("/compare-models", response_model=ModelComparisonResponse)
async def compare_models(request: ModelComparisonRequest):
    """Compare Prophet vs ARIMA model performance"""
    return await forecasting_api.compare_models(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Advanced Forecasting API",
        "timestamp": datetime.now().isoformat(),
        "models_available": ["prophet", "arima", "auto"],
        "metric_types": ["probability", "load", "general"]
    }

@app.get("/forecast/status")
async def forecast_status():
    """Get forecasting system status"""
    try:
        # Check if Prophet is available
        prophet_available = False
        try:
            from prophet import Prophet
            prophet_available = True
        except ImportError:
            pass

        # Check if ARIMA is available
        arima_available = False
        try:
            from statsmodels.tsa.arima.model import ARIMA
            arima_available = True
        except ImportError:
            pass

        return {
            "forecasting_enabled": True,
            "system_status": "operational",
            "models": {
                "prophet": {
                    "available": prophet_available,
                    "status": "active" if prophet_available else "unavailable"
                },
                "arima": {
                    "available": arima_available,
                    "status": "active" if arima_available else "unavailable"
                },
                "simple": {
                    "available": True,
                    "status": "active"
                }
            },
            "endpoints": {
                "forecast": "/forecast",
                "compare_models": "/compare-models",
                "gurukul_integration": "/gurukul/forecast",
                "status": "/forecast/status"
            },
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "forecasting_enabled": False,
            "system_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/models/status")
async def models_status():
    """Get status of available models"""
    return {
        "prophet": {
            "available": True,
            "description": "Facebook Prophet with logistic/linear growth",
            "best_for": ["seasonal data", "trend analysis", "probability metrics"]
        },
        "arima": {
            "available": True,
            "description": "Auto ARIMA with parameter optimization",
            "best_for": ["stationary data", "short-term forecasts", "load metrics"]
        },
        "auto": {
            "available": True,
            "description": "Smart model selection based on data characteristics",
            "best_for": ["unknown data patterns", "general forecasting"]
        }
    }

# Integration endpoints for existing Gurukul system
@app.post("/gurukul/forecast")
async def gurukul_forecast_integration(
    data: List[Dict[str, Union[str, float]]],
    metric_type: str = "general",
    forecast_periods: int = 30,
    language: str = "en",
    user_id: Optional[str] = None
):
    """
    Gurukul system integration endpoint for forecasting
    Maintains compatibility with existing multi-agent workflow
    """
    try:
        request = ForecastRequest(
            data=data,
            metric_type=metric_type,
            forecast_periods=forecast_periods,
            language=language,
            user_id=user_id
        )

        forecast_result = await forecasting_api.generate_forecast(request)

        # Format response for Gurukul system compatibility
        gurukul_response = {
            "report_type": "forecast",
            "language": language,
            "sentiment": "neutral",
            "content": {
                "forecast_data": forecast_result.forecast_data,
                "model_used": forecast_result.model_used,
                "summary": forecast_result.summary,
                "recommendations": forecast_result.recommendations,
                "accuracy_metrics": forecast_result.accuracy_metrics
            },
            "metadata": {
                "timestamp": forecast_result.timestamp,
                "user_id": user_id,
                "metric_type": metric_type,
                "forecast_periods": forecast_periods
            }
        }

        return gurukul_response

    except Exception as e:
        logger.error(f"Gurukul forecast integration failed: {e}")
        return {
            "report_type": "forecast",
            "language": language,
            "sentiment": "negative",
            "content": {
                "error": str(e),
                "message": "Forecast generation failed"
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "error": True
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
