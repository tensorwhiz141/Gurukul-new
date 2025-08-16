# 🎯 **DOMAIN-SPECIFIC FORECASTING MODELS**

## 🎉 **OVERVIEW**

Advanced domain-specific forecasting models for **Edumentor** (educational metrics) and **Wellness Bot** (wellness metrics) with cross-domain integration and holistic insights.

---

## 🏗️ **ARCHITECTURE**

### **📚 Edumentor Prediction Agent**
- **File**: `edumentor_prediction_agent.py`
- **Purpose**: Educational metrics forecasting for learning progress and quiz performance
- **Models**: Prophet, ARIMA, Simple Moving Average (fallback)

### **🧘 Wellness Prediction Agent**
- **File**: `wellness_prediction_agent.py`
- **Purpose**: Wellness metrics forecasting for financial health and burnout risk
- **Models**: Prophet, ARIMA, Simple Moving Average (fallback)

### **🔗 Domain Integration System**
- **File**: `domain_specific_forecasting.py`
- **Purpose**: Combines both agents for cross-domain analysis and holistic recommendations
- **Features**: Cross-domain insights, intervention priority scoring

---

## 📊 **METRICS TRACKED**

### **🎓 Educational Metrics (Edumentor)**
| Metric | Description | Range | Risk Threshold |
|--------|-------------|-------|----------------|
| `quiz_completion_rate` | Quiz completion percentage | 0-1 | < 0.6 |
| `learning_progress_risk` | Risk of learning delays | 0-1 | > 0.7 |
| `concept_mastery_decline` | Forgetting curve impact | 0-1 | > 0.5 |
| `engagement_drop_risk` | Student disengagement risk | 0-1 | > 0.6 |
| `study_session_frequency` | Sessions per day | 0-8 | < 1.0 |
| `knowledge_retention_score` | Long-term retention | 0-1 | < 0.7 |
| `learning_velocity` | Learning pace relative to expected | 0-2 | < 0.5 |

### **🌿 Wellness Metrics (Wellness Bot)**
| Metric | Description | Range | Risk Threshold |
|--------|-------------|-------|----------------|
| `financial_health_score` | Overall financial wellness | 0-100 | < 60 |
| `burnout_risk` | Workplace burnout probability | 0-1 | > 0.7 |
| `spending_volatility` | Financial spending instability | 0-1 | > 0.6 |
| `stress_level_prediction` | Predicted stress levels | 0-1 | > 0.7 |
| `work_life_balance_score` | Work-life balance quality | 0-100 | < 50 |
| `emotional_wellness_index` | Emotional health score | 0-100 | < 60 |
| `financial_anxiety_level` | Financial stress indicator | 0-1 | > 0.6 |
| `sleep_quality_impact` | Sleep disruption from stress | 0-1 | > 0.6 |

---

## 🚀 **API ENDPOINTS**

### **📚 Edumentor Forecasting**
```http
POST /forecast/edumentor
Content-Type: application/json

{
  "user_id": "student_123",
  "domain": "edumentor",
  "forecast_days": 30,
  "metrics": ["quiz_completion_rate", "learning_progress_risk"]
}
```

### **🧘 Wellness Forecasting**
```http
POST /forecast/wellness
Content-Type: application/json

{
  "user_id": "user_456",
  "domain": "wellness", 
  "forecast_days": 30,
  "metrics": ["financial_health_score", "burnout_risk"]
}
```

### **🔗 Combined Forecasting**
```http
POST /forecast/combined
Content-Type: application/json

{
  "user_id": "user_789",
  "domain": "combined",
  "forecast_days": 30
}
```

### **📋 Available Domains**
```http
GET /forecast/domains
```

---

## 📈 **RESPONSE FORMAT**

### **🎓 Edumentor Response**
```json
{
  "student_id": "student_123",
  "domain": "edumentor",
  "forecast_type": "educational_metrics",
  "forecast_period_days": 30,
  "predictions": {
    "quiz_completion_rate": {
      "forecast_values": [...],
      "mean_forecast": 0.75,
      "trend": "increasing",
      "model_used": "prophet"
    }
  },
  "insights": [
    "Quiz completion rate is improving - current approach is effective"
  ],
  "risk_assessment": {
    "quiz_completion_rate": "LOW",
    "learning_progress_risk": "MEDIUM"
  },
  "recommendations": [
    "Implement adaptive quiz difficulty",
    "Provide immediate feedback during quizzes"
  ],
  "timestamp": "2024-12-25T10:00:00"
}
```

### **🔗 Combined Response**
```json
{
  "user_id": "user_789",
  "domain": "combined",
  "forecast_type": "multi_domain_metrics",
  "forecast_period_days": 30,
  "edumentor_forecast": { ... },
  "wellness_forecast": { ... },
  "cross_domain_insights": [
    "High stress detected in both learning and wellness domains"
  ],
  "holistic_recommendations": [
    "Learning: Implement adaptive quiz difficulty",
    "Wellness: Practice stress reduction techniques",
    "Holistic: Implement integrated stress management"
  ],
  "overall_risk_assessment": {
    "overall_risk_level": "MEDIUM",
    "total_high_risks": 1,
    "total_medium_risks": 3,
    "intervention_priority": [
      "Schedule check-in within one week",
      "Implement preventive measures"
    ]
  }
}
```

---

## 🧪 **TESTING**

### **Run Test Suite**
```bash
cd Backend/Financial_simulator
python test_domain_forecasting.py
```

### **Expected Output**
```
🧪 DOMAIN-SPECIFIC FORECASTING TEST SUITE
==================================================
🔍 Testing Dependencies...
✅ pandas: OK - Data manipulation
✅ numpy: OK - Numerical computing
✅ prophet: OK - Time series forecasting (optional)
✅ statsmodels: OK - Statistical models (optional)

🎓 Testing Edumentor Prediction Agent...
✅ Edumentor agent initialized successfully
✅ Educational data loaded: 7 metrics
✅ Learning outcome prediction successful

🧘 Testing Wellness Prediction Agent...
✅ Wellness agent initialized successfully
✅ Wellness data loaded: 8 metrics
✅ Wellness outcome prediction successful

🔮 Testing Domain-Specific Forecasting Integration...
✅ Domain forecasting system initialized
✅ Combined forecasting successful

🎉 ALL TESTS PASSED!
```

---

## 🔧 **INTEGRATION WITH EXISTING SYSTEM**

### **Financial Simulator Integration**
The domain-specific forecasting is integrated into the existing Financial Simulator API (`langgraph_api.py`) with:

- ✅ **Automatic Import**: Graceful fallback if dependencies missing
- ✅ **Error Handling**: Comprehensive error responses
- ✅ **Async Support**: Non-blocking forecasting operations
- ✅ **CORS Enabled**: Frontend integration ready

### **Frontend Integration**
Add to your frontend forecasting dashboard:

```javascript
// Edumentor forecasting
const edumentorForecast = await fetch('/forecast/edumentor', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 'student_123',
    domain: 'edumentor',
    forecast_days: 30
  })
});

// Wellness forecasting
const wellnessForecast = await fetch('/forecast/wellness', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 'user_456',
    domain: 'wellness',
    forecast_days: 30
  })
});

// Combined forecasting
const combinedForecast = await fetch('/forecast/combined', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 'user_789',
    domain: 'combined',
    forecast_days: 30
  })
});
```

---

## 🎯 **KEY FEATURES**

### **🔮 Advanced Forecasting**
- **Prophet Model**: Facebook's time series forecasting with seasonality
- **ARIMA Model**: Statistical forecasting with auto-parameter selection
- **Fallback System**: Simple moving average when advanced models fail

### **📊 Realistic Data Patterns**
- **Educational Patterns**: Weekend effects, semester breaks, forgetting curves
- **Wellness Patterns**: Work stress cycles, holiday spending, seasonal variations
- **Cross-Domain Correlations**: Learning stress vs wellness stress

### **🎯 Risk Assessment**
- **Threshold-Based**: Configurable risk thresholds for each metric
- **Multi-Level**: LOW, MEDIUM, HIGH, CRITICAL risk levels
- **Intervention Priority**: Automated priority scoring for support teams

### **💡 Intelligent Insights**
- **Domain-Specific**: Tailored insights for education and wellness
- **Cross-Domain**: Correlations between learning and wellness metrics
- **Holistic Recommendations**: Integrated advice across domains

---

## 🚨 **TROUBLESHOOTING**

### **Missing Dependencies**
```bash
# Install required packages
pip install pandas numpy

# Install optional forecasting packages
pip install prophet statsmodels
```

### **Import Errors**
- Check that all files are in the same directory
- Verify Python path includes the Financial_simulator directory
- Ensure all dependencies are installed

### **Forecasting Failures**
- Prophet/ARIMA models automatically fall back to simple forecasting
- Check logs for specific error messages
- Verify data format and ranges

---

## 🎉 **SUCCESS INDICATORS**

### **✅ System Ready**
- All dependencies installed
- Test suite passes completely
- API endpoints respond correctly
- Frontend integration working

### **✅ Forecasting Working**
- Predictions generated for all metrics
- Risk assessments calculated
- Insights and recommendations provided
- Cross-domain analysis functioning

### **✅ Production Ready**
- Error handling robust
- Fallback systems operational
- Performance optimized
- Documentation complete

---

**🚀 Your domain-specific forecasting system is ready for production use with advanced AI capabilities! 🎯**
