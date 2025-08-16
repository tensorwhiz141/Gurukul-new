# Forecast Engine v2 - Troubleshooting Guide

## Quick Diagnostics

### 1. Check System Status
```bash
# Health check
curl http://localhost:8002/health

# System metrics
curl http://localhost:8002/metrics

# Test core functionality
python test_forecast_engine.py --no-api
```

### 2. Check Logs
```bash
# Application logs
tail -f forecast_simulation_logs.txt

# Error logs (if available)
tail -f error.log

# System logs
journalctl -u forecast-engine -f
```

## Common Issues

### Installation Problems

#### Prophet Installation Failed
**Symptoms:**
- `ImportError: No module named 'prophet'`
- `Prophet not available. Install with: pip install prophet`

**Solutions:**
```bash
# Option 1: Install system dependencies first
sudo apt-get update
sudo apt-get install python3-dev gcc g++ python3-pip

# Option 2: Use conda (recommended)
conda install -c conda-forge prophet

# Option 3: Install from source
pip install pystan==2.19.1.1
pip install prophet

# Option 4: Use Docker
docker run -it python:3.9 bash
pip install prophet
```

**Verification:**
```python
try:
    from prophet import Prophet
    print("✅ Prophet installed successfully")
except ImportError as e:
    print(f"❌ Prophet installation failed: {e}")
```

#### Dependencies Conflict
**Symptoms:**
- `VersionConflict: package_name x.x.x is installed but package_name>=y.y.y is required`

**Solutions:**
```bash
# Create fresh virtual environment
python -m venv forecast_env
source forecast_env/bin/activate  # Linux/Mac
# or
forecast_env\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt

# Alternative: Use pip-tools
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
```

### Runtime Errors

#### Model Training Failures
**Symptoms:**
- `Error training Prophet model for delay_risk`
- `ARIMA model training failed`

**Diagnosis:**
```python
from prediction_agent import create_prediction_agent

# Test with minimal data
agent = create_prediction_agent(use_prophet=True)
data = agent.load_task_history_data()
print(f"Data loaded: {len(data)} metrics")

# Check data quality
for metric, df in data.items():
    print(f"{metric}: {len(df)} records, date range: {df['ds'].min()} to {df['ds'].max()}")
```

**Solutions:**
```python
# 1. Use ARIMA fallback
agent = create_prediction_agent(use_prophet=False)

# 2. Increase data size
# Ensure at least 30 data points for training

# 3. Check data format
# Ensure 'ds' column is datetime and 'y' column is numeric

# 4. Handle missing values
df = df.dropna()
df = df[df['y'] > 0]  # Remove negative values for certain metrics
```

#### API Timeout Errors
**Symptoms:**
- `Request timeout after 30 seconds`
- `504 Gateway Timeout`

**Solutions:**
```bash
# 1. Increase timeout
curl --timeout 60 http://localhost:8002/forecast?days=30

# 2. Reduce forecast days
curl http://localhost:8002/forecast?days=7

# 3. Check system resources
htop
free -h
df -h

# 4. Restart service
sudo systemctl restart forecast-engine
```

#### Memory Issues
**Symptoms:**
- `MemoryError: Unable to allocate array`
- `Out of memory`

**Solutions:**
```bash
# 1. Check memory usage
free -h
ps aux | grep python

# 2. Reduce data size
# Limit historical data to last 90 days

# 3. Use ARIMA instead of Prophet
export USE_PROPHET=false

# 4. Increase swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### API Issues

#### Connection Refused
**Symptoms:**
- `Connection refused`
- `Failed to connect to localhost:8002`

**Diagnosis:**
```bash
# Check if server is running
ps aux | grep langgraph_api
netstat -tlnp | grep 8002
curl http://localhost:8002/health
```

**Solutions:**
```bash
# 1. Start the server
cd Backend/Financial_simulator/Financial_simulator
python langgraph_api.py

# 2. Check port availability
lsof -i :8002
sudo netstat -tlnp | grep 8002

# 3. Use different port
export PORT=8003
python langgraph_api.py

# 4. Check firewall
sudo ufw status
sudo ufw allow 8002
```

#### Invalid JSON Response
**Symptoms:**
- `JSONDecodeError: Expecting value`
- `Invalid JSON format`

**Solutions:**
```bash
# 1. Check response format
curl -v http://localhost:8002/forecast?days=7

# 2. Validate JSON
curl http://localhost:8002/forecast?days=7 | jq '.'

# 3. Check server logs
tail -f forecast_simulation_logs.txt

# 4. Test with minimal request
curl http://localhost:8002/health
```

### Performance Issues

#### Slow Response Times
**Symptoms:**
- API responses taking >5 seconds
- High CPU usage

**Diagnosis:**
```bash
# Monitor performance
curl -w "@curl-format.txt" http://localhost:8002/forecast?days=30

# Create curl-format.txt:
echo "time_namelookup:  %{time_namelookup}\ntime_connect:     %{time_connect}\ntime_appconnect:  %{time_appconnect}\ntime_pretransfer: %{time_pretransfer}\ntime_redirect:    %{time_redirect}\ntime_starttransfer: %{time_starttransfer}\ntime_total:       %{time_total}\n" > curl-format.txt
```

**Solutions:**
```python
# 1. Enable caching
import functools

@functools.lru_cache(maxsize=128)
def cached_forecast(days):
    return agent.generate_forecasts(days)

# 2. Reduce forecast days
# Use 7 days instead of 30 for frequent requests

# 3. Use async processing
import asyncio
async def async_forecast():
    return await asyncio.to_thread(agent.generate_forecasts, 7)

# 4. Implement request queuing
from queue import Queue
forecast_queue = Queue(maxsize=10)
```

#### High Memory Usage
**Solutions:**
```python
# 1. Clear cached predictions periodically
agent.predictions.clear()

# 2. Limit historical data
agent.historical_data = {k: v.tail(100) for k, v in agent.historical_data.items()}

# 3. Use garbage collection
import gc
gc.collect()

# 4. Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

### Data Issues

#### No Historical Data
**Symptoms:**
- `No history files found in data`
- `Using mock data for missing metrics`

**Solutions:**
```bash
# 1. Check data directory
ls -la data/
ls -la data/*_person_history.json

# 2. Create sample data
mkdir -p data
python -c "
import json
from datetime import datetime, timedelta
data = []
for i in range(90):
    date = datetime.now() - timedelta(days=i)
    data.append({
        'timestamp': date.isoformat(),
        'discipline_score': 0.8,
        'behavior_score': 0.7,
        'total_expenses': 1000
    })
with open('data/sample_person_history.json', 'w') as f:
    json.dump(data, f)
"

# 3. Check data format
head -5 data/*_person_history.json
```

#### Invalid Data Format
**Symptoms:**
- `Error processing record`
- `KeyError: 'timestamp'`

**Solutions:**
```python
# Validate data format
import json
import pandas as pd

with open('data/sample_person_history.json', 'r') as f:
    data = json.load(f)

# Check required fields
required_fields = ['timestamp', 'discipline_score', 'behavior_score', 'total_expenses']
for record in data[:5]:  # Check first 5 records
    missing = [field for field in required_fields if field not in record]
    if missing:
        print(f"Missing fields: {missing}")

# Fix timestamp format
for record in data:
    if 'timestamp' in record:
        try:
            pd.to_datetime(record['timestamp'])
        except:
            print(f"Invalid timestamp: {record['timestamp']}")
```

## Monitoring and Maintenance

### Health Monitoring
```bash
# Create monitoring script
cat > monitor_forecast.sh << 'EOF'
#!/bin/bash
while true; do
    status=$(curl -s http://localhost:8002/health | jq -r '.status')
    if [ "$status" != "healthy" ]; then
        echo "$(date): Service unhealthy - $status"
        # Send alert or restart service
    fi
    sleep 60
done
EOF

chmod +x monitor_forecast.sh
./monitor_forecast.sh &
```

### Log Rotation
```bash
# Setup logrotate
sudo tee /etc/logrotate.d/forecast-engine << 'EOF'
/path/to/forecast_simulation_logs.txt {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
```

### Performance Tuning
```python
# Optimize Prophet parameters
model = Prophet(
    growth='linear',  # Use linear instead of logistic for better performance
    daily_seasonality=False,  # Disable if not needed
    weekly_seasonality=True,
    yearly_seasonality=False,
    seasonality_mode='additive',
    interval_width=0.8  # Reduce for faster training
)
```

## Getting Help

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python langgraph_api.py

# Run with verbose output
python -v test_forecast_engine.py
```

### Collect System Information
```bash
# Create debug report
cat > debug_report.sh << 'EOF'
#!/bin/bash
echo "=== System Information ==="
uname -a
python --version
pip list | grep -E "(prophet|pandas|numpy|statsmodels)"

echo "=== Service Status ==="
curl -s http://localhost:8002/health | jq '.'

echo "=== Resource Usage ==="
free -h
df -h
ps aux | grep python

echo "=== Recent Logs ==="
tail -20 forecast_simulation_logs.txt
EOF

chmod +x debug_report.sh
./debug_report.sh > debug_report.txt
```

### Contact Support
When reporting issues, please include:
1. Error messages and stack traces
2. System information (OS, Python version)
3. Configuration files (agent_spec.json)
4. Recent log entries
5. Steps to reproduce the issue

---

**Last Updated**: 2025-01-11  
**Version**: 2.0  
**Support**: Development Team
