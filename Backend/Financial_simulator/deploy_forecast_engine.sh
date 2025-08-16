#!/bin/bash

# Forecast Engine v2 - Deployment Script
# Version: 2.0
# Date: 2025-01-11

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Forecast Engine v2 - Deployment Script${NC}"
echo "=============================================="
echo "Version: 2.0"
echo "Date: $(date)"
echo ""

# Configuration
PYTHON_VERSION="3.8"
SERVICE_NAME="forecast-engine"
SERVICE_PORT="8002"
LOG_DIR="/var/log/forecast-engine"
DATA_DIR="/opt/forecast-engine/data"
CONFIG_DIR="/etc/forecast-engine"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_info "Python version: $PYTHON_VER"
        
        if [[ $(echo "$PYTHON_VER >= $PYTHON_VERSION" | bc -l) -eq 0 ]]; then
            log_error "Python $PYTHON_VERSION or higher required"
            exit 1
        fi
    else
        log_error "Python3 not found"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 not found"
        exit 1
    fi
    
    # Check system dependencies for Prophet
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! dpkg -l | grep -q python3-dev; then
            log_warn "python3-dev not installed, installing..."
            sudo apt-get update
            sudo apt-get install -y python3-dev gcc g++
        fi
    fi
    
    log_info "✅ System requirements check passed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    log_info "✅ Environment setup complete"
}

install_dependencies() {
    log_info "Installing dependencies..."
    
    # Install Prophet first (can be tricky)
    log_info "Installing Prophet..."
    pip install prophet>=1.1.4 || {
        log_warn "Prophet installation failed, trying alternative method..."
        pip install pystan==2.19.1.1
        pip install prophet
    }
    
    # Install other requirements
    log_info "Installing other dependencies..."
    pip install -r requirements.txt
    
    # Verify installations
    python -c "from prophet import Prophet; print('✅ Prophet installed')" || {
        log_error "Prophet installation verification failed"
        exit 1
    }
    
    python -c "import pandas, numpy, statsmodels; print('✅ Core dependencies installed')" || {
        log_error "Core dependencies verification failed"
        exit 1
    }
    
    log_info "✅ Dependencies installed successfully"
}

run_tests() {
    log_info "Running tests..."
    
    # Run core functionality tests
    python test_forecast_engine.py --no-api || {
        log_error "Core functionality tests failed"
        exit 1
    }
    
    log_info "✅ Tests passed"
}

setup_directories() {
    log_info "Setting up directories..."
    
    # Create log directory
    sudo mkdir -p "$LOG_DIR"
    sudo chown $USER:$USER "$LOG_DIR"
    
    # Create data directory
    sudo mkdir -p "$DATA_DIR"
    sudo chown $USER:$USER "$DATA_DIR"
    
    # Create config directory
    sudo mkdir -p "$CONFIG_DIR"
    sudo chown $USER:$USER "$CONFIG_DIR"
    
    # Copy configuration files
    cp agent_spec.json "$CONFIG_DIR/"
    cp forecast_bridge.json "$CONFIG_DIR/"
    
    log_info "✅ Directories setup complete"
}

create_systemd_service() {
    log_info "Creating systemd service..."
    
    # Get current directory
    CURRENT_DIR=$(pwd)
    
    # Create service file
    sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=Forecast Engine v2 - Advanced Time Series Forecasting
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$CURRENT_DIR
Environment=PATH=$CURRENT_DIR/venv/bin
ExecStart=$CURRENT_DIR/venv/bin/python langgraph_api.py
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/forecast-engine.log
StandardError=append:$LOG_DIR/forecast-engine-error.log

# Environment variables
Environment=PYTHONPATH=$CURRENT_DIR
Environment=LOG_LEVEL=INFO
Environment=PORT=$SERVICE_PORT

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    log_info "✅ Systemd service created"
}

setup_logrotate() {
    log_info "Setting up log rotation..."
    
    sudo tee /etc/logrotate.d/${SERVICE_NAME} > /dev/null << EOF
$LOG_DIR/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su $USER $USER
}
EOF
    
    log_info "✅ Log rotation setup complete"
}

start_service() {
    log_info "Starting service..."
    
    # Enable and start service
    sudo systemctl enable ${SERVICE_NAME}
    sudo systemctl start ${SERVICE_NAME}
    
    # Wait for service to start
    sleep 5
    
    # Check service status
    if sudo systemctl is-active --quiet ${SERVICE_NAME}; then
        log_info "✅ Service started successfully"
    else
        log_error "Service failed to start"
        sudo systemctl status ${SERVICE_NAME}
        exit 1
    fi
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Wait for service to be ready
    sleep 10
    
    # Test health endpoint
    if curl -s "http://localhost:$SERVICE_PORT/health" | grep -q "healthy"; then
        log_info "✅ Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi
    
    # Test forecast endpoint
    if curl -s "http://localhost:$SERVICE_PORT/forecast?days=3" | grep -q "forecast"; then
        log_info "✅ Forecast endpoint working"
    else
        log_error "Forecast endpoint failed"
        exit 1
    fi
    
    # Test agent scoring endpoint
    if curl -s -X POST -H "Content-Type: application/json" \
            -d '{"agent_id": "test", "current_load": 10}' \
            "http://localhost:$SERVICE_PORT/score-agent" | grep -q "agent_score"; then
        log_info "✅ Agent scoring endpoint working"
    else
        log_error "Agent scoring endpoint failed"
        exit 1
    fi
    
    log_info "✅ Deployment verification complete"
}

create_monitoring_script() {
    log_info "Creating monitoring script..."
    
    cat > monitor_forecast_engine.sh << 'EOF'
#!/bin/bash

# Forecast Engine v2 - Monitoring Script

SERVICE_NAME="forecast-engine"
SERVICE_PORT="8002"
LOG_FILE="/var/log/forecast-engine/monitor.log"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

check_service() {
    if ! systemctl is-active --quiet $SERVICE_NAME; then
        log_message "ERROR: Service $SERVICE_NAME is not running"
        log_message "INFO: Attempting to restart service"
        sudo systemctl restart $SERVICE_NAME
        sleep 10
        
        if systemctl is-active --quiet $SERVICE_NAME; then
            log_message "INFO: Service restarted successfully"
        else
            log_message "ERROR: Failed to restart service"
            return 1
        fi
    fi
    return 0
}

check_health() {
    local response=$(curl -s --max-time 10 "http://localhost:$SERVICE_PORT/health")
    if echo "$response" | grep -q "healthy"; then
        return 0
    else
        log_message "ERROR: Health check failed - $response"
        return 1
    fi
}

check_performance() {
    local cpu_usage=$(ps -p $(pgrep -f langgraph_api) -o %cpu --no-headers 2>/dev/null | tr -d ' ')
    local mem_usage=$(ps -p $(pgrep -f langgraph_api) -o %mem --no-headers 2>/dev/null | tr -d ' ')
    
    if [[ -n "$cpu_usage" && -n "$mem_usage" ]]; then
        log_message "INFO: CPU: ${cpu_usage}%, Memory: ${mem_usage}%"
        
        # Alert if usage is too high
        if (( $(echo "$cpu_usage > 80" | bc -l) )); then
            log_message "WARN: High CPU usage: ${cpu_usage}%"
        fi
        
        if (( $(echo "$mem_usage > 80" | bc -l) )); then
            log_message "WARN: High memory usage: ${mem_usage}%"
        fi
    fi
}

# Main monitoring loop
while true; do
    if check_service && check_health; then
        check_performance
    else
        log_message "ERROR: Service health check failed"
    fi
    
    sleep 60  # Check every minute
done
EOF
    
    chmod +x monitor_forecast_engine.sh
    
    log_info "✅ Monitoring script created"
}

print_summary() {
    echo ""
    echo -e "${GREEN}🎉 Forecast Engine v2 Deployment Complete!${NC}"
    echo "=============================================="
    echo ""
    echo "📊 Service Information:"
    echo "  Service Name: $SERVICE_NAME"
    echo "  Port: $SERVICE_PORT"
    echo "  Status: $(sudo systemctl is-active $SERVICE_NAME)"
    echo ""
    echo "📁 Important Paths:"
    echo "  Logs: $LOG_DIR"
    echo "  Data: $DATA_DIR"
    echo "  Config: $CONFIG_DIR"
    echo ""
    echo "🔗 API Endpoints:"
    echo "  Health: http://localhost:$SERVICE_PORT/health"
    echo "  Forecast: http://localhost:$SERVICE_PORT/forecast?days=7"
    echo "  Chart Data: http://localhost:$SERVICE_PORT/forecast-json?days=7"
    echo "  Agent Score: http://localhost:$SERVICE_PORT/score-agent"
    echo ""
    echo "🛠️ Management Commands:"
    echo "  Start: sudo systemctl start $SERVICE_NAME"
    echo "  Stop: sudo systemctl stop $SERVICE_NAME"
    echo "  Restart: sudo systemctl restart $SERVICE_NAME"
    echo "  Status: sudo systemctl status $SERVICE_NAME"
    echo "  Logs: sudo journalctl -u $SERVICE_NAME -f"
    echo ""
    echo "📋 Next Steps:"
    echo "  1. Test API endpoints: bash curl_examples.sh"
    echo "  2. Start monitoring: ./monitor_forecast_engine.sh &"
    echo "  3. Configure team integrations"
    echo "  4. Set up production monitoring"
    echo ""
    echo "📚 Documentation:"
    echo "  README: FORECAST_ENGINE_README.md"
    echo "  Troubleshooting: TROUBLESHOOTING.md"
    echo "  API Examples: curl_examples.sh"
    echo ""
    echo -e "${GREEN}✅ Ready for Production!${NC}"
}

# Main deployment flow
main() {
    log_info "Starting Forecast Engine v2 deployment..."
    
    check_requirements
    setup_environment
    install_dependencies
    run_tests
    setup_directories
    create_systemd_service
    setup_logrotate
    start_service
    verify_deployment
    create_monitoring_script
    
    print_summary
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "test")
        log_info "Running tests only..."
        setup_environment
        run_tests
        ;;
    "start")
        log_info "Starting service..."
        sudo systemctl start $SERVICE_NAME
        ;;
    "stop")
        log_info "Stopping service..."
        sudo systemctl stop $SERVICE_NAME
        ;;
    "restart")
        log_info "Restarting service..."
        sudo systemctl restart $SERVICE_NAME
        ;;
    "status")
        sudo systemctl status $SERVICE_NAME
        ;;
    "logs")
        sudo journalctl -u $SERVICE_NAME -f
        ;;
    *)
        echo "Usage: $0 {deploy|test|start|stop|restart|status|logs}"
        exit 1
        ;;
esac
