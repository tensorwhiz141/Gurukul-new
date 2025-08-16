#!/bin/bash

echo "========================================"
echo "   Gurukul Unified Agent Mind"
echo "   Starting All Backend Services"
echo "========================================"
echo

# Get the directory where this script is located
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting all 5 backend services..."
echo

# Function to start a service in a new terminal
start_service() {
    local service_name="$1"
    local service_dir="$2"
    local command="$3"
    local port="$4"
    
    echo "📝 Starting $service_name on port $port..."
    
    # For different terminal emulators
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal --title="$service_name" -- bash -c "cd '$BASE_DIR/$service_dir' && $command; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -title "$service_name" -e "cd '$BASE_DIR/$service_dir' && $command; exec bash" &
    elif command -v konsole &> /dev/null; then
        konsole --title "$service_name" -e bash -c "cd '$BASE_DIR/$service_dir' && $command; exec bash" &
    elif command -v terminator &> /dev/null; then
        terminator --title="$service_name" -e "bash -c 'cd \"$BASE_DIR/$service_dir\" && $command; exec bash'" &
    else
        echo "⚠️  No supported terminal emulator found. Starting in background..."
        cd "$BASE_DIR/$service_dir"
        nohup $command > "${service_name// /_}.log" 2>&1 &
        cd "$BASE_DIR"
    fi
    
    sleep 2
}

# Start Memory Management API (Port 8003)
start_service "Memory Management API" "memory_management" "python run_server.py" "8003"

# Start API Data Service (Port 8001)
start_service "API Data Service" "api_data" "python api.py" "8001"

# Start Financial Simulator (Port 8002)
start_service "Financial Simulator" "Financial_simulator/Financial_simulator" "python langgraph_api.py" "8002"

# Start Lesson Generator (Port 8000)
start_service "Lesson Generator" "pipline-24-master" "python app.py" "8000"

# Start Augmed Kamal Middleware (Port 8004)
start_service "Augmed Kamal Middleware" "augmed kamal/augmed kamal" "python main.py" "8004"

echo
echo "✅ All 5 backend services are starting..."
echo
echo "🌐 Service URLs:"
echo "   Memory Management API: http://localhost:8003/memory/health"
echo "   API Data Service:      http://localhost:8001/health"
echo "   Financial Simulator:   http://localhost:8002/"
echo "   Lesson Generator:      http://localhost:8000/"
echo "   Augmed Kamal Middleware: http://localhost:8004/health"
echo
echo "📋 Next Steps:"
echo "   1. Wait 15-20 seconds for all services to start"
echo "   2. Check the service URLs above to verify they're running"
echo "   3. Verify Augmed Kamal authentication middleware is operational"
echo "   4. Start the frontend: cd gurukul_frontend-main && npm start"
echo "   5. Open http://localhost:3000 in your browser"
echo
echo "🔧 To stop all services: Close all the opened terminal windows"
echo "   Or run: pkill -f 'python.*api.py|python.*run_server.py|python.*langgraph_api.py|python.*app.py|python.*main.py'"
echo

# Function to check service health
check_services() {
    echo "🔍 Checking service health in 15 seconds..."
    sleep 15
    
    services=(
        "Memory Management API:http://localhost:8003/memory/health"
        "API Data Service:http://localhost:8001/health"
        "Financial Simulator:http://localhost:8002/"
        "Lesson Generator:http://localhost:8000/"
        "Augmed Kamal Middleware:http://localhost:8004/health"
    )
    
    echo
    echo "📊 Service Status:"
    echo "=================="
    
    for service in "${services[@]}"; do
        name="${service%%:*}"
        url="${service##*:}"
        
        if curl -s --max-time 5 "$url" > /dev/null 2>&1; then
            echo "✅ $name - Running"
        else
            echo "❌ $name - Not responding"
        fi
    done
    
    echo
    echo "🎯 If all services are running, you can now start the frontend!"
    echo "   cd ../gurukul_frontend-main && npm start"
}

# Ask if user wants to check service health
echo "Would you like to automatically check service health in 15 seconds? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    check_services &
fi

echo
echo "Press any key to exit..."
read -n 1 -s
