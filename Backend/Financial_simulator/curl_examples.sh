#!/bin/bash

# Forecast Engine v2 - API Testing Examples
# Usage: bash curl_examples.sh
# Make sure the server is running on localhost:8002

BASE_URL="http://localhost:8002"

echo "🚀 Forecast Engine v2 - API Testing Examples"
echo "=============================================="
echo "Base URL: $BASE_URL"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$4
    
    echo -e "${BLUE}Testing: $description${NC}"
    echo "Endpoint: $method $endpoint"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" -H "Content-Type: application/json" -d "$data" "$BASE_URL$endpoint")
    fi
    
    # Extract status code (last line)
    status_code=$(echo "$response" | tail -n1)
    # Extract response body (all but last line)
    body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" = "200" ]; then
        echo -e "${GREEN}✅ Success (200)${NC}"
        echo "Response preview:"
        echo "$body" | jq -r '.status // .message // .forecast_date // "Response received"' 2>/dev/null || echo "Response received"
    else
        echo -e "${RED}❌ Failed ($status_code)${NC}"
        echo "Error response:"
        echo "$body" | head -3
    fi
    echo ""
}

# Check if server is running
echo "🔍 Checking if server is running..."
if curl -s "$BASE_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Server is running${NC}"
else
    echo -e "${RED}❌ Server is not running. Please start the server first:${NC}"
    echo "cd Backend/Financial_simulator/Financial_simulator"
    echo "python langgraph_api.py"
    exit 1
fi
echo ""

# 1. Health Check
test_endpoint "GET" "/health" "" "Health Check"

# 2. System Metrics
test_endpoint "GET" "/metrics" "" "System Metrics"

# 3. Basic Forecast (7 days)
test_endpoint "GET" "/forecast?days=7" "" "Basic Forecast (7 days)"

# 4. Extended Forecast (30 days)
test_endpoint "GET" "/forecast?days=30&format=json" "" "Extended Forecast (30 days)"

# 5. Chart-ready Forecast
test_endpoint "GET" "/forecast-json?days=5" "" "Chart-ready Forecast (5 days)"

# 6. Agent Scoring - Available Agent
test_endpoint "POST" "/score-agent" '{"agent_id": "agent_001", "current_load": 10}' "Agent Scoring (Light Load)"

# 7. Agent Scoring - Busy Agent
test_endpoint "POST" "/score-agent" '{"agent_id": "agent_002", "current_load": 25}' "Agent Scoring (Heavy Load)"

# 8. Agent Scoring - Overloaded Agent
test_endpoint "POST" "/score-agent" '{"agent_id": "agent_003", "current_load": 35}' "Agent Scoring (Overloaded)"

# 9. Workflow Simulation
test_endpoint "POST" "/simulate-workflow" "" "End-to-End Workflow Simulation"

echo "🎯 Advanced Testing Examples"
echo "============================"

# 10. Forecast with different parameters
echo -e "${BLUE}Testing different forecast parameters:${NC}"
for days in 1 3 7 14 30; do
    echo "  📊 Forecast for $days days..."
    curl -s "$BASE_URL/forecast?days=$days" | jq -r '.content.forecast_days // "Error"' 2>/dev/null || echo "  Error"
done
echo ""

# 11. Multiple agent scoring
echo -e "${BLUE}Testing multiple agent scoring:${NC}"
for i in {1..5}; do
    load=$((RANDOM % 30 + 5))
    echo "  🤖 Scoring agent_$i with load $load..."
    curl -s -X POST -H "Content-Type: application/json" \
         -d "{\"agent_id\": \"agent_$i\", \"current_load\": $load}" \
         "$BASE_URL/score-agent" | jq -r '.agent_score.capacity_status // "Error"' 2>/dev/null || echo "  Error"
done
echo ""

# 12. Performance testing
echo -e "${BLUE}Performance Testing:${NC}"
echo "  ⏱️ Testing response times..."

start_time=$(date +%s.%N)
curl -s "$BASE_URL/health" > /dev/null
health_time=$(echo "$(date +%s.%N) - $start_time" | bc)

start_time=$(date +%s.%N)
curl -s "$BASE_URL/forecast?days=7" > /dev/null
forecast_time=$(echo "$(date +%s.%N) - $start_time" | bc)

start_time=$(date +%s.%N)
curl -s -X POST -H "Content-Type: application/json" \
     -d '{"agent_id": "perf_test", "current_load": 15}' \
     "$BASE_URL/score-agent" > /dev/null
score_time=$(echo "$(date +%s.%N) - $start_time" | bc)

echo "  Health check: ${health_time}s"
echo "  Forecast (7d): ${forecast_time}s"
echo "  Agent scoring: ${score_time}s"
echo ""

# 13. Error handling tests
echo -e "${BLUE}Error Handling Tests:${NC}"

echo "  🚫 Testing invalid endpoints..."
curl -s -w "%{http_code}" "$BASE_URL/invalid-endpoint" | tail -1
echo ""

echo "  🚫 Testing invalid JSON..."
curl -s -w "%{http_code}" -X POST -H "Content-Type: application/json" \
     -d '{"invalid": json}' "$BASE_URL/score-agent" | tail -1
echo ""

echo "  🚫 Testing missing parameters..."
curl -s -w "%{http_code}" -X POST -H "Content-Type: application/json" \
     -d '{}' "$BASE_URL/score-agent" | tail -1
echo ""

# 14. Sample data extraction
echo -e "${BLUE}Sample Data Extraction:${NC}"
echo "  📋 Extracting sample responses for documentation..."

# Save sample responses
mkdir -p samples

echo "  💾 Saving health check response..."
curl -s "$BASE_URL/health" | jq '.' > samples/health_response.json 2>/dev/null

echo "  💾 Saving forecast response..."
curl -s "$BASE_URL/forecast?days=3" | jq '.' > samples/forecast_response.json 2>/dev/null

echo "  💾 Saving chart data response..."
curl -s "$BASE_URL/forecast-json?days=3" | jq '.' > samples/chart_data_response.json 2>/dev/null

echo "  💾 Saving agent score response..."
curl -s -X POST -H "Content-Type: application/json" \
     -d '{"agent_id": "sample_agent", "current_load": 12}' \
     "$BASE_URL/score-agent" | jq '.' > samples/agent_score_response.json 2>/dev/null

echo "  ✅ Sample responses saved to samples/ directory"
echo ""

echo "🎉 API Testing Complete!"
echo "========================"
echo ""
echo "📊 Summary:"
echo "  - Health check and metrics endpoints tested"
echo "  - Forecast endpoints with various parameters tested"
echo "  - Agent scoring with different load scenarios tested"
echo "  - Workflow simulation tested"
echo "  - Performance and error handling tested"
echo "  - Sample responses saved for documentation"
echo ""
echo "🔗 Next Steps:"
echo "  1. Review sample responses in samples/ directory"
echo "  2. Check server logs for any errors"
echo "  3. Run full test suite: python test_forecast_engine.py"
echo "  4. Deploy to staging environment"
echo ""
echo "📚 Documentation:"
echo "  - API Documentation: FORECAST_ENGINE_README.md"
echo "  - Agent Specification: agent_spec.json"
echo "  - Sample Output: forecast_bridge.json"
echo ""

# Check if jq is available for JSON formatting
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}💡 Tip: Install 'jq' for better JSON formatting:${NC}"
    echo "  Ubuntu/Debian: sudo apt-get install jq"
    echo "  macOS: brew install jq"
    echo "  Windows: Download from https://stedolan.github.io/jq/"
fi
