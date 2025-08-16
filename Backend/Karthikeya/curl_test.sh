#!/bin/bash

# Karthikeya Multilingual Reporting Engine - API Test Suite
# Automated curl commands for testing /generate-report and /generate-nudge endpoints

set -e  # Exit on any error

# Configuration
BASE_URL="http://localhost:5000"
CONTENT_TYPE="Content-Type: application/json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Karthikeya API Test Suite${NC}"
echo "=================================="

# Function to test endpoint
test_endpoint() {
    local endpoint=$1
    local data_file=$2
    local description=$3
    
    echo -e "\n${YELLOW}Testing: $description${NC}"
    echo "Endpoint: $BASE_URL$endpoint"
    echo "Data file: $data_file"
    
    if [ ! -f "$data_file" ]; then
        echo -e "${RED}❌ Data file not found: $data_file${NC}"
        return 1
    fi
    
    echo "Request payload:"
    cat "$data_file" | jq '.' 2>/dev/null || cat "$data_file"
    
    echo -e "\n${BLUE}Sending request...${NC}"
    
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "$CONTENT_TYPE" \
        -d @"$data_file" \
        "$BASE_URL$endpoint")
    
    # Extract HTTP status code (last line)
    http_code=$(echo "$response" | tail -n1)
    # Extract response body (all but last line)
    response_body=$(echo "$response" | head -n -1)
    
    echo "HTTP Status: $http_code"
    
    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✅ Success!${NC}"
        echo "Response:"
        echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
    else
        echo -e "${RED}❌ Failed with status $http_code${NC}"
        echo "Response:"
        echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
        return 1
    fi
}

# Function to check if server is running
check_server() {
    echo -e "${BLUE}Checking if server is running...${NC}"
    
    if curl -s "$BASE_URL/health" > /dev/null; then
        echo -e "${GREEN}✅ Server is running${NC}"
        
        # Get server info
        echo "Server info:"
        curl -s "$BASE_URL/health" | jq '.' 2>/dev/null || curl -s "$BASE_URL/health"
    else
        echo -e "${RED}❌ Server is not running at $BASE_URL${NC}"
        echo "Please start the server with: python app_modular.py"
        exit 1
    fi
}

# Function to test health endpoint
test_health() {
    echo -e "\n${YELLOW}Testing Health Endpoint${NC}"
    
    response=$(curl -s -w "\n%{http_code}" "$BASE_URL/health")
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✅ Health check passed${NC}"
        echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
    else
        echo -e "${RED}❌ Health check failed${NC}"
        return 1
    fi
}

# Function to test languages endpoint
test_languages() {
    echo -e "\n${YELLOW}Testing Languages Endpoint${NC}"
    
    response=$(curl -s -w "\n%{http_code}" "$BASE_URL/languages")
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✅ Languages endpoint working${NC}"
        echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
    else
        echo -e "${RED}❌ Languages endpoint failed${NC}"
        return 1
    fi
}

# Function to create nudge test data
create_nudge_test_data() {
    cat > sample_nudge_edumentor.json << 'EOF'
{
  "user_id": "nudge_test_student",
  "context": "edumentor",
  "language": "en",
  "user_data": {
    "average_score": 45,
    "missed_quizzes": 3,
    "engagement_score": 30,
    "streak_days": 0,
    "subject_area": "Mathematics"
  },
  "historical_data": {
    "previous_scores": [50, 48, 45, 42],
    "trend": "declining"
  },
  "preferences": {
    "notification_time": "evening",
    "tone_preference": "encouraging"
  }
}
EOF

    cat > sample_nudge_wellness.json << 'EOF'
{
  "user_id": "nudge_test_wellness",
  "context": "wellness",
  "language": "hi",
  "user_data": {
    "spending_ratio": 1.2,
    "stress_level": 80,
    "savings_rate": 3,
    "score": 40
  },
  "historical_data": {
    "spending_trend": "increasing",
    "stress_trend": "high"
  },
  "preferences": {
    "notification_frequency": "daily",
    "language_preference": "hi"
  }
}
EOF
}

# Main test execution
main() {
    echo -e "${BLUE}Starting API tests...${NC}"
    
    # Check if jq is available for JSON formatting
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}⚠️  jq not found. JSON responses will not be formatted.${NC}"
    fi
    
    # Check server
    check_server
    
    # Test basic endpoints
    test_health
    test_languages
    
    # Test report generation
    echo -e "\n${BLUE}=== REPORT GENERATION TESTS ===${NC}"
    
    test_endpoint "/generate-report" "sample_input_edumentor.json" "Edumentor Progress Report (Hindi)"
    test_endpoint "/generate-report" "sample_input_wellness.json" "Wellness Financial Health Report (English)"
    
    # Create and test nudge generation
    echo -e "\n${BLUE}=== NUDGE GENERATION TESTS ===${NC}"
    
    create_nudge_test_data
    
    test_endpoint "/generate-nudge" "sample_nudge_edumentor.json" "Edumentor Nudges (English)"
    test_endpoint "/generate-nudge" "sample_nudge_wellness.json" "Wellness Nudges (Hindi)"
    
    # Test error cases
    echo -e "\n${BLUE}=== ERROR HANDLING TESTS ===${NC}"
    
    # Test malformed JSON
    echo -e "\n${YELLOW}Testing malformed JSON handling${NC}"
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "$CONTENT_TYPE" \
        -d '{"invalid": json}' \
        "$BASE_URL/generate-report")
    
    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" -eq 400 ]; then
        echo -e "${GREEN}✅ Malformed JSON properly rejected${NC}"
    else
        echo -e "${RED}❌ Malformed JSON not handled correctly${NC}"
    fi
    
    # Test missing fields
    echo -e "\n${YELLOW}Testing missing required fields${NC}"
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "$CONTENT_TYPE" \
        -d '{"user_id": "test"}' \
        "$BASE_URL/generate-report")
    
    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" -eq 400 ]; then
        echo -e "${GREEN}✅ Missing fields properly rejected${NC}"
    else
        echo -e "${RED}❌ Missing fields not handled correctly${NC}"
    fi
    
    # Cleanup
    rm -f sample_nudge_edumentor.json sample_nudge_wellness.json
    
    echo -e "\n${GREEN}🎉 All tests completed!${NC}"
    echo -e "${BLUE}API is ready for integration with Edumentor and Wellness Bot${NC}"
}

# Help function
show_help() {
    echo "Karthikeya API Test Suite"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -u, --url URL  Set base URL (default: http://localhost:5000)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all tests with default URL"
    echo "  $0 -u http://localhost:8080  # Run tests against different port"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -u|--url)
            BASE_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main
