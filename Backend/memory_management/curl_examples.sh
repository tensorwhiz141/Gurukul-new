#!/bin/bash

# Memory Management API - cURL Examples
# Make sure the API server is running on http://localhost:8003

# Configuration
API_BASE_URL="http://localhost:8003"
API_KEY="memory_api_key_dev"  # Replace with your actual API key

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Memory Management API - cURL Examples${NC}"
echo "======================================"

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}$1${NC}"
    echo "$(printf '=%.0s' {1..50})"
}

# Function to execute curl with error handling
execute_curl() {
    local description="$1"
    shift
    echo -e "\n${GREEN}$description${NC}"
    echo "Command: curl $*"
    echo "Response:"
    curl -s -w "\nHTTP Status: %{http_code}\n" "$@" | jq . 2>/dev/null || cat
    echo ""
}

# 1. Health Check
print_section "1. Health Check"

execute_curl "Check API health and database connectivity" \
    -X GET "$API_BASE_URL/memory/health"

# 2. Memory Storage
print_section "2. Memory Storage"

execute_curl "Store a user preference memory" \
    -X POST "$API_BASE_URL/memory" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "user_id": "user123",
        "persona_id": "financial_advisor",
        "content": "User prefers conservative investment strategies with low risk tolerance",
        "content_type": "preference",
        "metadata": {
            "tags": ["investment", "conservative", "low-risk"],
            "importance": 8,
            "topic": "investment_strategy",
            "context_type": "user_preference",
            "source": "user_input",
            "confidence": 0.95
        }
    }'

execute_curl "Store a factual memory" \
    -X POST "$API_BASE_URL/memory" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "user_id": "user123",
        "persona_id": "financial_advisor",
        "content": "User has monthly income of $5000 and wants to save $1000 per month",
        "content_type": "fact",
        "metadata": {
            "tags": ["income", "savings", "budget"],
            "importance": 9,
            "topic": "financial_profile",
            "source": "user_input"
        }
    }'

execute_curl "Store an interaction" \
    -X POST "$API_BASE_URL/memory/interaction" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "user_id": "user123",
        "persona_id": "financial_advisor",
        "user_message": "What is the best investment strategy for someone my age?",
        "agent_response": "Based on your conservative preference and 30-year timeline, I recommend a balanced portfolio with 60% bonds and 40% stocks.",
        "context": {
            "session_id": "session_001",
            "conversation_turn": 1,
            "domain": "finance",
            "intent": "investment_advice",
            "previous_context": "User just completed risk assessment"
        },
        "metadata": {
            "response_time": 1.5,
            "confidence": 0.92,
            "model_used": "financial_advisor_v1",
            "tags": ["investment", "advice"]
        }
    }'

# 3. Memory Retrieval
print_section "3. Memory Retrieval"

execute_curl "Get all memories for a persona" \
    -X GET "$API_BASE_URL/memory?persona=financial_advisor&limit=10" \
    -H "Authorization: Bearer $API_KEY"

execute_curl "Get memories for specific user and persona" \
    -X GET "$API_BASE_URL/memory?persona=financial_advisor&user_id=user123&limit=5" \
    -H "Authorization: Bearer $API_KEY"

execute_curl "Get memories filtered by content type" \
    -X GET "$API_BASE_URL/memory?persona=financial_advisor&content_type=preference&content_type=fact" \
    -H "Authorization: Bearer $API_KEY"

execute_curl "Get memories with minimum importance level" \
    -X GET "$API_BASE_URL/memory?persona=financial_advisor&min_importance=7" \
    -H "Authorization: Bearer $API_KEY"

execute_curl "Get recent interactions for chain-of-thought" \
    -X GET "$API_BASE_URL/memory?limit=5&recent_interactions=true" \
    -H "Authorization: Bearer $API_KEY"

execute_curl "Get recent interactions for specific user" \
    -X GET "$API_BASE_URL/memory?limit=3&recent_interactions=true&user_id=user123" \
    -H "Authorization: Bearer $API_KEY"

# 4. Memory Search
print_section "4. Memory Search"

execute_curl "Search memories by text query" \
    -X GET "$API_BASE_URL/memory/search?query=investment%20strategy&limit=5" \
    -H "Authorization: Bearer $API_KEY"

execute_curl "Search memories for specific persona" \
    -X GET "$API_BASE_URL/memory/search?query=conservative&persona_id=financial_advisor" \
    -H "Authorization: Bearer $API_KEY"

execute_curl "Search with content type filter" \
    -X GET "$API_BASE_URL/memory/search?query=user&content_type=preference&limit=3" \
    -H "Authorization: Bearer $API_KEY"

# 5. Persona Memory Summary
print_section "5. Persona Memory Summary"

execute_curl "Get persona memory summary" \
    -X GET "$API_BASE_URL/memory/persona/financial_advisor/summary" \
    -H "Authorization: Bearer $API_KEY"

execute_curl "Get persona summary for specific user" \
    -X GET "$API_BASE_URL/memory/persona/financial_advisor/summary?user_id=user123" \
    -H "Authorization: Bearer $API_KEY"

# 6. Individual Memory Operations
print_section "6. Individual Memory Operations"

# Note: You'll need to replace MEMORY_ID with an actual memory ID from previous responses
echo -e "${YELLOW}Note: Replace 'MEMORY_ID' with actual memory ID from previous responses${NC}"

execute_curl "Get specific memory by ID" \
    -X GET "$API_BASE_URL/memory/MEMORY_ID" \
    -H "Authorization: Bearer $API_KEY"

execute_curl "Update memory content" \
    -X PUT "$API_BASE_URL/memory/MEMORY_ID" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "content": "Updated: User strongly prefers conservative investment strategies",
        "metadata": {
            "tags": ["investment", "conservative", "updated"],
            "importance": 9,
            "topic": "investment_strategy"
        }
    }'

execute_curl "Soft delete memory (deactivate)" \
    -X DELETE "$API_BASE_URL/memory/MEMORY_ID" \
    -H "Authorization: Bearer $API_KEY"

execute_curl "Hard delete memory (permanent)" \
    -X DELETE "$API_BASE_URL/memory/MEMORY_ID?hard_delete=true" \
    -H "Authorization: Bearer $API_KEY"

# 7. Error Examples
print_section "7. Error Examples"

execute_curl "Request without authentication (should return 401)" \
    -X GET "$API_BASE_URL/memory?persona=test"

execute_curl "Request with invalid API key (should return 401)" \
    -X GET "$API_BASE_URL/memory?persona=test" \
    -H "Authorization: Bearer invalid_key"

execute_curl "Invalid request - missing required fields (should return 422)" \
    -X POST "$API_BASE_URL/memory" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "user_id": "user123"
    }'

execute_curl "Get non-existent memory (should return 404)" \
    -X GET "$API_BASE_URL/memory/non-existent-id" \
    -H "Authorization: Bearer $API_KEY"

# 8. Bulk Operations Example
print_section "8. Bulk Operations Example"

echo -e "${GREEN}Creating multiple memories for demonstration${NC}"

# Create multiple memories
for i in {1..3}; do
    execute_curl "Create memory $i" \
        -X POST "$API_BASE_URL/memory" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"user_id\": \"user123\",
            \"persona_id\": \"financial_advisor\",
            \"content\": \"Test memory $i for bulk operations\",
            \"content_type\": \"text\",
            \"metadata\": {
                \"tags\": [\"test\", \"bulk\", \"memory$i\"],
                \"importance\": $((i + 3)),
                \"topic\": \"testing\"
            }
        }"
done

# Retrieve all test memories
execute_curl "Retrieve all memories for persona (should include bulk test memories)" \
    -X GET "$API_BASE_URL/memory?persona=financial_advisor&limit=20" \
    -H "Authorization: Bearer $API_KEY"

print_section "Examples Complete"

echo -e "${GREEN}All cURL examples completed!${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Check the API documentation: $API_BASE_URL/memory/docs"
echo "2. Try the interactive API explorer: $API_BASE_URL/memory/redoc"
echo "3. Replace MEMORY_ID placeholders with actual IDs from responses"
echo "4. Modify the examples for your specific use cases"
echo ""
echo -e "${YELLOW}Tips:${NC}"
echo "- Use 'jq' for better JSON formatting: curl ... | jq ."
echo "- Save memory IDs from responses for update/delete operations"
echo "- Check HTTP status codes for error handling"
echo "- Monitor rate limits in response headers"
