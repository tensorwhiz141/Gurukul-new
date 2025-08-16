# BHIV Core PowerShell Testing Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🧪 BHIV Core PowerShell Testing Suite" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

$API_URL = "http://localhost:8010"

Write-Host ""
Write-Host "🔍 Testing BHIV Core API..." -ForegroundColor Green
Write-Host "API URL: $API_URL" -ForegroundColor Cyan
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test 1: Health Check" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$API_URL/health" -Method GET
    Write-Host "✅ Health Check Success" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ Health Check Failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test 2: Root Endpoint" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$API_URL/" -Method GET
    Write-Host "✅ Root Endpoint Success" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ Root Endpoint Failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test 3: System Statistics" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$API_URL/api/v1/bhiv-core/stats" -Method GET
    Write-Host "✅ System Stats Success" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ System Stats Failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test 4: Query Classification" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# Test Educational Query
Write-Host "Testing Educational Query..." -ForegroundColor Cyan
$eduQuery = @{
    user_id = "test_user_001"
    query = "How do I solve quadratic equations?"
    lang = "en-IN"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_URL/api/v1/bhiv-core/classify" -Method POST -Body $eduQuery -ContentType "application/json"
    Write-Host "✅ Educational Query Success" -ForegroundColor Green
    Write-Host "   🎯 Classification: $($response.classification)" -ForegroundColor Cyan
    Write-Host "   🤖 Agent: $($response.recommended_agent)" -ForegroundColor Cyan
    Write-Host "   📊 Confidence: $($response.confidence)" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Educational Query Failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test Wellness Query
Write-Host "Testing Wellness Query..." -ForegroundColor Cyan
$wellnessQuery = @{
    user_id = "test_user_002"
    query = "I am feeling very stressed about my exams"
    lang = "en-IN"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_URL/api/v1/bhiv-core/classify" -Method POST -Body $wellnessQuery -ContentType "application/json"
    Write-Host "✅ Wellness Query Success" -ForegroundColor Green
    Write-Host "   🎯 Classification: $($response.classification)" -ForegroundColor Cyan
    Write-Host "   🤖 Agent: $($response.recommended_agent)" -ForegroundColor Cyan
    Write-Host "   📊 Confidence: $($response.confidence)" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Wellness Query Failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test 5: Quick Query Processing" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# Test Simple Educational Query
Write-Host "Testing Simple Educational Query..." -ForegroundColor Cyan
$quickQuery = @{
    user_id = "student_123"
    query = "Explain machine learning basics"
    voice_enabled = $false
    lang = "en-IN"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_URL/api/v1/bhiv-core/quick-query" -Method POST -Body $quickQuery -ContentType "application/json"
    Write-Host "✅ Quick Query Success" -ForegroundColor Green
    Write-Host "   🤖 Agent: $($response.agent_used)" -ForegroundColor Cyan
    Write-Host "   📊 Confidence: $($response.confidence_score)" -ForegroundColor Cyan
    Write-Host "   ⏱️ Processing Time: $($response.processing_time_ms)ms" -ForegroundColor Cyan
    Write-Host "   📝 Response: $($response.response_text.Substring(0, [Math]::Min(100, $response.response_text.Length)))..." -ForegroundColor Cyan
} catch {
    Write-Host "❌ Quick Query Failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test 6: Full BHIV Processing" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

$fullRequest = @{
    session_id = "test-session-ps1"
    timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
    user_info = @{
        user_id = "ps1_test_user"
        auth_token = "bhiv-core-key"
        lang = "en-IN"
        persona = "student"
        permissions = @("read", "write")
    }
    input = @{
        raw_text = "Explain quantum computing applications"
        voice_enabled = $false
        mode = "chat"
    }
    knowledgebase_query = @{
        use_vector = $true
        top_k = 3
    }
    llm_request = @{
        model = "gemini"
        max_tokens = 512
    }
} | ConvertTo-Json -Depth 4

try {
    $response = Invoke-RestMethod -Uri "$API_URL/api/v1/bhiv-core/process" -Method POST -Body $fullRequest -ContentType "application/json"
    Write-Host "✅ Full BHIV Processing Success" -ForegroundColor Green
    Write-Host "   📋 Session ID: $($response.session_id)" -ForegroundColor Cyan
    Write-Host "   🤖 Agent: $($response.agent_directives.agent_name)" -ForegroundColor Cyan
    Write-Host "   🎯 Classification: $($response.agent_directives.classification)" -ForegroundColor Cyan
    Write-Host "   ⏱️ Processing Time: $($response.logs.processing_time_ms)ms" -ForegroundColor Cyan
    Write-Host "   🏆 Reward Score: $($response.logs.reward_score)" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Full BHIV Processing Failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🎯 Testing Complete!" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 To see formatted output, visit: $API_URL/docs" -ForegroundColor Green
Write-Host "📊 For detailed testing, run: python test_api_client.py" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to continue"
