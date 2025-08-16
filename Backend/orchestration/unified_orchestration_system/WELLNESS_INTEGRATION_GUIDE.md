# Wellness Bot Integration Guide

This guide explains how to set up and use the Wellness Bot integration between the Gurukul frontend and the unified orchestration system backend.

## 🎯 Overview

The Wellness Bot provides comprehensive wellness support through two specialized modes:
- **Emotional Wellness**: Mood tracking, stress management, emotional support
- **Financial Wellness**: Financial guidance, budgeting advice, financial stress management

## 🚀 Quick Setup

### 1. Start the Wellness API

**Option A: Windows (Recommended)**
```bash
cd "Backend/orchestration/unified_orchestration_system"
start_wellness_api.bat
```

**Option B: Cross-platform**
```bash
cd Backend/orchestration/unified_orchestration_system
python start_wellness_api.py
```

**Option C: Manual**
```bash
cd Backend/orchestration/unified_orchestration_system
pip install -r requirements.txt
python orchestration_api.py
```

### 2. Configure API Keys

1. Create/edit `.env` file in the `unified_orchestration_system` directory:
```env
GEMINI_API_KEY=your_primary_key_here
GEMINI_API_KEY_BACKUP=your_backup_key_here
```

2. Get API keys from: https://makersuite.google.com/app/apikey

### 3. Verify Connection

1. Open the Gurukul frontend
2. Navigate to Agent Simulator
3. Select the WellnessBot agent
4. Check for "API Connected" status indicator

## 🔧 API Endpoints

The system supports multiple endpoint formats for maximum compatibility:

### Full Orchestration System
- **Endpoint**: `POST /ask-wellness`
- **Features**: Complete wellness support with triggers, emotional nudges, and sub-agent integration
- **Response**: Structured wellness advice with emotional support

### Simple API (Fallback)
- **Endpoint**: `POST /wellness`
- **Features**: Basic wellness advice
- **Response**: Simple text response

## 💡 Features

### Emotional Wellness
- **Mood Tracking**: 1-10 scale mood scoring
- **Stress Monitoring**: 0-6 stress level tracking with descriptive labels
- **Emotional Support**: Encouragement, affirmations, and mindfulness tips
- **Trigger Detection**: Automatic intervention for concerning patterns

### Financial Wellness
- **Financial Guidance**: Budgeting advice and financial planning
- **Stress Management**: Financial stress detection and support
- **Practical Steps**: Actionable financial recommendations
- **Resource Suggestions**: Financial tools and strategies

### Smart Features
- **Memory Management**: Tracks user sessions and interaction history
- **Trigger System**: Automatic intervention detection
- **Sub-agent Integration**: Connects with specialized wellness bots
- **Fallback Support**: Works even when advanced features are unavailable

## 🎨 User Interface

### Wellness Controls
- **Type Toggle**: Switch between Emotional and Financial wellness
- **Mood Slider**: Interactive 1-10 mood rating
- **Stress Slider**: 0-6 stress level with descriptive labels
- **Query Input**: Contextual placeholder text for different wellness types

### Response Display
- **Structured Advice**: Main guidance with action steps and tips
- **Emotional Support**: Encouragement and mindfulness techniques
- **Wellness Alerts**: Highlighted triggers and interventions
- **History Tracking**: Recent wellness sessions with metrics

### Status Indicators
- **🟢 API Connected**: Backend is running and responsive
- **🔴 API Offline**: Backend needs to be started
- **🟡 Checking...**: Connection status being verified

## 🛠️ Troubleshooting

### Common Issues

**1. "API not available" error**
- Ensure the backend server is running on port 8000
- Check that no other service is using port 8000
- Verify the .env file has valid API keys

**2. "Failed to fetch" error**
- Check if the backend server is accessible at http://localhost:8000
- Verify CORS is properly configured
- Ensure firewall isn't blocking the connection

**3. Empty or generic responses**
- Check Gemini API key validity
- Verify internet connection for API calls
- Check server logs for detailed error messages

### Server Logs
Monitor the backend console for detailed error messages and API call logs.

### Testing Connection
Use the "Check Connection" button in the wellness interface to verify API status.

## 📁 File Structure

```
Backend/orchestration/unified_orchestration_system/
├── orchestration_api.py          # Full orchestration system
├── simple_api.py                 # Simplified fallback API
├── start_wellness_api.py         # Cross-platform startup script
├── start_wellness_api.bat        # Windows startup script
├── requirements.txt              # Python dependencies
├── .env                          # API keys configuration
└── data/                         # Wellness knowledge base
```

## 🔗 Integration Points

### Frontend Components
- **AgentSimulator.jsx**: Main wellness interface
- **Wellness Message Rendering**: Specialized chat display
- **API Status Checking**: Connection monitoring
- **Error Handling**: User-friendly error messages

### Backend Components
- **Wellness Triggers**: Automatic intervention detection
- **Memory Management**: User session tracking
- **Sub-agent Integration**: Specialized wellness bots
- **Vector Stores**: Wellness knowledge retrieval

## 📊 Data Flow

1. **User Input**: Query entered in wellness interface
2. **API Call**: Frontend sends request to backend
3. **Processing**: Backend analyzes query and retrieves relevant knowledge
4. **Response Generation**: AI generates personalized wellness advice
5. **Display**: Frontend renders structured response in chat
6. **History**: Interaction saved for future reference

## 🎯 Best Practices

1. **Regular Monitoring**: Check API status when using wellness features
2. **Meaningful Queries**: Provide specific wellness concerns for better advice
3. **Mood Tracking**: Use mood and stress sliders for emotional wellness
4. **Follow-up**: Review wellness history to track progress
5. **Professional Help**: Seek professional support for serious concerns

## 🔄 Updates and Maintenance

- **API Keys**: Rotate Gemini API keys periodically
- **Dependencies**: Update Python packages regularly
- **Logs**: Monitor server logs for performance issues
- **Backup**: Keep backup API keys configured

For additional support, check the main README.md file or server logs for detailed error information.
