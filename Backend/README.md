# Gurukul Backend Services

Backend microservices for the Gurukul Learning Platform.

## Services Overview

- **Base_backend/** - Main API service (Port 8000)
- **api_data/** - Data processing service (Port 8001)
- **Financial_simulator/** - Financial forecasting (Port 8002)
- **memory_management/** - User memory service (Port 8003)
- **augmed kamal/** - Auth & memory gateway (Port 8004)
- **Karthikeya/** - Multilingual tutoring (Port 8001)
- **orchestration/** - Service orchestration
- **dedicated_chatbot_service/** - TTS and chat services
- **tts_service/** - Text-to-speech service

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start all services
start_all_services.bat

# Or start individual services
cd Base_backend && python main.py
```

## Tech Stack

- **Python 3.8+** - Primary language
- **FastAPI** - Web framework
- **MongoDB** - Database
- **Supabase** - Authentication
- **LangChain** - LLM integration
- **Prophet/ARIMA** - Forecasting models