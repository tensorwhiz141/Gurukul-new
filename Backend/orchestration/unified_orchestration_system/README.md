# Unified Orchestration System

A comprehensive orchestration system that manages three specialized AI agents with dynamic RAG-based responses, sub-agent integration, and intelligent intervention triggers.

## 🌟 Features

### Core Capabilities
- **Dynamic RAG Pipeline**: Uses Gemini API for intelligent, context-aware responses
- **Three Specialized Endpoints**: Vedas (spiritual wisdom), Wellness (health support), Edumentor (educational content)
- **Dual API Failover**: Automatic failover between primary and backup Gemini API keys
- **Agent Memory Management**: Persistent user sessions and interaction tracking across all agents
- **Orchestration Triggers**: Automatic intervention detection and sub-agent activation
- **Sub-Agent Integration**: Seamless integration with tutorbot, wellness bots, and quizbot

### Data Processing
- **Unified Data Ingestion**: Processes PDFs and CSVs with intelligent content categorization
- **Specialized Vector Stores**: Separate FAISS indexes for Vedas, Wellness, Educational, and Unified content
- **Content Categorization**: Automatic classification based on keywords and metadata
- **29,930+ Documents**: Comprehensive knowledge base with spiritual texts and educational content

## 🏗️ Architecture

```
unified_orchestration_system/
├── data_ingestion.py          # Comprehensive data processing and vector store creation
├── orchestration_api.py       # Main API with three endpoints and orchestration logic
├── test_system.py            # Comprehensive test suite
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration
├── README.md                # This documentation
├── data/                     # Data directory
│   ├── *.pdf                # Vedic texts (Gita, Upanishads, Vedas, Ramayana)
│   └── *.csv                # Educational content (Plant, Seed, Tree datasets)
└── vector_stores/           # Generated vector databases
    ├── vedas_index/         # Spiritual content (23,953 documents)
    ├── wellness_index/      # Health and wellness content
    ├── educational_index/   # Learning content (6,224 documents)
    └── unified_index/       # All content combined
```

## 🚀 Quick Start

### 1. Installation
```bash
cd unified_orchestration_system
pip install -r requirements.txt
```

### 2. Environment Setup
Configure your `.env` file with API keys:
```env
GEMINI_API_KEY=your_primary_key_here
GEMINI_API_KEY_BACKUP=your_backup_key_here
```

### 3. Data Ingestion
```bash
python data_ingestion.py
```
This will:
- Process all PDF and CSV files in the `data/` directory
- Create specialized vector stores for different content types
- Generate ingestion statistics and reports

### 4. Start the API Server
```bash
python orchestration_api.py
```
Server will be available at: `http://localhost:8000`

### 5. Test the System
```bash
python test_system.py
```

## 📡 API Endpoints

### Core Endpoints

#### POST `/ask-vedas`
Get spiritual wisdom from ancient Vedic texts
```json
{
  "query": "What is the purpose of meditation in Vedic tradition?",
  "user_id": "user123"
}
```

#### POST `/ask-wellness`
Get comprehensive wellness advice with emotional support
```json
{
  "query": "I'm feeling very stressed lately",
  "user_id": "user123",
  "mood_score": 3.0,
  "stress_level": 5.0
}
```

#### POST `/ask-edumentor`
Get educational content with interactive learning activities
```json
{
  "query": "How do plants convert sunlight into energy?",
  "user_id": "user123",
  "quiz_score": 75.0
}
```

### Management Endpoints

#### GET `/user-session/{user_id}`
Retrieve user session data and interaction history

#### POST `/trigger-check/{user_id}`
Manually trigger comprehensive wellness and educational checks

#### GET `/system-status`
Get comprehensive system status and component health

## 🎯 Orchestration Triggers

The system automatically detects when users need intervention:

### Educational Triggers
- **Low Quiz Score**: Score below 60% triggers tutoring intervention
- **Declining Performance**: Consistent low scores trigger intensive support
- **Inactivity**: No learning activity for 7+ days triggers engagement nudge

### Wellness Triggers
- **High Stress**: Stress level above 3 triggers immediate emotional support
- **Low Mood**: Mood score below 3 triggers wellness check
- **Persistent Issues**: Ongoing low mood patterns trigger comprehensive assessment

### Sub-Agent Integration
When triggers activate, the system automatically calls:
- **Tutorbot**: Personalized lesson planning and educational support
- **Emotional Wellness Bot**: Immediate emotional support and coping strategies
- **Financial Wellness Bot**: Financial guidance and budgeting advice
- **Quizbot**: Adaptive assessment and knowledge gap identification

## 🧠 Agent Memory Management

### User Session Tracking
- Persistent user profiles with preferences and history
- Interaction tracking across all agents
- Wellness metrics and educational progress monitoring
- Spiritual journey tracking for Vedas interactions

### Memory Features
- **Cross-Agent Continuity**: Shared memory across all three main agents
- **Trigger History**: Track intervention patterns and effectiveness
- **Personalization**: Responses adapt based on user history and preferences
- **Progress Monitoring**: Long-term tracking of wellness and educational metrics

## 🔄 Gemini API Failover

### Dual Key Support
- Primary API key for normal operations
- Automatic failover to backup key if primary fails
- Graceful degradation to fallback responses if both keys fail

### Reliability Features
- Connection testing on startup
- Automatic retry logic with exponential backoff
- Real-time API health monitoring
- Fallback to pre-defined responses when needed

## 📊 Response Formats

### Vedas Response
```json
{
  "query_id": "uuid",
  "query": "user question",
  "wisdom": {
    "core_teaching": "main spiritual principle",
    "practical_application": "modern life guidance",
    "philosophical_insight": "deeper meaning",
    "relevant_quote": "ancient text reference"
  },
  "source_documents": [...],
  "timestamp": "ISO format",
  "user_id": "user123"
}
```

### Wellness Response
```json
{
  "query_id": "uuid",
  "query": "user concern",
  "advice": {
    "main_advice": "primary guidance",
    "practical_steps": ["actionable steps"],
    "tips": ["wellness tips"]
  },
  "emotional_nudge": {
    "encouragement": "supportive message",
    "affirmation": "strength affirmation",
    "mindfulness_tip": "coping technique"
  },
  "triggers_detected": [...],
  "trigger_interventions": [...],
  "source_documents": [...],
  "timestamp": "ISO format",
  "user_id": "user123"
}
```

### Edumentor Response
```json
{
  "query_id": "uuid",
  "query": "learning question",
  "explanation": "comprehensive explanation",
  "activity": {
    "title": "activity name",
    "description": "learning objectives",
    "instructions": ["step by step"],
    "materials_needed": ["required materials"]
  },
  "triggers_detected": [...],
  "trigger_interventions": [...],
  "source_documents": [...],
  "timestamp": "ISO format",
  "user_id": "user123"
}
```

## 🧪 Testing

The system includes a comprehensive test suite that validates:
- All endpoint functionality
- Sub-agent integration
- Trigger mechanisms
- API failover
- Memory management
- Response quality

Run tests with:
```bash
python test_system.py
```

## 🔧 Configuration

### Environment Variables
```env
# API Keys
GEMINI_API_KEY=primary_key
GEMINI_API_KEY_BACKUP=backup_key

# System Settings
LOG_LEVEL=INFO
MAX_RETRIES=3
TIMEOUT_SECONDS=30

# Vector Store Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MAX_DOCUMENTS_PER_QUERY=5

# Trigger Thresholds
LOW_QUIZ_SCORE_THRESHOLD=60
WELLNESS_CONCERN_THRESHOLD=3
INACTIVITY_DAYS_THRESHOLD=7
```

## 📈 Performance

### Data Processing
- **29,930 total documents** processed and indexed
- **23,953 Vedas documents** from ancient spiritual texts
- **6,224 educational documents** from CSV datasets
- **Specialized vector stores** for optimal retrieval performance

### Response Times
- Typical response time: 2-5 seconds
- Vector search: <1 second
- LLM generation: 1-4 seconds
- Trigger processing: <500ms

## 🛠️ Production Deployment

### Requirements
- Python 3.8+
- 4GB+ RAM (for vector stores)
- 2GB+ disk space
- Stable internet connection for Gemini API

### Scaling Considerations
- Vector stores can be moved to dedicated vector databases
- API can be containerized with Docker
- Load balancing for multiple instances
- Database backend for user sessions

## 📚 Documentation

- **API Documentation**: Available at `/docs` when server is running
- **System Status**: Real-time status at `/system-status`
- **Test Results**: Generated in `test_results.json`
- **Ingestion Stats**: Saved in `vector_stores/ingestion_stats.json`

## 🤝 Integration

The system is designed to integrate with existing sub-agents:
- **Tutorbot**: Educational lesson planning
- **Emotional Wellness Bot**: Mental health support
- **Financial Wellness Bot**: Financial guidance
- **Quizbot**: Assessment and evaluation

Each sub-agent can be called independently or triggered automatically based on user behavior patterns.

---

**Built with ❤️ for comprehensive AI orchestration and user support**
