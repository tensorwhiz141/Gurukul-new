# 🎓 Gurukul Learning Platform

A comprehensive AI-powered educational platform with advanced features including agent simulation, financial forecasting, multilingual support, and intelligent tutoring systems.

## 🏗️ Architecture Overview

Gurukul is built as a microservices architecture with the following components:

### Backend Services
- **Base Backend (Port 8000)** - Main API with orchestration integration
- **API Data Service (Port 8001)** - Subject/Lecture data, RAG, LLM processing
- **Financial Simulator (Port 8002)** - Financial forecasting and simulation
- **Memory Management (Port 8003)** - User memory and session management
- **Augmed Kamal (Port 8004)** - Authentication & Memory Gateway
- **Subject Generation (Port 8005)** - Subject content generation

### Frontend
- **React Application (Port 3000/5174)** - Modern UI with Vite
- **Agent Simulation Dashboard** - Real-time agent visualization
- **Forecasting Dashboard** - Financial and data forecasting
- **Educational Interface** - Lesson generation and learning tools

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB
- Git

### 1. Clone Repository
```bash
git clone <repository-url>
cd "Gurukul Front and Back ass"
```

### 2. Start Backend Services
```bash
cd Backend
start_all_services.bat
```

### 3. Start Frontend
```bash
cd "new frontend"
start_frontend.bat
```

### 4. Access Application
- **Frontend**: http://localhost:3000 or http://localhost:5174
- **API Documentation**: http://localhost:8000/docs
- **Financial Simulator**: http://localhost:8002
- **Memory Management**: http://localhost:8003

## 📋 Features

### 🤖 AI Agent Simulation
- **FinancialCrew**: Blue theme with finance icon
- **EduMentor**: Green theme with book icon  
- **WellnessBot**: Orange/purple theme with heart icon
- Real-time agent decision visualization
- Interactive timeline of user-agent exchanges
- Dynamic feedback indicators

### 📊 Financial Forecasting
- Prophet & ARIMA time series forecasting
- Interactive forecasting dashboards
- Risk assessment and trend analysis
- Multi-agent financial planning workflow

### 🎓 Educational Features
- AI-powered lesson generation
- Multilingual support (Karthikeya module)
- RAG-based content retrieval
- Personalized learning paths
- Quiz generation and assessment

### 🔐 Authentication & Security
- Supabase JWT authentication
- Unified user session management
- API key protection for sensitive endpoints
- CORS configuration for cross-origin requests

### 💾 Data Management
- MongoDB integration across services
- Unified memory management
- User progress tracking
- Chat history persistence

## 🛠️ Development

### Backend Development
```bash
cd Backend
pip install -r requirements.txt
python check_dependencies.py
```

### Frontend Development
```bash
cd "new frontend"
npm install
npm run dev
```

### Testing
```bash
# Backend tests
cd Backend
python test_api.py

# Frontend tests
cd "new frontend"
npm test
```

## 📁 Project Structure

```
Gurukul Front and Back ass/
├── Backend/
│   ├── Base_backend/          # Main API service
│   ├── api_data/              # Data processing service
│   ├── Financial_simulator/   # Financial forecasting
│   ├── memory_management/     # User memory service
│   ├── augmed kamal/          # Auth & memory gateway
│   ├── Karthikeya/           # Multilingual tutoring
│   ├── orchestration/        # Service orchestration
│   ├── dedicated_chatbot_service/ # TTS and chat
│   └── start_all_services.bat # Service startup script
├── new frontend/
│   ├── src/
│   │   ├── pages/            # React pages
│   │   ├── components/       # Reusable components
│   │   ├── api/              # API integration
│   │   └── styles/           # CSS and styling
│   ├── public/               # Static assets
│   └── package.json          # Dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Environment Variables
Create `.env` files in respective directories:

**Backend/.env**
```
MONGODB_URI=mongodb://localhost:27017/gurukul
GROQ_API_KEY=your_groq_key
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key
```

**new frontend/.env**
```
VITE_API_BASE_URL=http://localhost:8000
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_key
```

### Database Setup
1. Install MongoDB
2. Create databases: `gurukul`, `agent_memory`
3. Run setup scripts in `new frontend/` directory

## 🚨 Troubleshooting

### Common Issues

**Port Conflicts**
- Check if ports 8000-8005 are available
- Modify port configurations in service files
- Use `netstat -an | findstr :8000` to check port usage

**Database Connection**
- Ensure MongoDB is running (`mongod` command)
- Check connection strings in configuration files
- Verify database permissions and authentication

**Dependencies**
- Run `pip install -r requirements.txt` for Python
- Run `npm install` for Node.js dependencies
- Check Python version compatibility (3.8+)
- Ensure Node.js version is 16+

**Service Startup**
- Check logs in respective service directories
- Ensure all environment variables are set
- Verify file permissions for batch scripts
- Run services individually to isolate issues

**Authentication Issues**
- Verify Supabase configuration
- Check JWT token validity
- Ensure CORS settings are correct

### Service Health Check
After startup, verify services:
- http://localhost:8000/health - Base Backend
- http://localhost:8001/health - API Data Service
- http://localhost:8002/health - Financial Simulator
- http://localhost:8003/health - Memory Management
- http://localhost:8004/health - Augmed Kamal

### Debug Mode
Enable debug logging by setting:
```bash
export DEBUG=true
export LOG_LEVEL=debug
```

## 📚 API Documentation

### Main Endpoints

**Educational Services**
- `GET /subjects` - Get available subjects
- `POST /generate-lesson` - Generate AI lesson content
- `GET /lectures/{subject_id}` - Get lectures for subject
- `POST /quiz/generate` - Generate quiz questions
- `GET /user/progress` - Get user learning progress

**AI Agent Services**
- `POST /chat` - Chat with AI agents
- `POST /agent/simulate` - Run agent simulation
- `GET /agent/history` - Get agent interaction history
- `POST /agent/feedback` - Submit agent feedback

**Financial Services**
- `POST /simulate` - Financial simulation
- `GET /forecast` - Get forecasting data
- `POST /forecast/generate` - Generate new forecast
- `GET /financial/history` - Get simulation history

**Memory & User Services**
- `POST /memory/store` - Store user memory
- `GET /memory/retrieve` - Retrieve user memory
- `POST /user/session` - Create user session
- `GET /user/profile` - Get user profile

### Authentication
Most endpoints require Bearer token:
```bash
Authorization: Bearer <supabase_jwt_token>
Content-Type: application/json
```

### Response Format
All APIs return JSON in this format:
```json
{
  "success": true,
  "data": {...},
  "message": "Operation successful",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section above
- Review service logs in respective directories
- Ensure all dependencies are properly installed
- Verify environment variables are correctly set

## 🎯 Roadmap

### Phase 1 (Current)
- [x] Core AI agent simulation
- [x] Financial forecasting dashboard
- [x] Educational content generation
- [x] Multilingual support
- [x] User authentication system

### Phase 2 (Next Quarter)
- [ ] Enhanced AI agent capabilities
- [ ] Advanced forecasting models (LSTM, Transformer)
- [ ] Real-time collaboration features
- [ ] Advanced analytics dashboard
- [ ] Performance optimization

### Phase 3 (Future)
- [ ] Mobile application (React Native)
- [ ] Multi-tenant support
- [ ] Advanced personalization engine
- [ ] Integration with external LMS
- [ ] Blockchain-based certification

## 🏆 Key Features Highlights

### 🤖 Advanced AI Integration
- **Multi-Agent System**: Financial, Educational, and Wellness agents
- **RAG Implementation**: Retrieval-Augmented Generation for accurate responses
- **Memory Management**: Persistent conversation context
- **Real-time Processing**: Live agent decision visualization

### 📊 Forecasting Capabilities
- **Prophet & ARIMA Models**: Industry-standard time series forecasting
- **Interactive Dashboards**: Real-time data visualization
- **Risk Assessment**: Automated risk analysis and recommendations
- **Multi-metric Support**: Probability, load, and general forecasting

### 🎓 Educational Excellence
- **Personalized Learning**: AI-driven content adaptation
- **Multilingual Support**: Content in multiple languages
- **Progress Tracking**: Comprehensive learning analytics
- **Interactive Assessments**: Dynamic quiz generation

### 🔒 Enterprise Security
- **JWT Authentication**: Secure token-based authentication
- **Role-based Access**: Granular permission system
- **Data Encryption**: End-to-end data protection
- **Audit Logging**: Comprehensive activity tracking

## 📞 Contact & Support

### Development Team
- **Backend Development**: Python/FastAPI specialists
- **Frontend Development**: React/TypeScript experts
- **AI/ML Engineering**: Machine learning and NLP specialists
- **DevOps**: Infrastructure and deployment automation

### Getting Help
1. **Documentation**: Check this README and inline code comments
2. **Issues**: Create GitHub issues for bugs and feature requests
3. **Discussions**: Use GitHub discussions for questions
4. **Email**: Contact the development team directly

### Contributing Guidelines
1. Follow PEP 8 for Python code
2. Use ESLint/Prettier for JavaScript/TypeScript
3. Write comprehensive tests for new features
4. Update documentation for API changes
5. Follow semantic versioning for releases

---

**Built with ❤️ for modern education and AI-powered learning**
