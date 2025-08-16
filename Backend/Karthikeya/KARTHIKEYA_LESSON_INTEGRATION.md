# Karthikeya Lesson Integration - Multilingual Support

## Overview

The Karthikeya system has been enhanced to support the new standardized lesson format with comprehensive multilingual capabilities. This integration provides educational content generation in multiple Indian languages while maintaining the new lesson structure.

## New Lesson Format Integration

### Standardized Format
```json
{
  "title": "Introduction to Dharma",
  "level": "Seed",
  "text": "...",
  "quiz": [ {...} ],
  "tts": true,
  "language": "hi",
  "metadata": {
    "multilingual_support": true,
    "lesson_format_version": "2.0"
  }
}
```

### Supported Languages
- **English (en)**: Default language
- **Hindi (hi)**: हिंदी
- **Bengali (bn)**: বাংলা
- **Gujarati (gu)**: ગુજરાતી
- **Marathi (mr)**: मराठी
- **Tamil (ta)**: தமிழ்
- **Telugu (te)**: తెలుగు
- **Kannada (kn)**: ಕನ್ನಡ

## API Endpoints

### POST /generate-lesson

Generate a multilingual lesson in the new standardized format.

**Request:**
```json
{
  "subject": "Mathematics",
  "topic": "Algebra", 
  "language": "hi",
  "user_id": "user123",
  "level": "Seed"
}
```

**Response:**
```json
{
  "title": "बीजगणित का परिचय",
  "level": "Seed",
  "text": "बीजगणित गणित की एक मौलिक शाखा है...",
  "quiz": [
    {
      "question": "बीजगणित क्या है?",
      "options": ["गणित की शाखा", "विज्ञान", "कला", "भाषा"],
      "correct": 0,
      "explanation": "बीजगणित गणित की एक शाखा है..."
    }
  ],
  "tts": true,
  "language": "hi",
  "sentiment": "positive",
  "nudges": ["Continue learning!", "Great progress!"],
  "metadata": {
    "multilingual_support": true,
    "lesson_format_version": "2.0",
    "supported_languages": ["en", "hi", "bn", "gu", "mr", "ta", "te", "kn"]
  }
}
```

## Implementation Details

### 1. Enhanced API Endpoints (`nudge_engine/api_endpoints.py`)

Added new `/generate-lesson` endpoint that:
- Accepts multilingual lesson requests
- Validates input parameters
- Generates content using the report generator
- Returns lessons in the new standardized format

### 2. Lesson Report Generation

**Function:** `generate_lesson_report()`
- Creates educational content using the existing report infrastructure
- Integrates sentiment analysis for learning motivation
- Supports all configured languages
- Maintains compatibility with existing nudge system

### 3. Multilingual Content Generation

**Functions:**
- `generate_multilingual_lesson_text()`: Creates comprehensive lesson content
- `generate_multilingual_quiz()`: Generates quiz questions with explanations
- `get_localized_title()`: Provides localized lesson titles

### 4. Template Integration (`templates/report_templates.json`)

Added `lesson_content` templates for:
- Lesson structure and content
- Learning objectives
- Assessment guidelines
- Language-specific formatting

### 5. Format Transformation

**Function:** `transform_report_to_lesson_format()`
- Converts report data to new lesson format
- Preserves multilingual metadata
- Integrates sentiment and nudge data
- Maintains backward compatibility

## Language-Specific Features

### Content Localization
- **Titles**: Fully localized lesson titles
- **Content**: Native language explanations and examples
- **Quiz Questions**: Culturally appropriate questions and answers
- **Learning Objectives**: Language-specific learning goals

### Cultural Adaptation
- **Examples**: Relevant to local context
- **Terminology**: Appropriate academic vocabulary
- **Structure**: Follows language-specific educational patterns

## Integration with Existing Systems

### Report Generator Integration
- Uses existing `ReportGenerator` infrastructure
- Leverages sentiment analysis capabilities
- Maintains nudge system compatibility
- Preserves user tracking and analytics

### Template System Integration
- Extends existing template structure
- Supports dynamic content generation
- Maintains consistency across languages
- Enables easy template updates

### Sentiment Analysis Integration
- Analyzes learning progress sentiment
- Provides motivational nudges
- Supports language-specific sentiment mapping
- Enhances user engagement

## Usage Examples

### Basic Lesson Generation
```python
# English lesson
response = requests.post("http://localhost:5000/generate-lesson", json={
    "subject": "Science",
    "topic": "Physics",
    "language": "en",
    "user_id": "student123"
})

# Hindi lesson
response = requests.post("http://localhost:5000/generate-lesson", json={
    "subject": "विज्ञान",
    "topic": "भौतिकी", 
    "language": "hi",
    "user_id": "student123"
})
```

### Advanced Configuration
```python
response = requests.post("http://localhost:5000/generate-lesson", json={
    "subject": "Mathematics",
    "topic": "Calculus",
    "language": "bn",
    "user_id": "advanced_student",
    "level": "Advanced"
})
```

## Testing

### Test Script: `test_lesson_integration.py`

Comprehensive test suite that validates:
- ✅ Multilingual lesson generation
- ✅ New format compliance
- ✅ API endpoint functionality
- ✅ Direct integration testing
- ✅ Format compatibility

### Running Tests
```bash
cd Backend/Karthikeya
python test_lesson_integration.py
```

## Benefits

### 1. Multilingual Education
- Native language learning support
- Cultural context preservation
- Improved comprehension for non-English speakers
- Inclusive educational experience

### 2. Standardized Format
- Consistent lesson structure across languages
- Easy integration with frontend systems
- Scalable content management
- Enhanced user experience

### 3. Advanced Features
- TTS-ready content for accessibility
- Interactive quiz generation
- Sentiment-based learning motivation
- Progress tracking and analytics

### 4. System Integration
- Seamless integration with existing Gurukul infrastructure
- Backward compatibility with current systems
- Extensible architecture for future enhancements
- Robust error handling and validation

## Future Enhancements

### 1. Additional Languages
- Support for more regional languages
- Dialect-specific content generation
- Voice-based content delivery

### 2. Advanced Content Types
- Video lesson integration
- Interactive simulations
- Gamified learning experiences

### 3. AI-Powered Personalization
- Adaptive content difficulty
- Learning style optimization
- Personalized quiz generation

### 4. Analytics and Insights
- Learning pattern analysis
- Performance prediction
- Recommendation systems

## Configuration

### Environment Setup
1. Ensure Karthikeya service is running on port 5000
2. Verify all template files are present
3. Check language configuration files
4. Test multilingual font support

### Dependencies
- Flask for API endpoints
- Existing Karthikeya infrastructure
- Template and configuration systems
- Sentiment analysis components

## Troubleshooting

### Common Issues
1. **Missing Templates**: Ensure `lesson_content` templates are present
2. **Language Support**: Verify language codes in configuration
3. **Format Validation**: Check required fields in responses
4. **Service Connectivity**: Confirm Karthikeya service is accessible

### Debug Mode
Enable debug logging to troubleshoot issues:
```python
app.run(debug=True, host="0.0.0.0", port=5000)
```

This integration provides a robust foundation for multilingual educational content generation while maintaining the new standardized lesson format and leveraging the existing Karthikeya infrastructure.
