# Knowledge Base and Wikipedia Integration Fix

## Problem Description
The subject generation functionality was not properly handling the `include_wikipedia` and `use_knowledge_store` parameters. When users selected "search the knowledgebase", the system was returning Wikipedia data instead of content from the vector stores in the orchestration system.

## Root Cause
The GET endpoint `/generate_lesson` in `Backend/Base_backend/api.py` was ignoring the boolean parameters and only using a basic LLM service without integrating with:
1. The knowledge base vector stores located in `Backend/orchestration/unified_orchestration_system/vector_stores/`
2. The Wikipedia utilities from the subject generation module

## Solution Implemented

### 1. Enhanced GET Endpoint
Modified the `/generate_lesson` endpoint to:
- **Properly handle `use_knowledge_store` parameter**: When True, loads and searches the vector stores
- **Properly handle `include_wikipedia` parameter**: When True, fetches relevant Wikipedia content
- **Combine both sources**: When both are enabled, combines knowledge base and Wikipedia content
- **Provide accurate metadata**: Returns correct flags indicating which sources were actually used

### 2. Knowledge Base Integration
- Loads existing vector stores from the orchestration system
- Searches educational, unified, and vedas stores in priority order
- Retrieves relevant documents using semantic search
- Includes source metadata for transparency

### 3. Wikipedia Integration
- Uses the existing Wikipedia utilities from the subject generation module
- Fetches relevant articles based on subject and topic
- Caches results to improve performance
- Includes Wikipedia source information

### 4. Response Format
The enhanced endpoint now returns:
```json
{
  "title": "lesson title",
  "level": "educational level",
  "text": "comprehensive lesson content",
  "quiz": [...],
  "tts": true,
  "subject": "subject name",
  "topic": "topic name",
  "sources": [
    {
      "text": "content preview",
      "source": "source name",
      "store": "knowledge_base|wikipedia",
      "url": "url if available"
    }
  ],
  "knowledge_base_used": true/false,
  "wikipedia_used": true/false,
  "generated_at": "timestamp",
  "status": "success"
}
```

## Testing

### Test Cases
1. **Knowledge Base + Wikipedia**: Both sources enabled
2. **Wikipedia Only**: Only Wikipedia enabled
3. **Knowledge Base Only**: Only knowledge base enabled  
4. **Basic Generation**: Neither source enabled

### Test Script
Run the test script to verify functionality:
```bash
cd Backend/Base_backend
python test_enhanced_lesson_generation.py
```

For specific testing of the reported issue:
```bash
python test_enhanced_lesson_generation.py specific
```

## Configuration Requirements

### Dependencies
Added to `requirements.txt`:
- `wikipedia` - for Wikipedia API access

### Vector Stores
Ensure vector stores are available at:
- `Backend/orchestration/unified_orchestration_system/vector_stores/educational_index/`
- `Backend/orchestration/unified_orchestration_system/vector_stores/unified_index/`
- `Backend/orchestration/unified_orchestration_system/vector_stores/vedas_index/`

### Data Sources
Knowledge base content is sourced from:
- PDF files: Vedas, Gita, Upanishads, Ramayana
- CSV files: Educational curriculum data
- Located in: `Backend/orchestration/unified_orchestration_system/data/`

## Usage

### Frontend Integration
The frontend can now reliably use the parameters:
```javascript
const params = {
  subject: "science",
  topic: "environment", 
  include_wikipedia: true,    // Will fetch Wikipedia content
  use_knowledge_store: true   // Will search vector stores
};
```

### Expected Behavior
- **Knowledge Base Selected**: Returns content from educational vector stores
- **Wikipedia Selected**: Returns Wikipedia article content
- **Both Selected**: Combines both sources in the lesson
- **Neither Selected**: Uses basic LLM generation

## Verification
After implementing this fix:
1. Selecting "search the knowledgebase" will return content from vector stores
2. Selecting "include Wikipedia" will return Wikipedia content
3. The response metadata accurately reflects which sources were used
4. Sources are properly attributed in the response

The issue where knowledge base selection was returning Wikipedia data has been resolved.
