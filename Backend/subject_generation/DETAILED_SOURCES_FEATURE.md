# Detailed Source Attribution Feature

## Overview
Enhanced the subject generation system to provide detailed source information including database sources and page numbers from books. This feature gives users complete transparency about where the lesson content comes from.

## Features Added

### 1. **Enhanced Source Tracking**
- **Book Sources**: Page numbers, book type, language detection
- **Database Sources**: Record numbers, field information, database type
- **Wikipedia Sources**: URLs, access dates, reliability indicators
- **LLM Sources**: Model information, generation parameters, timestamps

### 2. **Detailed Source Information Structure**
```json
{
  "detailed_sources": [
    {
      "source_type": "book",
      "source_name": "Four-Vedas-English-Translation",
      "file_path": "/path/to/Four-Vedas-English-Translation.pdf",
      "page_number": 42,
      "content_preview": "Content preview from the page...",
      "vector_store": "vedas",
      "relevance_score": 0.85,
      "book_type": "Vedic Scripture",
      "language": "English"
    },
    {
      "source_type": "database",
      "source_name": "Plant_8-12",
      "file_path": "/path/to/Plant_8-12.csv",
      "record_number": 156,
      "content_preview": "Plant data fields...",
      "vector_store": "educational",
      "database_type": "Botanical Database",
      "fields_included": ["name", "family", "properties", "uses"]
    },
    {
      "source_type": "wikipedia",
      "source_name": "Wikipedia: Motion (physics)",
      "url": "https://en.wikipedia.org/wiki/Motion_(physics)",
      "content_preview": "Motion is a change in position...",
      "access_date": "2024-01-15",
      "reliability": "community_edited",
      "language": "English"
    },
    {
      "source_type": "llm_generation",
      "source_name": "Ollama LLM (llama2)",
      "model": "llama2",
      "generation_date": "2024-01-15 14:30:25",
      "content_preview": "Generated explanation...",
      "reliability": "ai_generated",
      "parameters": {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": true,
        "use_knowledge_store": true
      }
    }
  ]
}
```

### 3. **Source Type Detection**
- **Books**: Automatically detects PDF sources and extracts page numbers
- **Databases**: Identifies CSV sources and extracts record information
- **Wikipedia**: Tracks Wikipedia articles with URLs and metadata
- **LLM**: Records AI-generated content with model details

### 4. **Book Type Classification**
- **Vedic Scripture**: Four-Vedas-English-Translation.pdf
- **Upanishad**: 108upanishads.pdf
- **Bhagavad Gita**: Gita.pdf
- **Epic Literature**: ramayan.pdf
- **Religious/Philosophical Text**: Other spiritual texts

### 5. **Database Type Classification**
- **Botanical Database**: Plant_8-12.csv
- **Agricultural Database**: Seed_1-7.csv
- **Forestry Database**: Tree.csv
- **Educational Database**: Other CSV files

## Implementation Details

### 1. **Enhanced Lesson Data Structure**
```python
lesson_data = {
    "subject": subject,
    "topic": topic,
    "title": "lesson title",
    "text": "lesson content",
    "sources": ["basic source names"],
    "detailed_sources": [...]  # NEW: Detailed source information
}
```

### 2. **Knowledge Base Source Extraction**
```python
def get_detailed_knowledge_base_sources(subject: str, topic: str) -> List[Dict[str, Any]]:
    # Searches vector stores for relevant documents
    # Extracts metadata including page numbers and record information
    # Classifies source types and provides detailed attribution
```

### 3. **Source Information Functions**
- `get_book_type(book_name)`: Classifies book types
- `get_database_type(db_name)`: Classifies database types
- `detect_content_language(content)`: Detects content language
- `extract_csv_fields(content)`: Extracts CSV field names

## API Response Enhancement

### Before (Basic Sources)
```json
{
  "sources": [
    "Wikipedia",
    "Four-Vedas-English-Translation",
    "Plant_8-12"
  ]
}
```

### After (Detailed Sources)
```json
{
  "sources": [
    "Wikipedia: Motion (physics)",
    "Four-Vedas-English-Translation",
    "Plant_8-12"
  ],
  "detailed_sources": [
    {
      "source_type": "wikipedia",
      "source_name": "Wikipedia: Motion (physics)",
      "url": "https://en.wikipedia.org/wiki/Motion_(physics)",
      "content_preview": "Motion is a change in position of an object...",
      "access_date": "2024-01-15",
      "reliability": "community_edited"
    },
    {
      "source_type": "book",
      "source_name": "Four-Vedas-English-Translation",
      "page_number": 42,
      "book_type": "Vedic Scripture",
      "content_preview": "The concept of sound in Vedic tradition...",
      "language": "English"
    },
    {
      "source_type": "database",
      "source_name": "Plant_8-12",
      "record_number": 156,
      "database_type": "Botanical Database",
      "fields_included": ["name", "family", "properties"],
      "content_preview": "Medicinal plant data..."
    }
  ]
}
```

## Testing

### Test Script
```bash
cd Backend/subject_generation
python test_detailed_sources.py
```

### Test Cases
1. **Book Sources**: Test with Vedic content to get page numbers
2. **Database Sources**: Test with plant/seed topics to get CSV data
3. **Wikipedia Sources**: Test with scientific topics
4. **Combined Sources**: Test with multiple source types

### Expected Results
- ✅ Book sources include page numbers
- ✅ Database sources include record numbers and field names
- ✅ Wikipedia sources include URLs and access dates
- ✅ LLM sources include model and generation information
- ✅ All sources include content previews
- ✅ Source types are correctly classified

## Benefits

### 1. **Complete Transparency**
Users can see exactly where each piece of information comes from

### 2. **Academic Citation**
Provides enough information for proper academic citation

### 3. **Source Verification**
Users can verify information by checking the original sources

### 4. **Quality Assessment**
Different source types have reliability indicators

### 5. **Research Enhancement**
Students can explore original sources for deeper learning

## Usage Examples

### Frontend Display
```javascript
// Display detailed sources
data.detailed_sources.forEach(source => {
  if (source.source_type === 'book') {
    console.log(`📖 ${source.source_name} - Page ${source.page_number}`);
  } else if (source.source_type === 'database') {
    console.log(`🗃️ ${source.source_name} - Record ${source.record_number}`);
  } else if (source.source_type === 'wikipedia') {
    console.log(`🌐 ${source.source_name} - ${source.url}`);
  }
});
```

### Citation Generation
```javascript
function generateCitation(source) {
  if (source.source_type === 'book') {
    return `${source.source_name}, Page ${source.page_number}`;
  } else if (source.source_type === 'wikipedia') {
    return `${source.source_name}, accessed ${source.access_date}, ${source.url}`;
  }
  // ... other source types
}
```

This enhancement provides complete source transparency and enables proper academic attribution for all lesson content.
