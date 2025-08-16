# Orchestration System Data Integration

## Overview
The subject generation system is now fully integrated with the orchestration system's data folder, providing access to comprehensive educational content including Vedic texts, educational databases, and wellness information.

## Data Sources Integrated

### 📖 **Vedic Literature (PDF Books)**
- **Four-Vedas-English-Translation.pdf** - Complete Four Vedas in English
- **108upanishads.pdf** - Collection of 108 Upanishads
- **Gita.pdf** - Bhagavad Gita
- **ramayan.pdf** - Ramayana epic

### 🗃️ **Educational Databases (CSV Files)**
- **Plant_8-12.csv** - Botanical database for grades 8-12
- **Seed_1-7.csv** - Agricultural database for grades 1-7
- **Tree.csv** - Forestry database

## Integration Architecture

### 1. **Data Processing Pipeline**
```
Orchestration Data Folder
    ↓
UnifiedDataIngestion System
    ↓
Specialized Vector Stores
    ↓
Subject Generation API
    ↓
Detailed Source Attribution
```

### 2. **Vector Store Categories**
- **vedas_index**: Vedic scriptures and spiritual content
- **educational_index**: Educational curriculum data
- **wellness_index**: Health and wellness content
- **unified_index**: All content combined

### 3. **Content Categorization**
The system automatically categorizes content based on:
- **Filename analysis**: Detects Vedic texts by filename
- **Keyword matching**: Uses predefined keyword sets
- **Metadata analysis**: Examines CSV fields and PDF metadata

## Knowledge Store Integration

### 1. **Enhanced Source Tracking**
When `use_knowledge_store=True`, the system:
- Loads vector stores from orchestration system
- Searches in priority order based on subject
- Extracts detailed metadata including page numbers and record information
- Provides comprehensive source attribution

### 2. **Priority Search Order**
- **Vedic subjects** (`ved`, `sanskrit`, `spiritual`): vedas → educational → unified
- **Health subjects** (`health`, `wellness`, `psychology`): wellness → educational → unified  
- **Other subjects**: educational → unified → vedas

### 3. **Detailed Source Information**
```json
{
  "detailed_sources": [
    {
      "source_type": "book",
      "source_name": "Four-Vedas-English-Translation",
      "page_number": 42,
      "book_type": "Four Vedas",
      "content_category": "vedas",
      "vector_store": "vedas",
      "language": "English",
      "chunk_info": "Page 42"
    },
    {
      "source_type": "database", 
      "source_name": "Plant_8-12",
      "database_type": "Botanical Database",
      "education_level": "secondary",
      "grade": "8-12",
      "fields_included": ["Subject", "Topic", "Learning Outcome"]
    }
  ]
}
```

## Setup and Configuration

### 1. **Initial Setup**
```bash
cd Backend/subject_generation
python setup_orchestration_integration.py
```

This script will:
- Check for data files in orchestration system
- Create vector stores if they don't exist
- Test the integration

### 2. **Manual Vector Store Creation**
```bash
cd Backend/orchestration/unified_orchestration_system
python data_ingestion.py
```

### 3. **Verify Integration**
```bash
cd Backend/subject_generation
python test_orchestration_integration.py
```

## API Usage

### 1. **Knowledge Store Only**
```bash
curl "http://localhost:8000/generate_lesson?subject=ved&topic=sound&include_wikipedia=false&use_knowledge_store=true"
```

**Expected Response:**
- Sources from Vedic texts with page numbers
- Content categorized as "vedas"
- Book type identification (Four Vedas, Upanishads, etc.)

### 2. **Educational Database Content**
```bash
curl "http://localhost:8000/generate_lesson?subject=science&topic=plant&include_wikipedia=false&use_knowledge_store=true"
```

**Expected Response:**
- Sources from Plant_8-12.csv database
- Record numbers and field information
- Education level and grade metadata

## Content Examples

### 1. **Vedic Content Response**
```json
{
  "title": "Understanding Sound in Vedic Tradition",
  "text": "This lesson explores the concept of sound within Vedic scriptures...",
  "detailed_sources": [
    {
      "source_type": "book",
      "source_name": "Four-Vedas-English-Translation",
      "page_number": 156,
      "book_type": "Four Vedas",
      "content_preview": "The sacred sound Om represents the primordial vibration...",
      "vector_store": "vedas"
    }
  ],
  "knowledge_base_used": true,
  "wikipedia_used": false
}
```

### 2. **Educational Database Response**
```json
{
  "title": "Plant Biology Fundamentals",
  "text": "This lesson covers plant biology using educational database content...",
  "detailed_sources": [
    {
      "source_type": "database",
      "source_name": "Plant_8-12",
      "database_type": "Botanical Database",
      "education_level": "secondary",
      "grade": "8-12",
      "fields_included": ["Subject", "Topic", "Learning Outcome", "Description"],
      "content_preview": "Plants are multicellular organisms that perform photosynthesis..."
    }
  ],
  "knowledge_base_used": true,
  "wikipedia_used": false
}
```

## Benefits

### 1. **Comprehensive Content Access**
- Access to authentic Vedic literature with page references
- Educational curriculum data with grade-level appropriateness
- Structured database information with field details

### 2. **Accurate Source Attribution**
- Page numbers for book sources
- Record numbers for database sources
- Content categorization and metadata

### 3. **Intelligent Content Selection**
- Priority-based search for relevant content
- Subject-appropriate source selection
- Automatic content categorization

### 4. **Educational Value**
- Students can reference original sources
- Teachers can verify content accuracy
- Researchers can explore primary texts

## Troubleshooting

### 1. **No Sources Found**
- Check if data files exist in orchestration/unified_orchestration_system/data/
- Run setup script to create vector stores
- Verify vector stores exist in orchestration/unified_orchestration_system/vector_stores/

### 2. **Missing Page Numbers**
- Ensure PDF files have proper metadata
- Check if PDF processing completed successfully
- Verify chunk metadata includes page information

### 3. **Incorrect Categorization**
- Check keyword lists in data_ingestion.py
- Verify filename patterns for automatic detection
- Review content categorization logic

### 4. **Performance Issues**
- Vector stores may need time to load initially
- Large datasets may require batch processing
- Consider increasing timeout values for complex queries

## Future Enhancements

1. **Additional Data Sources**: Support for more file formats
2. **Advanced Search**: Semantic search improvements
3. **Caching**: Intelligent caching with parameter awareness
4. **Analytics**: Usage tracking and content popularity metrics

The integration provides a robust foundation for accessing comprehensive educational content with full source transparency and academic rigor.
