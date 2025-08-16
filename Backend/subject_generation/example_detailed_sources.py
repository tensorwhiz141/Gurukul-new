#!/usr/bin/env python3
"""
Example script demonstrating the detailed source attribution feature
"""

import requests
import json

def demonstrate_detailed_sources():
    """Demonstrate the detailed source attribution feature"""
    
    base_url = "http://localhost:8000"
    
    print("🎯 Detailed Source Attribution Demo")
    print("=" * 50)
    
    # Example request for Vedic content (likely to have book sources)
    params = {
        "subject": "ved",
        "topic": "sound",
        "include_wikipedia": False,
        "use_knowledge_store": True
    }
    
    print(f"📝 Requesting lesson: {params['subject']}/{params['topic']}")
    print(f"🔧 Parameters: Wikipedia={params['include_wikipedia']}, Knowledge Base={params['use_knowledge_store']}")
    print("-" * 50)
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Lesson generated successfully!")
            print(f"📖 Title: {data.get('title', 'N/A')}")
            print(f"📄 Content Length: {len(data.get('text', ''))} characters")
            
            # Display basic sources
            basic_sources = data.get('sources', [])
            print(f"\n📋 Basic Sources ({len(basic_sources)}):")
            for i, source in enumerate(basic_sources, 1):
                print(f"   {i}. {source}")
            
            # Display detailed sources
            detailed_sources = data.get('detailed_sources', [])
            print(f"\n🔍 Detailed Sources ({len(detailed_sources)}):")
            
            if detailed_sources:
                for i, source in enumerate(detailed_sources, 1):
                    print(f"\n   📚 Source {i}:")
                    print(f"      Type: {source.get('source_type', 'Unknown')}")
                    print(f"      Name: {source.get('source_name', 'Unknown')}")
                    
                    # Display type-specific information
                    if source.get('source_type') == 'book':
                        print(f"      📖 Book Type: {source.get('book_type', 'Unknown')}")
                        print(f"      📄 Page Number: {source.get('page_number', 'Unknown')}")
                        print(f"      🌐 Language: {source.get('language', 'Unknown')}")
                        print(f"      📁 File: {source.get('file_path', 'Unknown')}")
                        
                    elif source.get('source_type') == 'database':
                        print(f"      🗃️ Database Type: {source.get('database_type', 'Unknown')}")
                        print(f"      📊 Record Number: {source.get('record_number', 'Unknown')}")
                        print(f"      🏷️ Fields: {', '.join(source.get('fields_included', []))}")
                        print(f"      📁 File: {source.get('file_path', 'Unknown')}")
                        
                    elif source.get('source_type') == 'wikipedia':
                        print(f"      🌐 URL: {source.get('url', 'Unknown')}")
                        print(f"      📅 Access Date: {source.get('access_date', 'Unknown')}")
                        print(f"      🔒 Reliability: {source.get('reliability', 'Unknown')}")
                        
                    elif source.get('source_type') == 'llm_generation':
                        print(f"      🤖 Model: {source.get('model', 'Unknown')}")
                        print(f"      📅 Generated: {source.get('generation_date', 'Unknown')}")
                        print(f"      🔒 Reliability: {source.get('reliability', 'Unknown')}")
                    
                    # Common fields
                    print(f"      🎯 Vector Store: {source.get('vector_store', 'N/A')}")
                    print(f"      📊 Relevance: {source.get('relevance_score', 'N/A')}")
                    
                    # Content preview
                    preview = source.get('content_preview', '')
                    if preview:
                        print(f"      📝 Preview: {preview[:150]}...")
                        
                # Generate citation examples
                print(f"\n📚 Citation Examples:")
                for i, source in enumerate(detailed_sources[:3], 1):
                    citation = generate_citation(source)
                    print(f"   {i}. {citation}")
                    
            else:
                print("   ⚠️ No detailed sources available")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def generate_citation(source):
    """Generate a proper citation for a source"""
    
    source_type = source.get('source_type', 'unknown')
    source_name = source.get('source_name', 'Unknown Source')
    
    if source_type == 'book':
        page = source.get('page_number', 'Unknown')
        book_type = source.get('book_type', 'Text')
        return f"{source_name} ({book_type}), Page {page}"
        
    elif source_type == 'database':
        record = source.get('record_number', 'Unknown')
        db_type = source.get('database_type', 'Database')
        return f"{source_name} ({db_type}), Record {record}"
        
    elif source_type == 'wikipedia':
        url = source.get('url', '')
        access_date = source.get('access_date', 'Unknown')
        return f"{source_name}, accessed {access_date}, {url}"
        
    elif source_type == 'llm_generation':
        model = source.get('model', 'Unknown')
        gen_date = source.get('generation_date', 'Unknown')
        return f"{source_name}, generated {gen_date}"
        
    else:
        return f"{source_name} ({source_type})"

def show_feature_benefits():
    """Show the benefits of detailed source attribution"""
    
    print("\n🎯 Feature Benefits")
    print("=" * 30)
    print("✅ Complete Transparency: See exactly where content comes from")
    print("✅ Academic Citation: Proper attribution for research")
    print("✅ Source Verification: Check original sources")
    print("✅ Quality Assessment: Different reliability indicators")
    print("✅ Research Enhancement: Explore original materials")
    
    print("\n📖 Source Types Supported:")
    print("   📚 Books: Page numbers, book types, languages")
    print("   🗃️ Databases: Record numbers, field information")
    print("   🌐 Wikipedia: URLs, access dates, reliability")
    print("   🤖 LLM: Model info, generation parameters")
    
    print("\n💡 Usage Examples:")
    print("   - Students can cite sources properly")
    print("   - Teachers can verify content accuracy")
    print("   - Researchers can explore original texts")
    print("   - Content creators can ensure attribution")

if __name__ == "__main__":
    demonstrate_detailed_sources()
    show_feature_benefits()
    
    print("\n" + "=" * 50)
    print("🏁 Demo completed!")
    print("\n🧪 To test the feature:")
    print("   python test_detailed_sources.py")
    print("\n📚 For more information:")
    print("   See DETAILED_SOURCES_FEATURE.md")
