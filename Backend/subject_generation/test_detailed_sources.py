#!/usr/bin/env python3
"""
Test script to verify detailed source information including database and book sources
"""

import requests
import json
import time

def test_detailed_source_information():
    """Test that detailed source information is included in responses"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Detailed Source Information")
    print("=" * 60)
    print("Testing with subject='ved' and topic='sound' (likely to have book sources)")
    print("=" * 60)
    
    # Test Case 1: Knowledge Base with detailed sources
    print("\n1️⃣ Testing: Knowledge Base with Detailed Sources")
    print("   Parameters: use_knowledge_store=True, include_wikipedia=False")
    print("-" * 60)
    
    params_kb = {
        "subject": "ved",
        "topic": "sound",
        "include_wikipedia": False,
        "use_knowledge_store": True
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params_kb, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Status: Success")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            
            # Check basic sources
            sources = data.get('sources', [])
            print(f"📄 Basic Sources Count: {len(sources)}")
            
            # Check detailed sources
            detailed_sources = data.get('detailed_sources', [])
            print(f"🔍 Detailed Sources Count: {len(detailed_sources)}")
            
            if detailed_sources:
                print("\n📋 Detailed Source Information:")
                for i, source in enumerate(detailed_sources[:3], 1):  # Show first 3
                    print(f"\n   Source {i}:")
                    print(f"      Type: {source.get('source_type', 'Unknown')}")
                    print(f"      Name: {source.get('source_name', 'Unknown')}")
                    
                    if source.get('source_type') == 'book':
                        print(f"      📖 Book Type: {source.get('book_type', 'Unknown')}")
                        print(f"      📄 Page Number: {source.get('page_number', 'Unknown')}")
                        print(f"      🌐 Language: {source.get('language', 'Unknown')}")
                        print(f"      📝 Preview: {source.get('content_preview', '')[:100]}...")
                        
                    elif source.get('source_type') == 'database':
                        print(f"      🗃️ Database Type: {source.get('database_type', 'Unknown')}")
                        print(f"      📊 Record Number: {source.get('record_number', 'Unknown')}")
                        print(f"      🏷️ Fields: {source.get('fields_included', [])}")
                        print(f"      📝 Preview: {source.get('content_preview', '')[:100]}...")
                        
                    elif source.get('source_type') == 'wikipedia':
                        print(f"      🌐 URL: {source.get('url', 'Unknown')}")
                        print(f"      📅 Access Date: {source.get('access_date', 'Unknown')}")
                        print(f"      📝 Preview: {source.get('content_preview', '')[:100]}...")
                        
                    elif source.get('source_type') == 'llm_generation':
                        print(f"      🤖 Model: {source.get('model', 'Unknown')}")
                        print(f"      📅 Generation Date: {source.get('generation_date', 'Unknown')}")
                        print(f"      📝 Preview: {source.get('content_preview', '')[:100]}...")
                        
                    print(f"      🎯 Vector Store: {source.get('vector_store', 'N/A')}")
                    print(f"      📊 Relevance Score: {source.get('relevance_score', 'N/A')}")
                    
                if len(detailed_sources) > 3:
                    print(f"\n   ... and {len(detailed_sources) - 3} more sources")
                    
            else:
                print("⚠️  No detailed sources found")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_source_types():
    """Test different types of sources"""
    
    base_url = "http://localhost:8000"
    
    print("\n🔍 Testing Different Source Types")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Vedic Content (Books)",
            "params": {"subject": "ved", "topic": "sound", "include_wikipedia": False, "use_knowledge_store": True}
        },
        {
            "name": "Plant Database (CSV)",
            "params": {"subject": "science", "topic": "plant", "include_wikipedia": False, "use_knowledge_store": True}
        },
        {
            "name": "Wikipedia Source",
            "params": {"subject": "science", "topic": "motion", "include_wikipedia": True, "use_knowledge_store": False}
        },
        {
            "name": "Combined Sources",
            "params": {"subject": "ved", "topic": "knowledge", "include_wikipedia": True, "use_knowledge_store": True}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            response = requests.get(f"{base_url}/generate_lesson", params=test_case['params'], timeout=45)
            
            if response.status_code == 200:
                data = response.json()
                detailed_sources = data.get('detailed_sources', [])
                
                print(f"✅ Success - {len(detailed_sources)} detailed sources")
                
                # Categorize sources
                source_types = {}
                for source in detailed_sources:
                    source_type = source.get('source_type', 'unknown')
                    if source_type not in source_types:
                        source_types[source_type] = 0
                    source_types[source_type] += 1
                
                print(f"📊 Source Types Found:")
                for source_type, count in source_types.items():
                    print(f"   - {source_type}: {count}")
                    
                # Show one example of each type
                shown_types = set()
                for source in detailed_sources:
                    source_type = source.get('source_type', 'unknown')
                    if source_type not in shown_types:
                        print(f"\n   📋 Example {source_type} source:")
                        print(f"      Name: {source.get('source_name', 'Unknown')}")
                        if source_type == 'book':
                            print(f"      Page: {source.get('page_number', 'Unknown')}")
                        elif source_type == 'database':
                            print(f"      Record: {source.get('record_number', 'Unknown')}")
                        elif source_type == 'wikipedia':
                            print(f"      URL: {source.get('url', 'Unknown')}")
                        shown_types.add(source_type)
                        
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Failed: {e}")

def test_source_attribution_accuracy():
    """Test that source attribution is accurate"""
    
    base_url = "http://localhost:8000"
    
    print("\n🎯 Testing Source Attribution Accuracy")
    print("=" * 50)
    
    # Test knowledge base only
    params = {
        "subject": "ved",
        "topic": "sound",
        "include_wikipedia": False,
        "use_knowledge_store": True
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params, timeout=45)
        
        if response.status_code == 200:
            data = response.json()
            detailed_sources = data.get('detailed_sources', [])
            
            print(f"✅ Response received with {len(detailed_sources)} detailed sources")
            
            # Verify no Wikipedia sources when disabled
            wikipedia_sources = [s for s in detailed_sources if s.get('source_type') == 'wikipedia']
            if wikipedia_sources:
                print(f"❌ FAIL: Found {len(wikipedia_sources)} Wikipedia sources when disabled")
            else:
                print("✅ PASS: No Wikipedia sources when disabled")
            
            # Check for book sources (expected for Vedic content)
            book_sources = [s for s in detailed_sources if s.get('source_type') == 'book']
            if book_sources:
                print(f"✅ PASS: Found {len(book_sources)} book sources (expected for Vedic content)")
                
                # Check if page numbers are provided
                sources_with_pages = [s for s in book_sources if s.get('page_number') and s.get('page_number') != 'Unknown']
                print(f"📄 Sources with page numbers: {len(sources_with_pages)}/{len(book_sources)}")
                
                # Show example
                if sources_with_pages:
                    example = sources_with_pages[0]
                    print(f"📖 Example: {example.get('source_name')} - Page {example.get('page_number')}")
                    
            else:
                print("⚠️  No book sources found (may be normal depending on content)")
                
            # Check database sources
            db_sources = [s for s in detailed_sources if s.get('source_type') == 'database']
            if db_sources:
                print(f"🗃️ Found {len(db_sources)} database sources")
                
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_detailed_source_information()
    test_source_types()
    test_source_attribution_accuracy()
    
    print("\n" + "=" * 60)
    print("🏁 Detailed Source Testing Completed!")
    print("\n💡 Expected Features:")
    print("   - Book sources show page numbers")
    print("   - Database sources show record numbers and fields")
    print("   - Wikipedia sources show URLs and access dates")
    print("   - LLM sources show model and generation info")
    print("   - All sources include content previews")
    print("   - Vector store information is included")
    print("   - Source types are correctly categorized")
