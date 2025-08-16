#!/usr/bin/env python3
"""
Test script to verify integration with orchestration system data
Tests the knowledge store option with actual data from the data folder
"""

import requests
import json
import time
import os

def test_orchestration_data_integration():
    """Test that the knowledge store uses data from the orchestration system"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Orchestration System Data Integration")
    print("=" * 70)
    print("Testing knowledge store with data from Backend/orchestration/unified_orchestration_system/data/")
    print("=" * 70)
    
    # Test cases designed to match the data files in the orchestration system
    test_cases = [
        {
            "name": "Vedic Content (Four Vedas)",
            "params": {
                "subject": "ved",
                "topic": "sound",
                "include_wikipedia": False,
                "use_knowledge_store": True
            },
            "expected_sources": ["Four-Vedas-English-Translation", "vedas"],
            "expected_types": ["book"]
        },
        {
            "name": "Upanishads Content",
            "params": {
                "subject": "ved",
                "topic": "consciousness",
                "include_wikipedia": False,
                "use_knowledge_store": True
            },
            "expected_sources": ["108upanishads"],
            "expected_types": ["book"]
        },
        {
            "name": "Bhagavad Gita Content",
            "params": {
                "subject": "ved",
                "topic": "dharma",
                "include_wikipedia": False,
                "use_knowledge_store": True
            },
            "expected_sources": ["Gita"],
            "expected_types": ["book"]
        },
        {
            "name": "Plant Database Content",
            "params": {
                "subject": "science",
                "topic": "plant",
                "include_wikipedia": False,
                "use_knowledge_store": True
            },
            "expected_sources": ["Plant_8-12"],
            "expected_types": ["database"]
        },
        {
            "name": "Seed Database Content",
            "params": {
                "subject": "agriculture",
                "topic": "seed",
                "include_wikipedia": False,
                "use_knowledge_store": True
            },
            "expected_sources": ["Seed_1-7"],
            "expected_types": ["database"]
        },
        {
            "name": "Tree Database Content",
            "params": {
                "subject": "forestry",
                "topic": "tree",
                "include_wikipedia": False,
                "use_knowledge_store": True
            },
            "expected_sources": ["Tree"],
            "expected_types": ["database"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ Testing: {test_case['name']}")
        print(f"   Subject: {test_case['params']['subject']}, Topic: {test_case['params']['topic']}")
        print("-" * 60)
        
        try:
            response = requests.get(f"{base_url}/generate_lesson", params=test_case['params'], timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Status: Success")
                print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
                print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
                
                # Check detailed sources
                detailed_sources = data.get('detailed_sources', [])
                print(f"🔍 Detailed Sources Found: {len(detailed_sources)}")
                
                if detailed_sources:
                    # Verify source types
                    found_types = set(source.get('source_type', 'unknown') for source in detailed_sources)
                    expected_types = set(test_case['expected_types'])
                    
                    print(f"📋 Source Types Found: {list(found_types)}")
                    print(f"📋 Expected Types: {list(expected_types)}")
                    
                    if expected_types.intersection(found_types):
                        print("✅ PASS: Expected source types found")
                    else:
                        print("⚠️  WARNING: Expected source types not found")
                    
                    # Check for expected source names
                    found_sources = [source.get('source_name', '') for source in detailed_sources]
                    print(f"📄 Sources Found: {found_sources}")
                    
                    expected_found = False
                    for expected_source in test_case['expected_sources']:
                        if any(expected_source in source for source in found_sources):
                            expected_found = True
                            break
                    
                    if expected_found:
                        print("✅ PASS: Expected data sources found")
                    else:
                        print("⚠️  WARNING: Expected data sources not found")
                    
                    # Show detailed source information
                    for j, source in enumerate(detailed_sources[:2], 1):  # Show first 2
                        print(f"\n   📚 Source {j} Details:")
                        print(f"      Type: {source.get('source_type', 'Unknown')}")
                        print(f"      Name: {source.get('source_name', 'Unknown')}")
                        print(f"      Vector Store: {source.get('vector_store', 'Unknown')}")
                        print(f"      Content Category: {source.get('content_category', 'Unknown')}")
                        
                        if source.get('source_type') == 'book':
                            print(f"      📖 Book Type: {source.get('book_type', 'Unknown')}")
                            print(f"      📄 Page/Chunk: {source.get('chunk_info', 'Unknown')}")
                            print(f"      🌐 Language: {source.get('language', 'Unknown')}")
                            
                        elif source.get('source_type') == 'database':
                            print(f"      🗃️ Database Type: {source.get('database_type', 'Unknown')}")
                            print(f"      🎓 Education Level: {source.get('education_level', 'Unknown')}")
                            print(f"      📊 Grade: {source.get('grade', 'Unknown')}")
                            print(f"      🏷️ Fields: {source.get('fields_included', [])}")
                        
                        print(f"      📝 Preview: {source.get('content_preview', '')[:100]}...")
                        
                else:
                    print("❌ FAIL: No detailed sources found")
                    
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        # Small delay between requests
        time.sleep(1)

def test_vector_store_priority():
    """Test that vector stores are searched in the correct priority order"""
    
    base_url = "http://localhost:8000"
    
    print("\n🎯 Testing Vector Store Priority")
    print("=" * 50)
    
    priority_tests = [
        {
            "name": "Vedic Subject Priority",
            "params": {"subject": "ved", "topic": "knowledge", "include_wikipedia": False, "use_knowledge_store": True},
            "expected_priority": ["vedas", "educational", "unified"]
        },
        {
            "name": "Health Subject Priority", 
            "params": {"subject": "health", "topic": "wellness", "include_wikipedia": False, "use_knowledge_store": True},
            "expected_priority": ["wellness", "educational", "unified"]
        },
        {
            "name": "Science Subject Priority",
            "params": {"subject": "science", "topic": "biology", "include_wikipedia": False, "use_knowledge_store": True},
            "expected_priority": ["educational", "unified"]
        }
    ]
    
    for test in priority_tests:
        print(f"\n🧪 Testing: {test['name']}")
        print(f"   Expected Priority: {test['expected_priority']}")
        print("-" * 40)
        
        try:
            response = requests.get(f"{base_url}/generate_lesson", params=test['params'], timeout=45)
            
            if response.status_code == 200:
                data = response.json()
                detailed_sources = data.get('detailed_sources', [])
                
                if detailed_sources:
                    vector_stores_used = [source.get('vector_store', 'unknown') for source in detailed_sources]
                    unique_stores = list(dict.fromkeys(vector_stores_used))  # Preserve order, remove duplicates
                    
                    print(f"✅ Vector Stores Used: {unique_stores}")
                    
                    # Check if the first store matches expected priority
                    if unique_stores and unique_stores[0] in test['expected_priority'][:2]:
                        print("✅ PASS: Correct priority order")
                    else:
                        print("⚠️  INFO: Different priority order (may be normal)")
                else:
                    print("⚠️  No sources found")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")

def check_data_files():
    """Check if the orchestration system data files exist"""
    
    print("\n📁 Checking Orchestration System Data Files")
    print("=" * 50)
    
    # Expected data files
    orchestration_path = os.path.join(os.path.dirname(__file__), '..', 'orchestration', 'unified_orchestration_system', 'data')
    
    expected_files = [
        "Four-Vedas-English-Translation.pdf",
        "108upanishads.pdf", 
        "Gita.pdf",
        "ramayan.pdf",
        "Plant_8-12.csv",
        "Seed_1-7.csv",
        "Tree.csv"
    ]
    
    print(f"📂 Data Directory: {orchestration_path}")
    
    if os.path.exists(orchestration_path):
        print("✅ Data directory exists")
        
        for file in expected_files:
            file_path = os.path.join(orchestration_path, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✅ {file} - {file_size:,} bytes")
            else:
                print(f"❌ {file} - NOT FOUND")
    else:
        print("❌ Data directory does not exist")
        print("💡 Please ensure the orchestration system data is available")

if __name__ == "__main__":
    check_data_files()
    test_orchestration_data_integration()
    test_vector_store_priority()
    
    print("\n" + "=" * 70)
    print("🏁 Orchestration Integration Testing Completed!")
    print("\n💡 Expected Results:")
    print("   - Vedic content should come from PDF books with page numbers")
    print("   - Plant/Seed/Tree content should come from CSV databases")
    print("   - Vector stores should be searched in priority order")
    print("   - Detailed source attribution should include file paths and metadata")
    print("   - Content should be categorized correctly (vedas/educational/wellness)")
    print("\n🔧 If tests fail:")
    print("   1. Ensure orchestration system data files exist")
    print("   2. Run data ingestion: cd Backend/orchestration/unified_orchestration_system && python data_ingestion.py")
    print("   3. Restart the subject generation service")
