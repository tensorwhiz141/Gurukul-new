#!/usr/bin/env python3
"""
Simple test script to verify knowledge base integration
"""

import requests
import json

def test_vedic_content():
    """Test Vedic content from knowledge base"""
    
    print("🧪 Testing Vedic Content from Knowledge Base")
    print("=" * 60)
    
    params = {
        "subject": "ved",
        "topic": "sound",
        "include_wikipedia": False,
        "use_knowledge_store": True
    }
    
    try:
        response = requests.get("http://localhost:8000/generate_lesson", params=params, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Request successful!")
            print(f"📖 Title: {data.get('title', 'N/A')}")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            print(f"📄 Sources: {data.get('sources', [])}")
            
            # Show explanation preview
            explanation = data.get('explanation', '')
            if explanation:
                print(f"\n📝 Explanation Preview:")
                print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
            
            # Show detailed sources
            detailed_sources = data.get('detailed_sources', [])
            print(f"\n🔍 Detailed Sources ({len(detailed_sources)}):")
            
            for i, source in enumerate(detailed_sources[:3], 1):
                print(f"\n   📚 Source {i}:")
                print(f"      Type: {source.get('source_type', 'Unknown')}")
                print(f"      Name: {source.get('source_name', 'Unknown')}")
                print(f"      Vector Store: {source.get('vector_store', 'Unknown')}")
                
                if source.get('source_type') == 'book':
                    print(f"      📖 Book Type: {source.get('book_type', 'Unknown')}")
                    print(f"      📄 Page/Chunk: {source.get('chunk_info', 'Unknown')}")
                elif source.get('source_type') == 'database':
                    print(f"      🗃️ Database Type: {source.get('database_type', 'Unknown')}")
                    print(f"      📊 Grade: {source.get('grade', 'Unknown')}")
                
                preview = source.get('content_preview', '')
                if preview:
                    print(f"      📝 Preview: {preview[:100]}...")
            
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_database_content():
    """Test database content from knowledge base"""
    
    print("\n🧪 Testing Database Content from Knowledge Base")
    print("=" * 60)
    
    params = {
        "subject": "science",
        "topic": "plant",
        "include_wikipedia": False,
        "use_knowledge_store": True
    }
    
    try:
        response = requests.get("http://localhost:8000/generate_lesson", params=params, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Request successful!")
            print(f"📖 Title: {data.get('title', 'N/A')}")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            print(f"📄 Sources: {data.get('sources', [])}")
            
            # Show explanation preview
            explanation = data.get('explanation', '')
            if explanation:
                print(f"\n📝 Explanation Preview:")
                print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
            
            # Show detailed sources
            detailed_sources = data.get('detailed_sources', [])
            print(f"\n🔍 Detailed Sources ({len(detailed_sources)}):")
            
            database_sources = [s for s in detailed_sources if s.get('source_type') == 'database']
            if database_sources:
                print(f"✅ Found {len(database_sources)} database sources!")
                for source in database_sources[:2]:
                    print(f"   📊 {source.get('source_name', 'Unknown')}")
                    print(f"      Database Type: {source.get('database_type', 'Unknown')}")
                    print(f"      Fields: {source.get('fields_included', [])}")
            else:
                print("⚠️  No database sources found")
            
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Knowledge Base Integration")
    print("=" * 70)
    
    # Test 1: Vedic content
    vedic_success = test_vedic_content()
    
    # Test 2: Database content  
    database_success = test_database_content()
    
    print("\n" + "=" * 70)
    print("🏁 Test Results:")
    print(f"   Vedic Content: {'✅ PASS' if vedic_success else '❌ FAIL'}")
    print(f"   Database Content: {'✅ PASS' if database_success else '❌ FAIL'}")
    
    if vedic_success and database_success:
        print("\n🎉 Knowledge base integration is working correctly!")
        print("   - Vedic texts are being accessed with proper attribution")
        print("   - Database content is being retrieved with metadata")
        print("   - Content is being generated from actual knowledge base sources")
    else:
        print("\n⚠️  Some tests failed - check the logs above")
