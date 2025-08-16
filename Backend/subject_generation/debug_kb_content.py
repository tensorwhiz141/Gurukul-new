#!/usr/bin/env python3
"""
Debug script to check knowledge base content
"""

import requests
import json

def debug_kb_content():
    """Debug knowledge base content generation"""
    
    print("🔍 Debugging Knowledge Base Content")
    print("=" * 50)
    
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
            
            print(f"✅ Request successful!")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"📖 Title: {data.get('title', 'N/A')}")
            
            explanation = data.get('explanation', '')
            print(f"📝 Explanation Length: {len(explanation)}")
            if explanation:
                print(f"📝 Explanation Preview: {explanation[:300]}...")
            else:
                print("❌ No explanation content!")
            
            # Check detailed sources
            detailed_sources = data.get('detailed_sources', [])
            print(f"\n🔍 Detailed Sources: {len(detailed_sources)}")
            
            for i, source in enumerate(detailed_sources[:5], 1):
                print(f"\n   📚 Source {i}:")
                print(f"      Type: {source.get('source_type', 'Unknown')}")
                print(f"      Name: {source.get('source_name', 'Unknown')}")
                print(f"      Vector Store: {source.get('vector_store', 'Unknown')}")
                
                content_preview = source.get('content_preview', '')
                print(f"      Content Preview Length: {len(content_preview)}")
                if content_preview:
                    print(f"      Content Preview: {content_preview[:150]}...")
                else:
                    print("      ❌ No content preview!")
                
                # Check for specific fields
                if source.get('source_type') == 'book':
                    print(f"      Page Number: {source.get('page_number', 'Unknown')}")
                    print(f"      Book Type: {source.get('book_type', 'Unknown')}")
                elif source.get('source_type') == 'database':
                    print(f"      Record Number: {source.get('record_number', 'Unknown')}")
                    print(f"      Database Type: {source.get('database_type', 'Unknown')}")
            
            # Check if any sources have content
            sources_with_content = [s for s in detailed_sources if s.get('content_preview', '')]
            print(f"\n📊 Sources with content: {len(sources_with_content)}/{len(detailed_sources)}")
            
            if len(sources_with_content) == 0:
                print("❌ PROBLEM: No sources have content_preview!")
                print("This explains why the explanation is empty.")
            else:
                print("✅ Some sources have content - checking generation logic...")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    debug_kb_content()
