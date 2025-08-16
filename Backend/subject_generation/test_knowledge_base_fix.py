#!/usr/bin/env python3
"""
Test script to verify the knowledge base vs Wikipedia fix
This tests the specific issue where selecting "search knowledgebase" was returning Wikipedia data
"""

import requests
import json
import time

def test_knowledge_base_vs_wikipedia():
    """Test the specific issue: knowledge base vs Wikipedia differentiation"""
    
    base_url = "http://localhost:8000"  # Subject generation service
    
    print("🧪 Testing Knowledge Base vs Wikipedia Fix")
    print("=" * 60)
    print("Testing with subject='science' and topic='motion'")
    print("=" * 60)
    
    # Test Case 1: Knowledge Base Only (should NOT return Wikipedia content)
    print("\n1️⃣ Testing: Knowledge Base ONLY (use_knowledge_store=True, include_wikipedia=False)")
    print("-" * 50)
    
    params_kb_only = {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": False,  # Explicitly disable Wikipedia
        "use_knowledge_store": True   # Enable knowledge base
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params_kb_only, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Status: Success")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            print(f"📄 Sources Count: {len(data.get('sources', []))}")
            
            # Check sources
            sources = data.get('sources', [])
            wikipedia_sources = [s for s in sources if s.get('store') == 'wikipedia']
            kb_sources = [s for s in sources if s.get('store') == 'knowledge_base']
            
            print(f"🌐 Wikipedia Sources: {len(wikipedia_sources)}")
            print(f"📚 Knowledge Base Sources: {len(kb_sources)}")
            
            # Verify no Wikipedia content when disabled
            if data.get('wikipedia_used', False) or wikipedia_sources:
                print("❌ FAIL: Wikipedia content found when it should be disabled!")
            else:
                print("✅ PASS: No Wikipedia content (as expected)")
                
            # Check if content mentions Wikipedia
            text_content = data.get('text', '').lower()
            if 'wikipedia' in text_content:
                print("⚠️  WARNING: Text content mentions 'Wikipedia'")
            else:
                print("✅ PASS: Text content doesn't mention Wikipedia")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test Case 2: Wikipedia Only (should NOT return knowledge base content)
    print("\n2️⃣ Testing: Wikipedia ONLY (use_knowledge_store=False, include_wikipedia=True)")
    print("-" * 50)
    
    params_wiki_only = {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": True,   # Enable Wikipedia
        "use_knowledge_store": False  # Explicitly disable knowledge base
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params_wiki_only, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Status: Success")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            print(f"📄 Sources Count: {len(data.get('sources', []))}")
            
            # Check sources
            sources = data.get('sources', [])
            wikipedia_sources = [s for s in sources if s.get('store') == 'wikipedia']
            kb_sources = [s for s in sources if s.get('store') == 'knowledge_base']
            
            print(f"🌐 Wikipedia Sources: {len(wikipedia_sources)}")
            print(f"📚 Knowledge Base Sources: {len(kb_sources)}")
            
            # Verify no knowledge base content when disabled
            if data.get('knowledge_base_used', False) or kb_sources:
                print("❌ FAIL: Knowledge base content found when it should be disabled!")
            else:
                print("✅ PASS: No knowledge base content (as expected)")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test Case 3: Both Enabled (should return both types of content)
    print("\n3️⃣ Testing: BOTH ENABLED (use_knowledge_store=True, include_wikipedia=True)")
    print("-" * 50)
    
    params_both = {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": True,   # Enable Wikipedia
        "use_knowledge_store": True   # Enable knowledge base
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params_both, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Status: Success")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            print(f"📄 Sources Count: {len(data.get('sources', []))}")
            
            # Check sources
            sources = data.get('sources', [])
            wikipedia_sources = [s for s in sources if s.get('store') == 'wikipedia']
            kb_sources = [s for s in sources if s.get('store') == 'knowledge_base']
            
            print(f"🌐 Wikipedia Sources: {len(wikipedia_sources)}")
            print(f"📚 Knowledge Base Sources: {len(kb_sources)}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test Case 4: Neither Enabled (should return basic content)
    print("\n4️⃣ Testing: NEITHER ENABLED (use_knowledge_store=False, include_wikipedia=False)")
    print("-" * 50)
    
    params_neither = {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": False,  # Disable Wikipedia
        "use_knowledge_store": False  # Disable knowledge base
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params_neither, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Status: Success")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            print(f"📄 Sources Count: {len(data.get('sources', []))}")
            
            # Should have no sources when both are disabled
            if data.get('knowledge_base_used', False) or data.get('wikipedia_used', False):
                print("❌ FAIL: Some content source was used when both should be disabled!")
            else:
                print("✅ PASS: Basic generation (no external sources)")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")
    print("\n💡 Expected Results:")
    print("   - Knowledge Base Only: wikipedia_used=False, knowledge_base_used=True")
    print("   - Wikipedia Only: wikipedia_used=True, knowledge_base_used=False")
    print("   - Both Enabled: Both flags should be True")
    print("   - Neither Enabled: Both flags should be False")

def test_content_difference():
    """Test that the content is actually different between the two modes"""
    
    base_url = "http://localhost:8000"
    
    print("\n🔍 Testing Content Difference")
    print("=" * 40)
    
    # Get content with Wikipedia
    params_wiki = {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": True,
        "use_knowledge_store": False
    }
    
    # Get content without Wikipedia
    params_no_wiki = {
        "subject": "science", 
        "topic": "motion",
        "include_wikipedia": False,
        "use_knowledge_store": True
    }
    
    try:
        response_wiki = requests.get(f"{base_url}/generate_lesson", params=params_wiki, timeout=30)
        response_no_wiki = requests.get(f"{base_url}/generate_lesson", params=params_no_wiki, timeout=30)
        
        if response_wiki.status_code == 200 and response_no_wiki.status_code == 200:
            data_wiki = response_wiki.json()
            data_no_wiki = response_no_wiki.json()
            
            text_wiki = data_wiki.get('text', '')
            text_no_wiki = data_no_wiki.get('text', '')
            
            print(f"📝 Wikipedia content length: {len(text_wiki)} chars")
            print(f"📚 Knowledge base content length: {len(text_no_wiki)} chars")
            
            if text_wiki == text_no_wiki:
                print("❌ FAIL: Content is identical! The fix is not working.")
            else:
                print("✅ PASS: Content is different between Wikipedia and knowledge base modes")
                
            # Show preview of differences
            print(f"\n📖 Wikipedia preview: {text_wiki[:150]}...")
            print(f"📚 Knowledge base preview: {text_no_wiki[:150]}...")
            
        else:
            print("❌ Failed to get responses for comparison")
            
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")

if __name__ == "__main__":
    test_knowledge_base_vs_wikipedia()
    test_content_difference()
