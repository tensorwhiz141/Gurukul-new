#!/usr/bin/env python3
"""
Test script to verify that "According to Wikipedia:" is no longer hardcoded
and only appears when Wikipedia is actually requested
"""

import requests
import json

def test_wikipedia_hardcode_removal():
    """Test that Wikipedia text only appears when Wikipedia is enabled"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Wikipedia Hardcode Removal")
    print("=" * 60)
    print("Testing with subject='science' and topic='motion'")
    print("=" * 60)
    
    # Test Case 1: Wikipedia DISABLED - should NOT contain "According to Wikipedia"
    print("\n1️⃣ Testing: Wikipedia DISABLED (include_wikipedia=False)")
    print("-" * 50)
    
    params_no_wiki = {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": False,
        "use_knowledge_store": True
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params_no_wiki, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            text_content = data.get('text', '').lower()
            
            print(f"✅ Status: Success")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            
            # Check for hardcoded Wikipedia references
            wikipedia_phrases = [
                "according to wikipedia",
                "wikipedia states",
                "wikipedia says",
                "from wikipedia",
                "wikipedia article",
                "wikipedia:"
            ]
            
            found_wikipedia_refs = []
            for phrase in wikipedia_phrases:
                if phrase in text_content:
                    found_wikipedia_refs.append(phrase)
            
            if found_wikipedia_refs:
                print(f"❌ FAIL: Found Wikipedia references when disabled: {found_wikipedia_refs}")
                print(f"📄 Content preview: {text_content[:200]}...")
            else:
                print("✅ PASS: No Wikipedia references found (as expected)")
                
            # Check sources
            sources = data.get('sources', [])
            wikipedia_sources = [s for s in sources if 'wikipedia' in s.get('source', '').lower()]
            
            if wikipedia_sources:
                print(f"❌ FAIL: Found Wikipedia sources when disabled: {len(wikipedia_sources)}")
            else:
                print("✅ PASS: No Wikipedia sources (as expected)")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test Case 2: Wikipedia ENABLED - may contain Wikipedia references
    print("\n2️⃣ Testing: Wikipedia ENABLED (include_wikipedia=True)")
    print("-" * 50)
    
    params_with_wiki = {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": True,
        "use_knowledge_store": False
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params_with_wiki, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            text_content = data.get('text', '').lower()
            
            print(f"✅ Status: Success")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            
            # Check sources
            sources = data.get('sources', [])
            wikipedia_sources = [s for s in sources if 'wikipedia' in s.get('source', '').lower()]
            
            print(f"📄 Wikipedia Sources: {len(wikipedia_sources)}")
            
            if data.get('wikipedia_used', False):
                print("✅ PASS: Wikipedia is properly enabled and used")
            else:
                print("⚠️  INFO: Wikipedia enabled but no content found (may be normal)")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")
    print("\n💡 Expected Results:")
    print("   - Wikipedia DISABLED: No 'According to Wikipedia' text")
    print("   - Wikipedia ENABLED: May contain Wikipedia content if available")

def test_content_comparison():
    """Compare content between Wikipedia enabled and disabled"""
    
    base_url = "http://localhost:8000"
    
    print("\n🔍 Testing Content Difference (Wikipedia On vs Off)")
    print("=" * 50)
    
    # Test with same subject/topic but different Wikipedia settings
    test_params = [
        {
            "name": "Wikipedia OFF",
            "params": {
                "subject": "science",
                "topic": "motion", 
                "include_wikipedia": False,
                "use_knowledge_store": True
            }
        },
        {
            "name": "Wikipedia ON",
            "params": {
                "subject": "science",
                "topic": "motion",
                "include_wikipedia": True,
                "use_knowledge_store": False
            }
        }
    ]
    
    results = {}
    
    for test in test_params:
        try:
            response = requests.get(f"{base_url}/generate_lesson", params=test["params"], timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results[test["name"]] = {
                    "text": data.get('text', ''),
                    "wikipedia_used": data.get('wikipedia_used', False),
                    "sources": len(data.get('sources', []))
                }
                print(f"✅ {test['name']}: Success")
            else:
                print(f"❌ {test['name']}: Error {response.status_code}")
                results[test["name"]] = None
                
        except Exception as e:
            print(f"❌ {test['name']}: Failed - {e}")
            results[test["name"]] = None
    
    # Compare results
    if results.get("Wikipedia OFF") and results.get("Wikipedia ON"):
        off_text = results["Wikipedia OFF"]["text"].lower()
        on_text = results["Wikipedia ON"]["text"].lower()
        
        print(f"\n📊 Comparison Results:")
        print(f"   Wikipedia OFF - Length: {len(off_text)} chars, Sources: {results['Wikipedia OFF']['sources']}")
        print(f"   Wikipedia ON  - Length: {len(on_text)} chars, Sources: {results['Wikipedia ON']['sources']}")
        
        if off_text == on_text:
            print("❌ FAIL: Content is identical!")
        else:
            print("✅ PASS: Content is different")
            
        # Check for Wikipedia references in OFF mode
        if "wikipedia" in off_text:
            print("❌ FAIL: Wikipedia references found in OFF mode")
        else:
            print("✅ PASS: No Wikipedia references in OFF mode")
            
        # Show previews
        print(f"\n📖 Wikipedia OFF preview: {off_text[:150]}...")
        print(f"📖 Wikipedia ON preview: {on_text[:150]}...")
    else:
        print("❌ Could not compare - one or both requests failed")

if __name__ == "__main__":
    test_wikipedia_hardcode_removal()
    test_content_comparison()
