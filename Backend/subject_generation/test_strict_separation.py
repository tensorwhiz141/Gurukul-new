#!/usr/bin/env python3
"""
Test script to verify strict separation between knowledge base and Wikipedia data
and ensure fresh generation each time
"""

import requests
import json
import time

def test_strict_separation():
    """Test that knowledge base and Wikipedia data are strictly separated"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Strict Knowledge Base vs Wikipedia Separation")
    print("=" * 70)
    print("Testing with subject='science' and topic='motion'")
    print("=" * 70)
    
    # Test Case 1: KNOWLEDGE BASE ONLY - Should have NO Wikipedia content
    print("\n1️⃣ Testing: KNOWLEDGE BASE ONLY")
    print("   Parameters: use_knowledge_store=True, include_wikipedia=False")
    print("-" * 60)
    
    params_kb_only = {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": False,  # DISABLED
        "use_knowledge_store": True   # ENABLED
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params_kb_only, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            text_content = data.get('text', '').lower()
            
            print(f"✅ Status: Success")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            
            # Check sources
            sources = data.get('sources', [])
            wikipedia_sources = [s for s in sources if s.get('store') == 'wikipedia']
            kb_sources = [s for s in sources if s.get('store') == 'knowledge_base']
            
            print(f"📄 Total Sources: {len(sources)}")
            print(f"📚 Knowledge Base Sources: {len(kb_sources)}")
            print(f"🌐 Wikipedia Sources: {len(wikipedia_sources)}")
            
            # Strict validation
            violations = []
            
            if data.get('wikipedia_used', False):
                violations.append("wikipedia_used flag is True when it should be False")
            
            if wikipedia_sources:
                violations.append(f"Found {len(wikipedia_sources)} Wikipedia sources when none should exist")
            
            # Check for Wikipedia references in text
            wikipedia_phrases = [
                "wikipedia", "according to wikipedia", "from wikipedia", 
                "wikipedia states", "wikipedia article", "based on wikipedia"
            ]
            
            found_phrases = [phrase for phrase in wikipedia_phrases if phrase in text_content]
            if found_phrases:
                violations.append(f"Found Wikipedia references in text: {found_phrases}")
            
            if violations:
                print("❌ VIOLATIONS FOUND:")
                for violation in violations:
                    print(f"   - {violation}")
                print(f"📄 Content preview: {text_content[:200]}...")
            else:
                print("✅ PASS: Strict knowledge base only - no Wikipedia contamination")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test Case 2: WIKIPEDIA ONLY - Should have NO knowledge base content
    print("\n2️⃣ Testing: WIKIPEDIA ONLY")
    print("   Parameters: use_knowledge_store=False, include_wikipedia=True")
    print("-" * 60)
    
    params_wiki_only = {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": True,   # ENABLED
        "use_knowledge_store": False  # DISABLED
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params_wiki_only, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            text_content = data.get('text', '').lower()
            
            print(f"✅ Status: Success")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"🌐 Wikipedia Used: {data.get('wikipedia_used', False)}")
            
            # Check sources
            sources = data.get('sources', [])
            wikipedia_sources = [s for s in sources if s.get('store') == 'wikipedia']
            kb_sources = [s for s in sources if s.get('store') == 'knowledge_base']
            
            print(f"📄 Total Sources: {len(sources)}")
            print(f"📚 Knowledge Base Sources: {len(kb_sources)}")
            print(f"🌐 Wikipedia Sources: {len(wikipedia_sources)}")
            
            # Strict validation
            violations = []
            
            if data.get('knowledge_base_used', False):
                violations.append("knowledge_base_used flag is True when it should be False")
            
            if kb_sources:
                violations.append(f"Found {len(kb_sources)} knowledge base sources when none should exist")
            
            if violations:
                print("❌ VIOLATIONS FOUND:")
                for violation in violations:
                    print(f"   - {violation}")
            else:
                print("✅ PASS: Strict Wikipedia only - no knowledge base contamination")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_fresh_generation():
    """Test that each request generates fresh content"""
    
    base_url = "http://localhost:8000"
    
    print("\n🔄 Testing Fresh Generation (No Repetitive Responses)")
    print("=" * 60)
    
    params = {
        "subject": "science",
        "topic": "motion",
        "include_wikipedia": False,
        "use_knowledge_store": True
    }
    
    responses = []
    
    print("Making 3 identical requests to check for repetitive responses...")
    
    for i in range(3):
        try:
            print(f"\n🔄 Request {i+1}/3...")
            response = requests.get(f"{base_url}/generate_lesson", params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                text_content = data.get('text', '')
                responses.append({
                    'request': i+1,
                    'text': text_content,
                    'length': len(text_content),
                    'sources': len(data.get('sources', []))
                })
                print(f"✅ Success - Length: {len(text_content)} chars, Sources: {len(data.get('sources', []))}")
            else:
                print(f"❌ Error: {response.status_code}")
                
            # Small delay between requests
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Request {i+1} failed: {e}")
    
    # Analyze responses
    if len(responses) >= 2:
        print(f"\n📊 Analysis of {len(responses)} responses:")
        
        # Check if all responses are identical
        first_text = responses[0]['text']
        identical_count = sum(1 for r in responses if r['text'] == first_text)
        
        if identical_count == len(responses):
            print("❌ FAIL: All responses are identical (repetitive responses detected)")
            print("💡 This indicates caching is still active")
        else:
            print("✅ PASS: Responses are different (fresh generation working)")
            
        # Show response details
        for i, resp in enumerate(responses):
            print(f"   Response {resp['request']}: {resp['length']} chars, {resp['sources']} sources")
            print(f"      Preview: {resp['text'][:100]}...")
            
    else:
        print("❌ Not enough successful responses to analyze")

def test_content_quality():
    """Test that different parameter combinations produce meaningfully different content"""
    
    base_url = "http://localhost:8000"
    
    print("\n🎯 Testing Content Quality and Differentiation")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Knowledge Base Only",
            "params": {"subject": "science", "topic": "motion", "include_wikipedia": False, "use_knowledge_store": True}
        },
        {
            "name": "Wikipedia Only", 
            "params": {"subject": "science", "topic": "motion", "include_wikipedia": True, "use_knowledge_store": False}
        },
        {
            "name": "Basic Generation",
            "params": {"subject": "science", "topic": "motion", "include_wikipedia": False, "use_knowledge_store": False}
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        try:
            print(f"\n🧪 Testing: {test_case['name']}")
            response = requests.get(f"{base_url}/generate_lesson", params=test_case['params'], timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results[test_case['name']] = {
                    'text': data.get('text', ''),
                    'kb_used': data.get('knowledge_base_used', False),
                    'wiki_used': data.get('wikipedia_used', False),
                    'sources': len(data.get('sources', []))
                }
                print(f"✅ Success - KB: {results[test_case['name']]['kb_used']}, Wiki: {results[test_case['name']]['wiki_used']}")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Compare results
    if len(results) >= 2:
        print(f"\n📊 Content Comparison:")
        
        texts = [results[name]['text'] for name in results.keys()]
        unique_texts = len(set(texts))
        
        print(f"   Total responses: {len(results)}")
        print(f"   Unique content: {unique_texts}")
        
        if unique_texts == len(results):
            print("✅ PASS: All responses have different content")
        else:
            print("❌ FAIL: Some responses have identical content")
            
        # Show details
        for name, result in results.items():
            print(f"\n   {name}:")
            print(f"      Length: {len(result['text'])} chars")
            print(f"      Sources: {result['sources']}")
            print(f"      Preview: {result['text'][:150]}...")

if __name__ == "__main__":
    test_strict_separation()
    test_fresh_generation()
    test_content_quality()
    
    print("\n" + "=" * 70)
    print("🏁 Testing completed!")
    print("\n💡 Expected Results:")
    print("   - Knowledge Base Only: No Wikipedia content at all")
    print("   - Wikipedia Only: No knowledge base content at all") 
    print("   - Fresh Generation: Different content each time")
    print("   - Content Quality: Meaningful differences between modes")
