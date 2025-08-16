#!/usr/bin/env python3
"""
Test script for enhanced lesson generation with knowledge base and Wikipedia integration
"""

import requests
import json
import sys
import os

def test_lesson_generation():
    """Test the enhanced /generate_lesson endpoint"""
    
    base_url = "http://localhost:8001"  # Base_backend runs on port 8001
    
    # Test cases
    test_cases = [
        {
            "name": "Knowledge Base + Wikipedia",
            "params": {
                "subject": "science",
                "topic": "environment",
                "include_wikipedia": True,
                "use_knowledge_store": True
            }
        },
        {
            "name": "Wikipedia Only",
            "params": {
                "subject": "mathematics",
                "topic": "algebra",
                "include_wikipedia": True,
                "use_knowledge_store": False
            }
        },
        {
            "name": "Knowledge Base Only",
            "params": {
                "subject": "ved",
                "topic": "sound",
                "include_wikipedia": False,
                "use_knowledge_store": True
            }
        },
        {
            "name": "Basic Generation",
            "params": {
                "subject": "english",
                "topic": "grammar",
                "include_wikipedia": False,
                "use_knowledge_store": False
            }
        }
    ]
    
    print("🧪 Testing Enhanced Lesson Generation")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Make GET request
            response = requests.get(f"{base_url}/generate_lesson", params=test_case['params'], timeout=60)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Print key information
                print(f"✅ Success!")
                print(f"Title: {data.get('title', 'N/A')}")
                print(f"Level: {data.get('level', 'N/A')}")
                print(f"Knowledge Base Used: {data.get('knowledge_base_used', False)}")
                print(f"Wikipedia Used: {data.get('wikipedia_used', False)}")
                print(f"Sources Count: {len(data.get('sources', []))}")
                
                # Print sources
                if data.get('sources'):
                    print("Sources:")
                    for j, source in enumerate(data['sources'][:2], 1):  # Show first 2 sources
                        print(f"  {j}. {source.get('source', 'Unknown')} ({source.get('store', 'N/A')})")
                
                # Print text preview
                text_content = data.get('text', '')
                if text_content:
                    preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
                    print(f"Text Preview: {preview}")
                
                # Print quiz info
                quiz = data.get('quiz', [])
                print(f"Quiz Questions: {len(quiz)}")
                
            else:
                print(f"❌ Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error Details: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"Error Response: {response.text}")
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")

def test_specific_case():
    """Test a specific case that was mentioned in the issue"""
    
    base_url = "http://localhost:8001"
    
    print("🔍 Testing Specific Case: Science Environment with Knowledge Base")
    print("=" * 60)
    
    params = {
        "subject": "science",
        "topic": "environment",
        "include_wikipedia": False,  # Disable Wikipedia
        "use_knowledge_store": True  # Enable Knowledge Base
    }
    
    try:
        response = requests.get(f"{base_url}/generate_lesson", params=params, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Success!")
            print(f"Title: {data.get('title', 'N/A')}")
            print(f"Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"Wikipedia Used: {data.get('wikipedia_used', False)}")
            
            # Check if we're getting knowledge base content vs Wikipedia content
            sources = data.get('sources', [])
            print(f"Sources Count: {len(sources)}")
            
            for i, source in enumerate(sources, 1):
                store_type = source.get('store', 'unknown')
                source_name = source.get('source', 'Unknown')
                print(f"  {i}. Store: {store_type}, Source: {source_name}")
            
            # Check the content
            text_content = data.get('text', '')
            if 'Wikipedia' in text_content and not data.get('wikipedia_used', False):
                print("⚠️  WARNING: Content contains Wikipedia reference but wikipedia_used is False")
            elif data.get('knowledge_base_used', False):
                print("✅ Knowledge base content is being used correctly")
            else:
                print("ℹ️  Using basic generation (no knowledge base content found)")
                
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error Details: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Error Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "specific":
        test_specific_case()
    else:
        test_lesson_generation()
