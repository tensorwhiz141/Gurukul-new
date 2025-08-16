#!/usr/bin/env python3
"""
Connection test script for MongoDB Atlas and Groq API
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mongodb_connection():
    """Test MongoDB Atlas connection"""
    print("🔍 Testing MongoDB Atlas Connection...")
    
    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        
        # Get MongoDB URI from environment
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            print("❌ MONGODB_URI not found in environment variables")
            return False
            
        print(f"📡 Connecting to: {mongodb_uri[:50]}...")
        
        # Create MongoDB client with server API
        client = MongoClient(mongodb_uri, server_api=ServerApi('1'))
        
        # Test the connection
        client.admin.command('ping')
        print("✅ MongoDB Atlas connection successful!")
        
        # Test database operations
        db = client.financial_simulation
        collection = db.test_collection
        
        # Insert a test document
        test_doc = {"test": "connection", "status": "success"}
        result = collection.insert_one(test_doc)
        print(f"✅ Test document inserted with ID: {result.inserted_id}")
        
        # Read the test document
        found_doc = collection.find_one({"_id": result.inserted_id})
        if found_doc:
            print("✅ Test document retrieved successfully")
        
        # Clean up test document
        collection.delete_one({"_id": result.inserted_id})
        print("✅ Test document cleaned up")
        
        # Close connection
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

def test_groq_api():
    """Test Groq API connection"""
    print("\n🔍 Testing Groq API Connection...")
    
    try:
        from langchain_groq import ChatGroq
        
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("❌ GROQ_API_KEY not found in environment variables")
            return False
            
        print(f"🔑 Using API key: {groq_api_key[:20]}...")
        
        # Initialize Groq client
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.1
        )
        
        # Test with a simple query
        test_message = "Hello, this is a test. Please respond with 'Groq API working'."
        response = llm.invoke(test_message)
        
        print(f"✅ Groq API response: {response.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Groq API connection failed: {e}")
        return False

def test_environment_variables():
    """Test if environment variables are loaded correctly"""
    print("\n🔍 Testing Environment Variables...")
    
    required_vars = [
        "MONGODB_URI",
        "GROQ_API_KEY",
        "AGENTOPS_API_KEY",
        "HOST",
        "PORT"
    ]
    
    all_good = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "URI" in var:
                display_value = f"{value[:20]}..." if len(value) > 20 else value
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: Not set")
            all_good = False
    
    return all_good

def main():
    """Run all connection tests"""
    print("🚀 Starting Connection Tests for Financial Simulator Backend\n")
    
    # Test environment variables
    env_ok = test_environment_variables()
    
    # Test MongoDB connection
    mongodb_ok = test_mongodb_connection()
    
    # Test Groq API
    groq_ok = test_groq_api()
    
    # Summary
    print("\n📊 Connection Test Summary:")
    print(f"Environment Variables: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"MongoDB Atlas: {'✅ PASS' if mongodb_ok else '❌ FAIL'}")
    print(f"Groq API: {'✅ PASS' if groq_ok else '❌ FAIL'}")
    
    if all([env_ok, mongodb_ok, groq_ok]):
        print("\n🎉 All connections successful! The backend should work properly.")
        return 0
    else:
        print("\n⚠️ Some connections failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
