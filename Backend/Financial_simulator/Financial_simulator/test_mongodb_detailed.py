#!/usr/bin/env python3
"""
Detailed MongoDB Atlas connection test with multiple URI formats
"""

import os
import urllib.parse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mongodb_with_different_formats():
    """Test MongoDB with different URI formats"""
    print("🔍 Testing MongoDB Atlas with different URI formats...")
    
    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        
        # Original URI from .env
        original_uri = os.getenv("MONGODB_URI")
        print(f"📡 Original URI: {original_uri}")
        
        # Test different URI formats
        test_uris = [
            # Original
            original_uri,
            
            # Without database name
            "mongodb+srv://blackhole:sidd%232002@cluster0.tjy5a.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
            
            # With different encoding
            "mongodb+srv://blackhole:sidd%23002@cluster0.tjy5a.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
            
            # URL encoded password
            f"mongodb+srv://blackhole:{urllib.parse.quote_plus('sidd#2002')}@cluster0.tjy5a.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
            
            # Simple format
            "mongodb+srv://blackhole:sidd%232002@cluster0.tjy5a.mongodb.net/",
        ]
        
        for i, uri in enumerate(test_uris, 1):
            print(f"\n🧪 Test {i}: {uri[:60]}...")
            try:
                client = MongoClient(uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                print(f"✅ Test {i} SUCCESS!")
                client.close()
                return uri
            except Exception as e:
                print(f"❌ Test {i} FAILED: {e}")
                
        return None
        
    except Exception as e:
        print(f"❌ MongoDB test failed: {e}")
        return None

def test_credentials_manually():
    """Test with manually constructed credentials"""
    print("\n🔍 Testing with manually constructed credentials...")
    
    # Credentials
    username = "blackhole"
    password = "sidd#2002"  # Original password
    cluster = "cluster0.tjy5a.mongodb.net"
    
    # URL encode the password
    encoded_password = urllib.parse.quote_plus(password)
    print(f"🔑 Original password: {password}")
    print(f"🔑 URL encoded password: {encoded_password}")
    
    # Construct URI
    uri = f"mongodb+srv://{username}:{encoded_password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"
    print(f"📡 Constructed URI: {uri}")
    
    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        
        client = MongoClient(uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=10000)
        client.admin.command('ping')
        print("✅ Manual construction SUCCESS!")
        client.close()
        return uri
    except Exception as e:
        print(f"❌ Manual construction FAILED: {e}")
        return None

def main():
    """Run detailed MongoDB tests"""
    print("🚀 Starting Detailed MongoDB Atlas Connection Tests\n")
    
    # Test with different formats
    working_uri = test_mongodb_with_different_formats()
    
    if not working_uri:
        # Test with manual construction
        working_uri = test_credentials_manually()
    
    if working_uri:
        print(f"\n🎉 SUCCESS! Working URI found:")
        print(f"📝 Use this URI in your .env file:")
        print(f"MONGODB_URI={working_uri}")
        
        # Update .env file
        try:
            with open('.env', 'r') as f:
                content = f.read()
            
            # Replace the MongoDB URI line
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('MONGODB_URI='):
                    lines[i] = f"MONGODB_URI={working_uri}"
                    break
            
            with open('.env', 'w') as f:
                f.write('\n'.join(lines))
            
            print("✅ .env file updated with working URI!")
            
        except Exception as e:
            print(f"⚠️ Could not update .env file: {e}")
            
    else:
        print("\n❌ No working URI found. Please check:")
        print("1. MongoDB Atlas cluster is running")
        print("2. Username and password are correct")
        print("3. IP address is whitelisted (or 0.0.0.0/0 for all)")
        print("4. Database user has proper permissions")

if __name__ == "__main__":
    main()
