from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime
import json
from bson import json_util

# Load environment variables from .env file
load_dotenv()

# Get MongoDB URI from environment variable
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://blackholeinfiverse1:ImzKJBDjogqox4nQ@user.y9b2fg6.mongodb.net/?retryWrites=true&w=majority&appName=user")

print(f"Connecting to MongoDB with URI: {MONGO_URI}")

# Connect to MongoDB
client = MongoClient(MONGO_URI)

# Test connection
try:
    # The ismaster command is cheap and does not require auth
    client.admin.command('ismaster')
    print("MongoDB connection successful!")
    
    # Use the gurukul database
    db = client["gurukul"]
    
    # Test inserting a document into user_data collection
    print("\nInserting a test document into user_data collection...")
    user_data_collection = db["user_data"]
    
    timestamp = datetime.datetime.now().isoformat()
    
    test_doc = {
        "message": "This is a direct test message from test_post.py",
        "llm": "grok",
        "timestamp": timestamp,
        "type": "chat_message"
    }
    
    result = user_data_collection.insert_one(test_doc)
    print(f"Inserted document with ID: {result.inserted_id}")
    
    # Verify the document was inserted
    inserted_doc = user_data_collection.find_one({"_id": result.inserted_id})
    print("\nVerifying inserted document:")
    inserted_json = json.loads(json_util.dumps(inserted_doc))
    print(json.dumps(inserted_json, indent=2))
    
    # Now test retrieving the latest document with no response
    print("\nRetrieving the latest document with no response...")
    latest_query = user_data_collection.find_one(
        {"type": "chat_message", "response": None},
        sort=[("timestamp", -1)]
    )
    
    if latest_query:
        print("Found latest query:")
        latest_json = json.loads(json_util.dumps(latest_query))
        print(json.dumps(latest_json, indent=2))
        
        # Now add a response to this document
        print("\nAdding a response to this document...")
        response_data = {
            "message": "This is a test response",
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "chat_response",
            "query_id": str(latest_query["_id"]),
            "llm": "grok"
        }
        
        user_data_collection.update_one(
            {"_id": latest_query["_id"]},
            {"$set": {"response": response_data}}
        )
        
        # Verify the response was added
        updated_doc = user_data_collection.find_one({"_id": latest_query["_id"]})
        print("\nVerifying updated document:")
        updated_json = json.loads(json_util.dumps(updated_doc))
        print(json.dumps(updated_json, indent=2))
    else:
        print("No documents found with no response")
    
except Exception as e:
    print(f"Error: {e}")
