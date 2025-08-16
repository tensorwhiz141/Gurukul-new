from pymongo import MongoClient
import os
from dotenv import load_dotenv
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
    
    # List databases
    print("\nAvailable databases:")
    for db_name in client.list_database_names():
        print(f" - {db_name}")
    
    # Use the gurukul database
    db = client["gurukul"]
    
    # List collections
    print("\nCollections in database:")
    for collection_name in db.list_collection_names():
        print(f" - {collection_name}")
    
    # Check each collection
    collections_to_check = ["user_data", "User", "lectures", "subjects"]
    
    for collection_name in collections_to_check:
        if collection_name in db.list_collection_names():
            collection = db[collection_name]
            count = collection.count_documents({})
            print(f"\nCollection '{collection_name}' exists with {count} documents")
            
            # Show a sample document if available
            if count > 0:
                sample = collection.find_one()
                print(f"Sample document from '{collection_name}':")
                # Convert MongoDB document to JSON string
                sample_json = json.loads(json_util.dumps(sample))
                print(json.dumps(sample_json, indent=2))
        else:
            print(f"\nCollection '{collection_name}' does not exist in the database")
    
    # Test inserting a document into user_data collection
    print("\nInserting a test document into user_data collection...")
    user_data_collection = db["user_data"]
    
    test_doc = {
        "message": "This is a test message from test_collections.py",
        "llm": "grok",
        "timestamp": "2023-07-01T12:00:00Z",
        "type": "chat_message"
    }
    
    result = user_data_collection.insert_one(test_doc)
    print(f"Inserted document with ID: {result.inserted_id}")
    
    # Verify the document was inserted
    inserted_doc = user_data_collection.find_one({"_id": result.inserted_id})
    print("\nVerifying inserted document:")
    inserted_json = json.loads(json_util.dumps(inserted_doc))
    print(json.dumps(inserted_json, indent=2))
    
except Exception as e:
    print(f"Error: {e}")
