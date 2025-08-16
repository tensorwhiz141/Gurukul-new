from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime

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
    for db in client.list_database_names():
        print(f" - {db}")
        
    # Check if our database exists
    if "gurukul" in client.list_database_names():
        db = client["gurukul"]
        print("\nCollections in gurukul database:")
        for collection in db.list_collection_names():
            print(f" - {collection}")
            
        # Count documents in chat_collection
        if "chat_collection" in db.list_collection_names():
            count = db.chat_collection.count_documents({})
            print(f"\nNumber of documents in chat_collection: {count}")
            
            # Show the most recent document
            if count > 0:
                latest = db.chat_collection.find_one(sort=[("timestamp", -1)])
                print("\nMost recent document:")
                print(latest)
                
    # Insert a test document
    print("\nInserting a test document...")
    db = client["gurukul"]
    test_collection = db["chat_collection"]
    
    test_doc = {
        "message": "This is a test message from test_mongo.py",
        "llm": "grok",
        "timestamp": datetime.datetime.now().isoformat(),
        "type": "chat_message"
    }
    
    result = test_collection.insert_one(test_doc)
    print(f"Inserted document with ID: {result.inserted_id}")
    
    # Verify the document was inserted
    inserted_doc = test_collection.find_one({"_id": result.inserted_id})
    print("\nVerifying inserted document:")
    print(inserted_doc)
    
except Exception as e:
    print(f"MongoDB connection failed: {e}")
