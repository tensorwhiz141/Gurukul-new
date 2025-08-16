from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://blackholeinfiverse1:ImzKJBDjogqox4nQ@user.y9b2fg6.mongodb.net/?retryWrites=true&w=majority&appName=user")
print(f"Connecting to MongoDB with URI: {MONGO_URI}")

client = MongoClient(MONGO_URI)
db = client["gurukul"]
user_collection = db["user_data"]

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define chat message model
class ChatMessage(BaseModel):
    message: str
    llm: str = Field(..., pattern="^(grok|llama|chatgpt|uniguru)$")

# Chat endpoints
@app.post("/test-chat")
async def test_chat(chat: ChatMessage):
    try:
        # Create record
        timestamp = datetime.datetime.now().isoformat()
        query_record = {
            "message": chat.message,
            "llm": chat.llm,
            "timestamp": timestamp,
            "type": "chat_message"
        }
        
        # Insert into MongoDB
        result = user_collection.insert_one(query_record)
        
        # Return success response
        return {
            "status": "success",
            "message": "Message stored successfully",
            "data": {
                "message": chat.message,
                "llm": chat.llm,
                "timestamp": timestamp,
                "id": str(result.inserted_id)
            }
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-get")
async def test_get():
    try:
        # Get the latest message
        latest = user_collection.find_one(sort=[("timestamp", -1)])
        
        if latest:
            # Convert ObjectId to string
            latest["_id"] = str(latest["_id"])
            return {
                "status": "success",
                "data": latest
            }
        else:
            return {
                "status": "error",
                "message": "No messages found"
            }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="192.168.0.99", port=8001)  # Using a different port
    except Exception as e:
        print(f"Failed to start server: {e}")
        raise
