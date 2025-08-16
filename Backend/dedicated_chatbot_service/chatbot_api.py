"""
Dedicated Chatbot Service - Port 8001
Separated from other backend services to avoid conflicts
"""

import os
import sys
import shutil
import time
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
import uvicorn
import logging

# Add parent directory to path to import required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Base_backend'))
from llm_service import LLMService

# Define streaming TTS function (always available)
def text_to_speech_stream(text):
    """Generate TTS audio and return as bytes stream"""
    try:
        from gtts import gTTS
        import io

        # Generate TTS audio in memory
        tts = gTTS(text=text, lang="en")

        # Create in-memory buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return audio_buffer.getvalue()
    except Exception as e:
        print(f"TTS Streaming Error: {e}")
        return None

# Import PDF and image processing functions
try:
    from rag import parse_pdf, build_qa_agent, extract_text_easyocr, text_to_speech
    # Use extract_text_easyocr as process_image_with_ocr
    process_image_with_ocr = extract_text_easyocr
    print("✅ Successfully imported processing modules from rag.py")
except ImportError as e:
    print(f"⚠️ Some processing modules not available: {e}")
    # Define fallback functions
    def parse_pdf(path):
        return {"body": "PDF processing not available"}
    def build_qa_agent(content, groq_api_key):
        class MockAgent:
            def invoke(self, query):
                return {"result": "PDF processing not available"}
        return MockAgent()
    def process_image_with_ocr(path):
        return "Image processing not available"

    # Keep legacy function for backward compatibility (deprecated)
    def text_to_speech(text, file_prefix="output"):
        try:
            from gtts import gTTS
            import time
            import os

            # Create static directory if it doesn't exist
            static_dir = "static"
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(static_dir, f"{file_prefix}_{timestamp}.mp3")

            tts = gTTS(text=text, lang="en")
            tts.save(output_file)

            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return output_file
            else:
                return None
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '..', 'Base_backend', '.env')
load_dotenv(env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://blackholeinfiverse1:ImzKJBDjogqox4nQ@user.y9b2fg6.mongodb.net/?retryWrites=true&w=majority&appName=user")

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI)
    client.admin.command('ismaster')
    logger.info("✅ MongoDB connection successful!")
    
    db = client["gurukul"]
    # Use a separate collection for dedicated chatbot to avoid conflicts
    chat_collection = db["dedicated_chat_messages"]
    
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    raise

# Initialize LLM Service
llm_service = LLMService()

# Create temporary directory for file processing
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Global variables for storing latest processing results
pdf_response = None
image_response = None

# FastAPI App
app = FastAPI(
    title="Dedicated Chatbot Service",
    description="Standalone chatbot service running on port 8001",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class ChatMessage(BaseModel):
    message: str
    llm: Optional[str] = "grok"
    type: str = "chat_message"

class ChatResponse(BaseModel):
    message: str
    timestamp: str
    type: str = "chat_response"
    user_id: str

class PDFResponse(BaseModel):
    summary: str
    audio_file: str
    timestamp: str

class ImageResponse(BaseModel):
    summary: str
    audio_file: str
    timestamp: str

# Health Check Endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        client.admin.command('ismaster')
        
        # Test LLM service
        test_providers = llm_service.test_providers()
        
        return {
            "status": "healthy",
            "service": "Dedicated Chatbot Service",
            "port": 8001,
            "mongodb": "connected",
            "llm_providers": test_providers,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

# Chat Endpoints
@app.post("/chatpost")
async def receive_chat_message(chat: ChatMessage, user_id: str = "guest-user"):
    """
    Receive and store chat message from frontend
    """
    try:
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Create message record
        message_record = {
            "message": chat.message,
            "timestamp": timestamp,
            "type": chat.type,
            "user_id": user_id,
            "llm_model": chat.llm,
            "response": None,  # Will be filled when response is generated
            "status": "pending"
        }
        
        # Store in MongoDB
        result = chat_collection.insert_one(message_record)
        message_record["_id"] = str(result.inserted_id)
        
        logger.info(f"📝 Received message from user {user_id}: {chat.message}")
        
        return {
            "status": "success",
            "message": "Query received and stored",
            "data": {
                "id": str(result.inserted_id),
                "user_id": user_id,
                "timestamp": timestamp
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error storing chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store message: {str(e)}")

@app.get("/chatbot")
async def get_chat_response(user_id: str = "guest-user"):
    """
    Generate and return AI response for the latest user message
    """
    try:
        # Find the latest pending message for this user
        latest_query = chat_collection.find_one(
            {
                "user_id": user_id,
                "type": "chat_message", 
                "response": None,
                "status": "pending"
            }, 
            sort=[("timestamp", -1)]
        )
        
        if not latest_query:
            logger.warning(f"⚠️ No pending queries found for user: {user_id}")
            return {"error": "No queries yet"}
        
        query_message = latest_query["message"]
        llm_model = latest_query.get("llm_model", "grok")
        
        logger.info(f"🤖 Processing query for user {user_id}: {query_message}")
        
        # Generate AI response using LLM service
        try:
            if llm_model == "grok":
                # Use Groq as primary
                ai_response = llm_service.generate_response(query_message, preferred_provider="groq")
            else:
                # Use default provider selection
                ai_response = llm_service.generate_response(query_message)
                
        except Exception as llm_error:
            logger.error(f"❌ LLM generation failed: {llm_error}")
            ai_response = "I apologize, but I'm experiencing technical difficulties right now. Please try again in a few moments."
        
        # Create response data
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        response_data = {
            "message": ai_response,
            "timestamp": timestamp,
            "type": "chat_response",
            "user_id": user_id,
            "llm_model": llm_model
        }
        
        # Update the original message with the response
        chat_collection.update_one(
            {"_id": latest_query["_id"]},
            {
                "$set": {
                    "response": response_data,
                    "status": "completed"
                }
            }
        )
        
        logger.info(f"✅ Generated response for user {user_id}")
        
        return {
            "_id": str(latest_query["_id"]),
            "query": query_message,
            "response": response_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating chat response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.get("/chat-history")
async def get_chat_history(user_id: str = "guest-user", limit: int = 50):
    """
    Get chat history for a user
    """
    try:
        # Get completed chat messages for the user
        messages = list(chat_collection.find(
            {
                "user_id": user_id,
                "status": "completed"
            },
            sort=[("timestamp", -1)],
            limit=limit
        ))
        
        # Format messages for frontend
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "id": str(msg["_id"]),
                "user_message": msg["message"],
                "ai_response": msg["response"]["message"] if msg["response"] else None,
                "timestamp": msg["timestamp"],
                "llm_model": msg.get("llm_model", "unknown")
            })
        
        return {
            "user_id": user_id,
            "messages": formatted_messages,
            "total": len(formatted_messages)
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

# PDF Processing Endpoints
@app.post("/process-pdf", response_model=PDFResponse)
async def process_pdf(file: UploadFile = File(...)):
    """Process PDF file and generate summary with audio"""
    global pdf_response
    temp_pdf_path = ""

    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Save uploaded file temporarily
        temp_pdf_path = os.path.join(TEMP_DIR, f"temp_pdf_{time.strftime('%Y%m%d_%H%M%S')}.pdf")
        with open(temp_pdf_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Parse PDF content
        structured_data = parse_pdf(temp_pdf_path)
        if not structured_data["body"]:
            raise HTTPException(status_code=400, detail="Failed to parse PDF content")

        # Generate summary using QA agent
        query = "give me detail summary of this pdf"
        groq_api_key = os.getenv("GROQ_API_KEY")
        agent = build_qa_agent([structured_data["body"]], groq_api_key=groq_api_key)
        result = agent.invoke({"query": query})
        answer = result["result"]

        # Generate audio
        audio_file = text_to_speech(answer, file_prefix="output_pdf")
        audio_url = f"/static/{os.path.basename(audio_file)}" if audio_file else "No audio generated"

        # Create response
        timestamp = datetime.now(timezone.utc).isoformat()
        pdf_response = PDFResponse(
            summary=answer,
            audio_file=audio_url,
            timestamp=timestamp
        )

        logger.info(f"✅ PDF processed successfully: {file.filename}")
        return pdf_response

    except Exception as e:
        logger.error(f"❌ Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")

@app.post("/process-img", response_model=ImageResponse)
async def process_image(file: UploadFile = File(...)):
    """Process image file and generate summary with audio"""
    global image_response
    temp_image_path = ""

    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are allowed")

        # Save uploaded file temporarily
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        temp_image_path = os.path.join(TEMP_DIR, f"temp_image_{time.strftime('%Y%m%d_%H%M%S')}.{file_extension}")
        with open(temp_image_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Process image with OCR
        ocr_text = process_image_with_ocr(temp_image_path)

        if not ocr_text or ocr_text.strip() == "":
            answer = "No readable text found in the image."
        else:
            # Generate summary using QA agent
            query = "give me detail summary of this image"
            groq_api_key = os.getenv("GROQ_API_KEY")
            agent = build_qa_agent([ocr_text], groq_api_key=groq_api_key)
            result = agent.invoke({"query": query})
            answer = result["result"]

        # Generate audio
        audio_file = text_to_speech(answer, file_prefix="output_image")
        audio_url = f"/static/{os.path.basename(audio_file)}" if audio_file else "No audio generated"

        # Create response
        timestamp = datetime.now(timezone.utc).isoformat()
        image_response = ImageResponse(
            summary=answer,
            audio_file=audio_url,
            timestamp=timestamp
        )

        logger.info(f"✅ Image processed successfully: {file.filename}")
        return image_response

    except Exception as e:
        logger.error(f"❌ Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")

@app.get("/summarize-pdf", response_model=PDFResponse)
async def summarize_pdf():
    """Get the latest PDF summary"""
    if pdf_response is None:
        raise HTTPException(status_code=404, detail="No PDF has been processed yet.")
    return pdf_response

@app.get("/summarize-img", response_model=ImageResponse)
async def summarize_image():
    """Get the latest image summary"""
    if image_response is None:
        raise HTTPException(status_code=404, detail="No image has been processed yet.")
    return image_response

# Streaming TTS Endpoint (New)
@app.post("/tts/stream")
async def generate_tts_stream(request: dict):
    """Generate TTS audio and stream directly without saving to disk"""
    try:
        text = request.get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        # Generate audio stream
        audio_data = text_to_speech_stream(text)

        if not audio_data:
            raise HTTPException(status_code=500, detail="Failed to generate audio stream")

        # Create streaming response
        def generate_audio():
            yield audio_data

        return StreamingResponse(
            generate_audio(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=tts_audio.mp3",
                "Cache-Control": "no-cache",
                "X-Text-Length": str(len(text)),
                "X-Timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    except Exception as e:
        logger.error(f"TTS streaming error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS streaming failed: {str(e)}")

# TTS Endpoint (Legacy - for backward compatibility)
@app.post("/tts")
async def generate_tts(request: dict):
    """Generate TTS audio from text using Google TTS (Legacy - saves to file)"""
    try:
        text = request.get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        # Generate audio file
        audio_file = text_to_speech(text, file_prefix="chatbot_tts")

        if not audio_file or not os.path.exists(audio_file):
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        # Return audio file info
        filename = os.path.basename(audio_file)
        file_size = os.path.getsize(audio_file)

        return {
            "status": "success",
            "message": "Audio generated successfully",
            "audio_url": f"/static/{filename}",
            "filename": filename,
            "file_size": file_size,
            "text_length": len(text),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

# Static file serving for audio files
@app.get("/static/{filename}")
async def serve_audio_file(filename: str):
    """Serve generated audio files"""
    try:
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        file_path = os.path.join(temp_dir, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='audio/mpeg',
            headers={"Cache-Control": "public, max-age=3600"}
        )

    except Exception as e:
        logger.error(f"Error serving audio file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve audio: {str(e)}")

# Test endpoint for debugging
@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify service is working"""
    return {
        "status": "working",
        "service": "Dedicated Chatbot Service",
        "port": 8001,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "Chatbot service is running correctly!"
    }

if __name__ == "__main__":
    import socket
    import time

    def check_port_available(port, host="127.0.0.1"):
        """Check if a port is available for binding"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return True
        except OSError:
            return False

    def find_available_port(start_port=8001, max_attempts=10):
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            if check_port_available(port):
                return port
        return None

    # Check if port 8001 is available
    target_port = 8001
    if not check_port_available(target_port):
        logger.warning(f"⚠️ Port {target_port} is already in use. Searching for alternative...")

        # Try to find an available port
        alternative_port = find_available_port(8001, 10)
        if alternative_port:
            target_port = alternative_port
            logger.info(f"🔄 Using alternative port: {target_port}")
        else:
            logger.error("❌ No available ports found in range 8001-8010")
            logger.info("💡 Trying to kill process using port 8001...")

            # Try to kill process using port 8001
            try:
                import subprocess
                result = subprocess.run(
                    ['netstat', '-ano'],
                    capture_output=True,
                    text=True,
                    shell=True
                )

                for line in result.stdout.split('\n'):
                    if ':8001' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            logger.info(f"🔍 Found process {pid} using port 8001, attempting to terminate...")
                            subprocess.run(['taskkill', '/F', '/PID', pid], shell=True)
                            time.sleep(2)  # Wait for process to terminate

                            # Check if port is now available
                            if check_port_available(8001):
                                target_port = 8001
                                logger.info("✅ Successfully freed port 8001")
                            break
            except Exception as e:
                logger.error(f"❌ Failed to free port 8001: {e}")

    try:
        logger.info(f"🚀 Starting Dedicated Chatbot Service on port {target_port}...")
        uvicorn.run(
            "chatbot_api:app",
            host="127.0.0.1",
            port=target_port,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        logger.info("💡 Possible solutions:")
        logger.info("   1. Run as administrator")
        logger.info("   2. Check Windows Firewall settings")
        logger.info("   3. Ensure no other services are using the port")
        logger.info("   4. Try restarting your computer")
        raise
