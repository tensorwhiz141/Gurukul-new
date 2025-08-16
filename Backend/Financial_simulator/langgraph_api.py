"""
FastAPI application for the Financial Crew simulation using LangGraph.
This replaces the CrewAI implementation in api_app.py.
Integrated with MongoDB Atlas for persistent storage and learning from past simulations.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
import json
import uvicorn
import os
import uuid
import shutil
from pathlib import Path
from dotenv import load_dotenv
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import agentops
import logging
import numpy as np
from datetime import datetime
from langgraph_implementation import simulate_timeline_langgraph
from teacher_agent import run_teacher_agent, handle_pdf_upload, handle_pdf_removal

# Import MongoDB client
from database.mongodb_client import save_chat_message

# Import domain-specific forecasting
try:
    from domain_specific_forecasting import (
        domain_forecasting,
        forecast_edumentor,
        forecast_wellness,
        forecast_combined,
        ForecastRequest
    )
    DOMAIN_FORECASTING_AVAILABLE = True
    print("✅ Domain-specific forecasting models loaded successfully")
except ImportError as e:
    DOMAIN_FORECASTING_AVAILABLE = False
    print(f"⚠️ Domain-specific forecasting not available: {e}")

# Import Forecast Engine v2
from prediction_agent import PredictionAgent, create_prediction_agent, run_forecast_simulation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ************************************************FastAPI configuration************************************************************
# Temporarily disable AgentOps to focus on core functionality
try:
    agentops.init(
         api_key='4be58a32-e415-4142-82b7-834ae6b95422',
         default_tags=['langgraph']
    )
    print("✅ AgentOps initialized successfully")
except Exception as e:
    print(f"⚠️ AgentOps initialization failed: {e}")
    print("⚠️ Continuing without AgentOps...")
app = FastAPI()

# Add CORS middleware to handle OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a thread pool executor for running blocking operations
executor = ThreadPoolExecutor(max_workers=5)

# Store for simulation tasks and their status
simulation_tasks = {}

# Store for teacher agent tasks and their status
teacher_tasks = {}

# Initialize Forecast Engine v2
try:
    prediction_agent = create_prediction_agent(use_prophet=True)
    logger.info("✅ Forecast Engine v2 initialized with Prophet")
except Exception as e:
    logger.warning(f"⚠️ Prophet initialization failed, falling back to ARIMA: {e}")
    prediction_agent = create_prediction_agent(use_prophet=False)

# Define the expected schema for the incoming JSON
class ExpenseItem(BaseModel):
    name: str
    amount: float

class SimulationInput(BaseModel):
    user_id: str
    user_name: str
    income: float
    expenses: List[ExpenseItem]
    total_expenses: float
    goal: str
    financial_type: str
    risk_level: str

class SimulateRequest(BaseModel):
    n_months: int = 6  # Default to 6 months
    simulation_unit: str = "Months"
    user_inputs: dict
    simulation_id: Optional[str] = None  # Optional simulation ID

# Teacher agent models
class TeacherQuery(BaseModel):
    user_id: str
    query: str
    pdf_id: Optional[Union[str, List[str]]] = None  # Optional PDF ID(s) to search in specific PDF(s)
    wait: Optional[bool] = False  # Whether to wait for the response or return immediately

class TeacherResponse(BaseModel):
    """Response model for the teacher agent endpoint"""
    response: str
    chat_history: Optional[List[Dict[str, str]]] = None
    learning_task_id: Optional[str] = None  # Only use learning_task_id for clarity
    status: Optional[str] = None

def run_teacher_agent_background(task_id: str, user_id: str, query: str, chat_history: List[Dict[str, str]], pdf_id: Union[str, List[str], None] = None):
    """Background task to run the teacher agent with MongoDB Atlas Vector Search"""
    try:
        # Update task status to running
        teacher_tasks[task_id]["status"] = "running"

        print(f"🚀 Running teacher agent in background - task_id: {task_id}, user_id: {user_id}")

        # Check if MongoDB is available
        from database.mongodb_client import USE_MOCK_DB
        if USE_MOCK_DB:
            error_msg = "MongoDB Atlas connection is required for vector search functionality."
            error_message = f"I'm sorry, but MongoDB Atlas is required to process your question. Please contact the administrator."

            teacher_tasks[task_id]["status"] = "failed"
            teacher_tasks[task_id]["error"] = error_msg
            teacher_tasks[task_id]["response"] = error_message
            teacher_tasks[task_id]["chat_history"] = chat_history + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": error_message}
            ]

            # Save the error message to the database
            try:
                save_chat_message(user_id, "user", query)
                save_chat_message(user_id, "assistant", error_message)
            except Exception:
                pass  # Ignore database errors at this point

            print(f"❌ {error_msg}")
            return

        try:
            # Run the teacher agent
            result = run_teacher_agent(
                user_query=query,
                user_id=user_id,
                chat_history=chat_history,
                pdf_id=pdf_id
            )

            # Ensure we have a valid response
            response = result.get("response")
            if not response:
                response = "I'm sorry, I couldn't generate a response to your question."
                result["response"] = response

            # Save the response and chat history
            teacher_tasks[task_id]["response"] = response
            teacher_tasks[task_id]["chat_history"] = result.get("chat_history", [])

            # Save the new messages to the database
            save_chat_message(user_id, "user", query)
            save_chat_message(user_id, "assistant", response)

            # Update task status
            teacher_tasks[task_id]["status"] = "completed"
            print(f"✅ Teacher agent task completed - task_id: {task_id}")

        except ValueError as ve:
            if "MongoDB" in str(ve):
                error_msg = str(ve)
                error_message = f"I'm sorry, but MongoDB Atlas is required to process your question. Please contact the administrator."

                teacher_tasks[task_id]["status"] = "failed"
                teacher_tasks[task_id]["error"] = error_msg
                teacher_tasks[task_id]["response"] = error_message
                teacher_tasks[task_id]["chat_history"] = chat_history + [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": error_message}
                ]

                # Save the error message to the database
                try:
                    save_chat_message(user_id, "user", query)
                    save_chat_message(user_id, "assistant", error_message)
                except Exception:
                    pass  # Ignore database errors at this point

                print(f"❌ {error_msg}")
            else:
                raise

    except Exception as e:
        error_message = f"I'm sorry, but there was an error processing your question: {str(e)}"
        teacher_tasks[task_id]["status"] = "failed"
        teacher_tasks[task_id]["error"] = str(e)
        teacher_tasks[task_id]["response"] = error_message
        teacher_tasks[task_id]["chat_history"] = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_message}
        ]

        # Save the error message to the database
        try:
            save_chat_message(user_id, "user", query)
            save_chat_message(user_id, "assistant", error_message)
        except Exception:
            pass  # Ignore database errors at this point

        print(f"❌ Error in teacher agent task {task_id}: {e}")
        import traceback
        traceback.print_exc()

def run_simulation_background(task_id: str, user_inputs: dict, simulation_steps: int, simulation_unit: str):
    """Background task to run the simulation"""
    try:
        # Update task status to running
        simulation_tasks[task_id]["status"] = "running"

        # Generate a simulation ID
        from database.mongodb_client import (
            save_user_input,
            save_agent_output,
            get_previous_month_outputs,
            get_agent_outputs_for_month,
            generate_simulation_id,
            get_database
        )

        # Initialize MongoDB collections
        db = get_database()
        pdf_collection = db["pdf_metadata"]
        chunks_collection = db["pdf_chunks"]

        # Store simulation_id in task details
        simulation_id = generate_simulation_id()
        simulation_tasks[task_id]["simulation_id"] = simulation_id
        print(f"📝 Simulation ID: {simulation_id} for task {task_id}")

        # Run the simulation with task_id and simulation_id for status updates
        result = simulate_timeline_langgraph(
            simulation_steps,
            simulation_unit,
            user_inputs,
            task_id,
            simulation_id
        )

        # Update task status based on result
        if result:
            simulation_tasks[task_id]["status"] = "completed"
        else:
            simulation_tasks[task_id]["status"] = "failed"

    except Exception as e:
        simulation_tasks[task_id]["status"] = "failed"
        simulation_tasks[task_id]["error"] = str(e)
        print(f"Error in simulation task {task_id}: {e}")

@app.post("/start-simulation")
async def start_simulation(payload: SimulationInput, background_tasks: BackgroundTasks):
    """Start a simulation in the background and return a task ID"""
    try:
        # Convert pydantic model to dict
        user_inputs = payload.model_dump()

        # Get user_id
        user_id = user_inputs["user_id"]

        # Clear previous simulation data for this user (optional)
        try:
            # Get the database
            from database.mongodb_client import get_database
            db = get_database()

            if db:
                # Delete previous agent outputs for this user
                collection = db["agent_outputs"]
                delete_result = collection.delete_many({"user_id": user_id})
                print(f"🧹 Deleted {delete_result.deleted_count} previous simulation records for user {user_id}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clear previous simulation data: {e}")

        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Initialize task status
        simulation_tasks[task_id] = {
            "status": "queued",
            "user_id": user_inputs["user_id"],
            "user_name": user_inputs["user_name"],
            "created_at": asyncio.get_event_loop().time()
        }

        # Run simulation in background using the executor
        loop = asyncio.get_event_loop()
        background_tasks.add_task(
            loop.run_in_executor,
            executor,
            run_simulation_background,
            task_id,
            user_inputs,
            6,  # simulation_steps (6 months)
            "Months"  # simulation_unit
        )

        return {
            "status": "success",
            "message": "Simulation started",
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation-status/{task_id}")
async def get_simulation_status(task_id: str):
    """Get the status of a simulation task"""
    if task_id not in simulation_tasks:
        # Try to look up the task in MongoDB by simulation_id
        from database.mongodb_client import get_database
        db = get_database()
        collection = db["agent_outputs"]
        # Try to find a document with this task_id as simulation_id
        mongo_result = collection.find_one({"simulation_id": task_id})
        if mongo_result:
            print(f"[RECOVERY] Task {task_id} not found in memory but found in MongoDB.")
            return {
                "status": "success",
                "task_status": "completed",
                "task_details": {"simulation_id": task_id, "status": "completed"}
            }
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "status": "success",
        "task_status": simulation_tasks[task_id]["status"],
        "task_details": simulation_tasks[task_id]
    }

@app.get("/simulation-results/{task_id}")
async def get_simulation_results(task_id: str):
    """Get the latest results for a simulation task in progress"""
    if task_id not in simulation_tasks:
        # Try to look up the task in MongoDB by simulation_id
        from database.mongodb_client import get_database
        db = get_database()
        collection = db["agent_outputs"]
        mongo_results = list(collection.find({"simulation_id": task_id}))
        if mongo_results:
            print(f"[RECOVERY] Task {task_id} not found in memory but found in MongoDB.")
            # Try to get user_id from the first result
            user_id = mongo_results[0].get("user_id", "unknown")
            # Prepare default structure
            results = {
                "simulated_cashflow": [],
                "discipline_report": [],
                "goal_status": [],
                "behavior_tracker": [],
                "karmic_tracker": [],
                "financial_strategy": [],
                "person_history": [],
                "monthly_reflections": []
            }
            for item in mongo_results:
                agent_name = item.get("agent_name", "")
                month = item.get("month", 0)
                data = item.get("data", {})
                if data:
                    if "month" not in data:
                        data["month"] = month
                    if agent_name == "cashflow" or agent_name == "cashflow_simulator":
                        results["simulated_cashflow"].append(data)
                    elif agent_name == "discipline_tracker":
                        results["discipline_report"].append(data)
                    elif agent_name == "goal_tracker":
                        results["goal_status"].append(data)
                    elif agent_name == "behavior_tracker":
                        results["behavior_tracker"].append(data)
                    elif agent_name == "karma_tracker":
                        results["karmic_tracker"].append(data)
                    elif agent_name == "financial_strategy":
                        results["financial_strategy"].append(data)
            # (Optional) Add person_history and monthly_reflections if needed
            return {
                "status": "success",
                "ready": True,
                "message": "Simulation results available (recovered from DB).",
                "task_id": task_id,
                "task_status": "completed",
                "user_id": user_id,
                "data": results,
                "source": "mongodb"
            }
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "ready": False,
                "message": "Task not found",
                "data": {
                    "simulated_cashflow": [],
                    "discipline_report": [],
                    "goal_status": [],
                    "behavior_tracker": [],
                    "karmic_tracker": [],
                    "financial_strategy": [],
                    "person_history": [],
                    "monthly_reflections": []
                }
            }
        )
    # Get user_id from the task
    user_id = simulation_tasks[task_id].get("user_id")
    if not user_id:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "ready": False,
                "message": "User ID not found in task",
                "data": {
                    "simulated_cashflow": [],
                    "discipline_report": [],
                    "goal_status": [],
                    "behavior_tracker": [],
                    "karmic_tracker": [],
                    "financial_strategy": [],
                    "person_history": [],
                    "monthly_reflections": []
                }
            }
        )

    # Get simulation_id if available
    simulation_id = simulation_tasks[task_id].get("simulation_id")

    # Get the latest results from MongoDB
    try:
        # If simulation_id is available, filter by it for more precise results
        if simulation_id:
            from database.mongodb_client import get_database
            db = get_database()
            query = {
                "user_id": user_id,
                "simulation_id": simulation_id
            }
            collection = db["agent_outputs"]
            mongo_results = list(collection.find(query))
            for result in mongo_results:
                if "_id" in result:
                    result["_id"] = str(result["_id"])
        else:
            from database.mongodb_client import get_all_agent_outputs_for_user
            mongo_results = get_all_agent_outputs_for_user(user_id)

        # Prepare default structure
        results = {
            "simulated_cashflow": [],
            "discipline_report": [],
            "goal_status": [],
            "behavior_tracker": [],
            "karmic_tracker": [],
            "financial_strategy": [],
            "person_history": [],
            "monthly_reflections": []
        }

        if mongo_results:
            # Group by agent name
            for item in mongo_results:
                agent_name = item.get("agent_name", "")
                month = item.get("month", 0)
                data = item.get("data", {})
                if data:
                    if "month" not in data:
                        data["month"] = month
                    if agent_name == "cashflow" or agent_name == "cashflow_simulator":
                        results["simulated_cashflow"].append(data)
                    elif agent_name == "discipline_tracker":
                        results["discipline_report"].append(data)
                    elif agent_name == "goal_tracker":
                        results["goal_status"].append(data)
                    elif agent_name == "behavior_tracker":
                        results["behavior_tracker"].append(data)
                    elif agent_name == "karma_tracker":
                        results["karmic_tracker"].append(data)
                    elif agent_name == "financial_strategy":
                        results["financial_strategy"].append(data)

            # Add person_history from data folder with user_id prefix
            data_dir = "data"
            person_history_path = f"{data_dir}/{user_id}_person_history.json"
            alt_person_history_path = f"{data_dir}/string_persona_history.json"
            fallback_path = f"{data_dir}/person_history.json"
            person_history_data = []
            if os.path.exists(person_history_path):
                try:
                    with open(person_history_path, "r") as f:
                        person_history_data = json.load(f)
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {user_id}_person_history.json: {e}")
            elif os.path.exists(alt_person_history_path):
                try:
                    with open(alt_person_history_path, "r") as f:
                        person_history_data = json.load(f)
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {alt_person_history_path}: {e}")
            elif os.path.exists(fallback_path):
                try:
                    with open(fallback_path, "r") as f:
                        person_history_data = json.load(f)
                except Exception as e:
                    print(f"⚠️ Warning: Could not load fallback person_history.json: {e}")
            results["person_history"] = person_history_data

            # Add monthly reflections from monthly_output folder
            monthly_output_dir = "monthly_output"
            monthly_reflections = []
            os.makedirs(monthly_output_dir, exist_ok=True)
            months = set()
            for category in ["simulated_cashflow", "discipline_report", "goal_status", "behavior_tracker", "karmic_tracker", "financial_strategy"]:
                for item in results.get(category, []):
                    if "month" in item:
                        months.add(item["month"])
            for month in months:
                reflection_path = f"{monthly_output_dir}/{user_id}_reflection_month_{month}.json"
                fallback_reflection_path = f"{monthly_output_dir}/reflection_month_{month}.json"
                if os.path.exists(reflection_path):
                    try:
                        with open(reflection_path, "r") as f:
                            reflection_data = json.load(f)
                            if isinstance(reflection_data, dict):
                                reflection_data["month"] = month
                            monthly_reflections.append(reflection_data)
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load {user_id}_reflection_month_{month}.json: {e}")
                        if os.path.exists(fallback_reflection_path):
                            try:
                                with open(fallback_reflection_path, "r") as f:
                                    reflection_data = json.load(f)
                                    if isinstance(reflection_data, dict):
                                        reflection_data["month"] = month
                                    monthly_reflections.append(reflection_data)
                            except Exception as e:
                                print(f"⚠️ Warning: Could not load fallback reflection_month_{month}.json: {e}")
                elif os.path.exists(fallback_reflection_path):
                    try:
                        with open(fallback_reflection_path, "r") as f:
                            reflection_data = json.load(f)
                            if isinstance(reflection_data, dict):
                                reflection_data["month"] = month
                            monthly_reflections.append(reflection_data)
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load fallback reflection_month_{month}.json: {e}")
            results["monthly_reflections"] = monthly_reflections

            return {
                "status": "success",
                "ready": True,
                "message": "Simulation results available.",
                "task_id": task_id,
                "task_status": simulation_tasks[task_id]["status"],
                "user_id": user_id,
                "data": results,
                "source": "mongodb"
            }
        else:
            return {
                "status": "success",
                "ready": False,
                "message": "No simulation results available yet.",
                "task_id": task_id,
                "task_status": simulation_tasks[task_id]["status"],
                "user_id": user_id,
                "data": results,
                "source": "mongodb"
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "ready": False,
                "message": f"Error retrieving simulation results: {str(e)}",
                "data": {
                    "simulated_cashflow": [],
                    "discipline_report": [],
                    "goal_status": [],
                    "behavior_tracker": [],
                    "karmic_tracker": [],
                    "financial_strategy": [],
                    "person_history": [],
                    "monthly_reflections": []
                }
            }
        )

# /simulate endpoint removed as it's not being used

# /get-simulation-result/{user_id} endpoint removed as it's not being used

# Teacher agent endpoints
@app.post("/user/learning", response_model=TeacherResponse)
async def learning_endpoint(
    query: TeacherQuery,
    background_tasks: BackgroundTasks
):
    """Process a learning query from the user and return a response from the teacher agent using MongoDB Atlas Vector Search"""
    try:
        actual_user_id = query.user_id
        actual_query = query.query
        actual_pdf_id = query.pdf_id
        actual_wait = query.wait

        print(f"📥 Received learning request - user_id: {actual_user_id}, query: '{actual_query}'")

        # Check if MongoDB is available
        from database.mongodb_client import USE_MOCK_DB
        if USE_MOCK_DB:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "MongoDB Atlas connection is required for vector search functionality.",
                    "requires_mongodb_atlas": True
                }
            )

        # Get chat history from database but limit to last 10 messages to avoid overwhelming context
        from database.mongodb_client import get_chat_history_for_user
        chat_history = get_chat_history_for_user(actual_user_id, limit=10)

        # Convert to the format expected by the teacher agent
        formatted_chat_history = []
        for msg in chat_history:
            formatted_chat_history.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        print(f"📜 Retrieved {len(formatted_chat_history)} chat history messages")
        if formatted_chat_history:
            print(f"📜 Most recent message - Role: {formatted_chat_history[-1].get('role')}, Content: {formatted_chat_history[-1].get('content', '')[:50]}...")

        # Handle PDF IDs
        pdf_id_param = None
        if actual_pdf_id:
            if isinstance(actual_pdf_id, list):
                # If multiple PDF IDs are provided
                print(f"📚 Using multiple PDFs: {', '.join(actual_pdf_id)}")
                if len(actual_pdf_id) > 0:
                    pdf_id_param = actual_pdf_id
            else:
                # Single PDF ID
                print(f"📚 Using specific PDF: {actual_pdf_id}")
                pdf_id_param = actual_pdf_id

        # Ensure query is properly formatted
        current_query = actual_query.strip()
        print(f"🔍 Processing query: '{current_query}'")

        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Check if we should wait for the response or run in the background
        if actual_wait:
            # Run the teacher agent synchronously
            print(f"⏱️ Running teacher agent synchronously - task_id: {task_id}")
            try:
                result = run_teacher_agent(
                    user_query=current_query,
                    user_id=actual_user_id,
                    chat_history=formatted_chat_history,
                    pdf_id=pdf_id_param
                )

                # Log the response
                print(f"✅ Teacher agent response: '{result['response'][:50]}...'")

                # Save the new messages to the database
                save_chat_message(actual_user_id, "user", current_query)
                save_chat_message(actual_user_id, "assistant", result["response"])

                return TeacherResponse(
                    response=result["response"],
                    chat_history=result["chat_history"],
                    learning_task_id=task_id,  # Only use learning_task_id for clarity
                    status="completed"
                )
            except ValueError as ve:
                if "MongoDB" in str(ve):
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": str(ve),
                            "requires_mongodb_atlas": True
                        }
                    )
                else:
                    raise
        else:
            # Run the teacher agent in the background
            print(f"🔄 Running teacher agent in background - task_id: {task_id}")

            # Initialize task status
            teacher_tasks[task_id] = {
                "status": "queued",
                "user_id": actual_user_id,
                "query": current_query,
                "created_at": asyncio.get_event_loop().time(),
                "response": None,
                "chat_history": None
            }

            # Run teacher agent in background using the executor
            loop = asyncio.get_event_loop()
            background_tasks.add_task(
                loop.run_in_executor,
                executor,
                run_teacher_agent_background,
                task_id,
                actual_user_id,
                current_query,
                formatted_chat_history,
                pdf_id_param
            )

            # Save the user message to the database immediately
            save_chat_message(actual_user_id, "user", current_query)

            return TeacherResponse(
                response="Your question is being processed. Please check back in a moment for the response.",
                learning_task_id=task_id,  # Only use learning_task_id for clarity
                status="queued"
            )
    except Exception as e:
        print(f"❌ Error in learning endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/learning/{learning_task_id}", response_model=TeacherResponse)
async def get_learning_status(learning_task_id: str):
    """Get the status and response of a teacher agent task"""
    if learning_task_id not in teacher_tasks:
        raise HTTPException(status_code=404, detail="Learning task not found")

    task = teacher_tasks[learning_task_id]
    status = task.get("status", "queued")

    # Ensure we have a valid response string
    response = task.get("response")
    if response is None:
        if status == "failed":
            response = "I'm sorry, but there was an error processing your question. Please try again."
        else:
            response = "Your question is still being processed. Please check back in a moment."

    # Return the response with all required fields
    return TeacherResponse(
        response=response,
        chat_history=task.get("chat_history", []),
        learning_task_id=learning_task_id,  # Only use learning_task_id for clarity
        status=status
    )

@app.post("/pdf/chat")
async def pdf_upload_endpoint(
    user_id: str = Form(...),
    pdf_file: UploadFile = File(...)
):
    """Upload a PDF file for the teacher agent to use in explanations with MongoDB Atlas Vector Search"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp_pdfs")
        temp_dir.mkdir(exist_ok=True)

        # Save the uploaded file
        file_path = temp_dir / f"{user_id}_{pdf_file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)

        # Process the PDF
        result = handle_pdf_upload(str(file_path), user_id)

        if result["success"]:
            return {
                "status": "success",
                "message": result["message"],
                "user_id": user_id,
                "pdf_id": result["pdf_id"],
                "chunk_count": result.get("chunk_count", 0)
            }
        else:
            # Return a more specific error message for MongoDB Atlas requirement
            if "MongoDB Atlas" in result["message"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": result["message"],
                        "requires_mongodb_atlas": True
                    }
                )
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": result["message"]
                    }
                )
    except Exception as e:
        print(f"❌ Error in PDF upload: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Define a model for PDF removal request
class PDFRemovalRequest(BaseModel):
    user_id: str
    pdf_id: Optional[Union[str, List[str]]] = None

# Forecast Engine v2 models
class ForecastRequest(BaseModel):
    days: Optional[int] = 30
    format: Optional[str] = "json"

class AgentScoreRequest(BaseModel):
    agent_id: str
    current_load: Optional[int] = None

@app.post("/pdf/removed")
async def pdf_removal_endpoint(request: PDFRemovalRequest):
    """
    Remove PDF data for a user or a specific PDF

    Request body:
        user_id: User identifier
        pdf_id: Optional PDF ID or list of PDF IDs to remove specific PDFs
    """
    try:
        print(f"🗑️ PDF removal request - user_id: {request.user_id}, pdf_id: {request.pdf_id}")

        # Remove PDF data
        result = handle_pdf_removal(request.user_id, request.pdf_id)

        if result["success"]:
            return {
                "status": "success",
                "message": result["message"],
                "user_id": request.user_id,
                "pdf_id": result.get("pdf_id")
            }
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": result["message"]
                }
            )
    except Exception as e:
        print(f"❌ Error in PDF removal: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdf/list")
async def pdf_list_endpoint(user_id: str):
    try:
        # Get MongoDB database
        from database.mongodb_client import get_database
        db = get_database()

        # Get all PDFs for this user
        pdf_docs = list(db["pdf_metadata"].find({"user_id": user_id}))

        # Convert ObjectId to string
        for doc in pdf_docs:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        return {
            "status": "success",
            "pdfs": pdf_docs
        }
    except Exception as e:
        print(f"❌ Error listing PDFs: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ************************************************Forecast Engine v2 Endpoints************************************************************

@app.get("/forecast")
async def get_forecast(days: int = 30, format: str = "json"):
    """
    Get comprehensive forecast for all metrics with multilingual support.

    Args:
        days: Number of days to forecast (default: 30)
        format: Output format - json or csv (default: json)

    Returns:
        Multilingual JSON with forecast data
    """
    try:
        logger.info(f"📊 Generating forecast for {days} days in {format} format")

        # Generate forecasts
        forecasts = prediction_agent.generate_forecasts(forecast_days=days)
        daily_summary = prediction_agent.get_daily_forecast_summary()

        # Create multilingual response structure
        response = {
            "report_type": "forecast",
            "language": "en",  # Default to English, can be extended for multilingual support
            "sentiment": "neutral",
            "content": {
                "risk": daily_summary.get('overall_status', 'stable'),
                "recommendation": _get_primary_recommendation(daily_summary),
                "forecast_data": forecasts,
                "summary": daily_summary,
                "generated_at": datetime.now().isoformat(),
                "forecast_days": days
            },
            "metadata": {
                "model_type": "prophet" if prediction_agent.use_prophet else "arima",
                "confidence_level": "high",
                "data_quality": "good"
            }
        }

        logger.info("✅ Forecast generated successfully")
        return response

    except Exception as e:
        logger.error(f"❌ Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/forecast-json")
async def get_forecast_json(days: int = 7):
    """
    Get daily load and escalation risk in chart-ready format for dashboard consumption.

    Args:
        days: Number of days for chart data (default: 7)

    Returns:
        Chart-ready JSON format
    """
    try:
        logger.info(f"📈 Generating chart-ready forecast for {days} days")

        # Generate forecasts
        forecasts = prediction_agent.generate_forecasts(forecast_days=days)

        # Extract chart data
        chart_data = {
            "labels": [],
            "datasets": [
                {
                    "label": "Daily Agent Load",
                    "data": [],
                    "borderColor": "rgb(75, 192, 192)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "tension": 0.1
                },
                {
                    "label": "Escalation Risk",
                    "data": [],
                    "borderColor": "rgb(255, 99, 132)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "tension": 0.1,
                    "yAxisID": "y1"
                }
            ]
        }

        # Populate chart data
        if 'daily_agent_load' in forecasts and 'escalation_likelihood' in forecasts:
            load_predictions = forecasts['daily_agent_load']['predictions'][:days]
            escalation_predictions = forecasts['escalation_likelihood']['predictions'][:days]

            for i in range(min(len(load_predictions), len(escalation_predictions))):
                load_pred = load_predictions[i]
                escalation_pred = escalation_predictions[i]

                # Format date for chart labels
                date_obj = pd.to_datetime(load_pred['ds'])
                chart_data["labels"].append(date_obj.strftime("%Y-%m-%d"))

                # Add data points
                chart_data["datasets"][0]["data"].append(round(load_pred['yhat'], 2))
                chart_data["datasets"][1]["data"].append(round(escalation_pred['yhat'] * 100, 2))  # Convert to percentage

        response = {
            "chart_data": chart_data,
            "summary": {
                "total_days": days,
                "avg_daily_load": round(np.mean(chart_data["datasets"][0]["data"]), 2) if chart_data["datasets"][0]["data"] else 0,
                "avg_escalation_risk": round(np.mean(chart_data["datasets"][1]["data"]), 2) if chart_data["datasets"][1]["data"] else 0,
                "peak_load_day": chart_data["labels"][np.argmax(chart_data["datasets"][0]["data"])] if chart_data["datasets"][0]["data"] else None
            },
            "generated_at": datetime.now().isoformat()
        }

        logger.info("✅ Chart-ready forecast generated successfully")
        return response

    except Exception as e:
        logger.error(f"❌ Error generating chart forecast: {e}")
        raise HTTPException(status_code=500, detail=f"Chart forecast generation failed: {str(e)}")

@app.post("/score-agent")
async def score_agent(request: AgentScoreRequest):
    """
    Score an agent based on current load and predicted capacity.

    Args:
        request: Agent scoring request with agent_id and optional current_load

    Returns:
        Agent score and capacity assessment
    """
    try:
        logger.info(f"🎯 Scoring agent {request.agent_id} with load {request.current_load}")

        # Get agent score
        score_result = prediction_agent.get_agent_score(
            agent_id=request.agent_id,
            current_load=request.current_load
        )

        # Add additional context
        response = {
            "status": "success",
            "agent_score": score_result,
            "timestamp": datetime.now().isoformat(),
            "recommendations": _get_agent_recommendations(score_result)
        }

        logger.info(f"✅ Agent {request.agent_id} scored: {score_result['score']:.2f}")
        return response

    except Exception as e:
        logger.error(f"❌ Error scoring agent {request.agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent scoring failed: {str(e)}")

def _get_primary_recommendation(daily_summary: Dict[str, Any]) -> str:
    """Extract primary recommendation from daily summary."""
    recommendations = daily_summary.get('recommendations', [])
    if not recommendations:
        return "continue_monitoring"

    # Return highest priority recommendation
    high_priority = [r for r in recommendations if r.get('priority') == 'high']
    if high_priority:
        return high_priority[0].get('action', 'monitor_closely')

    return recommendations[0].get('action', 'monitor_closely')

def _get_agent_recommendations(score_result: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on agent score."""
    recommendations = []

    score = score_result.get('score', 0.5)
    status = score_result.get('capacity_status', 'unknown')

    if status == 'overloaded':
        recommendations.extend([
            "Reassign some tasks to other agents",
            "Consider increasing agent capacity",
            "Monitor for burnout indicators"
        ])
    elif status == 'available':
        recommendations.extend([
            "Agent can accept additional tasks",
            "Consider for high-priority assignments",
            "Maintain current workload balance"
        ])
    else:  # moderate
        recommendations.extend([
            "Monitor load carefully",
            "Avoid additional complex tasks",
            "Prepare backup resources"
        ])

    return recommendations

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Test prediction agent
        test_forecasts = prediction_agent.get_daily_forecast_summary()

        return {
            "status": "healthy",
            "service": "Forecast Engine v2",
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "prediction_agent": "operational",
                "prophet_model": "available" if prediction_agent.use_prophet else "unavailable",
                "arima_fallback": "available",
                "api_endpoints": "operational"
            },
            "last_forecast": test_forecasts.get('forecast_date', 'unknown')
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Forecast Engine v2",
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring."""
    try:
        import psutil

        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "forecast_engine": {
                "model_type": "prophet" if prediction_agent.use_prophet else "arima",
                "predictions_cached": len(prediction_agent.predictions),
                "historical_data_loaded": len(prediction_agent.historical_data)
            }
        }
    except ImportError:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": "psutil not available for system metrics",
            "forecast_engine": {
                "model_type": "prophet" if prediction_agent.use_prophet else "arima",
                "predictions_cached": len(prediction_agent.predictions),
                "historical_data_loaded": len(prediction_agent.historical_data)
            }
        }

# Domain-Specific Forecasting Endpoints
@app.post("/forecast/edumentor")
async def forecast_edumentor_endpoint(request: ForecastRequest):
    """Generate educational metrics forecasting for Edumentor"""
    if not DOMAIN_FORECASTING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Domain-specific forecasting not available - missing dependencies"
        )

    try:
        result = await forecast_edumentor(request)
        return result
    except Exception as e:
        logger.error(f"❌ Error in Edumentor forecasting: {e}")
        raise HTTPException(status_code=500, detail=f"Edumentor forecasting failed: {str(e)}")

@app.post("/forecast/wellness")
async def forecast_wellness_endpoint(request: ForecastRequest):
    """Generate wellness metrics forecasting for Wellness Bot"""
    if not DOMAIN_FORECASTING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Domain-specific forecasting not available - missing dependencies"
        )

    try:
        result = await forecast_wellness(request)
        return result
    except Exception as e:
        logger.error(f"❌ Error in Wellness forecasting: {e}")
        raise HTTPException(status_code=500, detail=f"Wellness forecasting failed: {str(e)}")

@app.post("/forecast/combined")
async def forecast_combined_endpoint(request: ForecastRequest):
    """Generate combined forecasting across Edumentor and Wellness domains"""
    if not DOMAIN_FORECASTING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Domain-specific forecasting not available - missing dependencies"
        )

    try:
        result = await forecast_combined(request)
        return result
    except Exception as e:
        logger.error(f"❌ Error in combined forecasting: {e}")
        raise HTTPException(status_code=500, detail=f"Combined forecasting failed: {str(e)}")

@app.get("/forecast/domains")
async def get_available_domains():
    """Get information about available forecasting domains"""
    domains = {
        "available": DOMAIN_FORECASTING_AVAILABLE,
        "domains": []
    }

    if DOMAIN_FORECASTING_AVAILABLE:
        domains["domains"] = [
            {
                "name": "edumentor",
                "description": "Educational metrics forecasting",
                "metrics": [
                    "quiz_completion_rate",
                    "learning_progress_risk",
                    "concept_mastery_decline",
                    "engagement_drop_risk",
                    "study_session_frequency",
                    "knowledge_retention_score",
                    "learning_velocity"
                ]
            },
            {
                "name": "wellness",
                "description": "Wellness metrics forecasting",
                "metrics": [
                    "financial_health_score",
                    "burnout_risk",
                    "spending_volatility",
                    "stress_level_prediction",
                    "work_life_balance_score",
                    "emotional_wellness_index",
                    "financial_anxiety_level",
                    "sleep_quality_impact"
                ]
            },
            {
                "name": "combined",
                "description": "Cross-domain forecasting with holistic insights",
                "features": [
                    "Cross-domain correlation analysis",
                    "Holistic recommendations",
                    "Integrated risk assessment",
                    "Intervention priority scoring"
                ]
            }
        ]

    return domains

@app.post("/simulate-workflow")
async def simulate_workflow():
    """Run end-to-end agent simulation workflow."""
    try:
        from agent_integration import run_end_to_end_simulation

        logger.info("🚀 Starting end-to-end workflow simulation")
        results = run_end_to_end_simulation()

        return {
            "status": "success",
            "message": "Workflow simulation completed",
            "results": results
        }
    except Exception as e:
        logger.error(f"❌ Error in workflow simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow simulation failed: {str(e)}")

def main():
    try:
        uvicorn.run("langgraph_api:app",host="0.0.0.0",port=8002,reload=False)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        raise

# If you want to run with `python langgraph_api.py`
if __name__ == "__main__":
    uvicorn.run("langgraph_api:app", host="0.0.0.0", port=8002, reload=False)
