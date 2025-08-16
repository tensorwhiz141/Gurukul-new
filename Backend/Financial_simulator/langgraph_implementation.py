"""
LangGraph implementation of the Financial Crew simulation system.
This replaces the CrewAI implementation in crew.py.
Integrated with MongoDB Atlas for persistent storage and learning from past simulations.
Enhanced with Redis for caching, session management, and rate limiting.
"""

from typing import Dict, List, Any, TypedDict, Annotated, Literal, Optional, Union
import json
import os
from datetime import datetime, timedelta
import time
import uuid
import redis
import yaml

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug

# Import custom JSON parsing utilities
from utils.json_fix import safe_parse_json, create_fallback_json

import langgraph.graph as lg
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
from functions.economic_context import EconomicEnvironment, simulate_monthly_market
from functions.monthly_simulation import deduplicate_and_save, assign_persona, generate_monthly_reflection_report
from functions.task_functions import (
    build_discipline_report_context,
    build_goal_status_context,
    build_behavior_tracker_context,
    build_karmic_tracker_context,
    build_financial_strategy_context
)
from functions.task_functions_fixed import build_simulated_cashflow_context

# Import MongoDB client
from database.mongodb_client import (
    save_user_input,
    save_agent_output,
    get_previous_month_outputs,
    get_agent_outputs_for_month,
    generate_simulation_id
)

# Load environment variables
load_dotenv()

# Initialize Redis client with error handling
redis_client = None
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", ""),
        decode_responses=True,  # Automatically decode responses to strings
        socket_connect_timeout=5,  # 5 second timeout
        socket_timeout=5
    )
    # Test the connection
    redis_client.ping()
    print("✅ Redis connection established successfully")
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    print("⚠️ Continuing without Redis caching - using in-memory fallback")
    redis_client = None

# In-memory cache fallback when Redis is not available
_memory_cache = {}

# Redis utility functions with fallback
def redis_cache_get(key, namespace="financial_crew"):
    """Get data from Redis cache or memory fallback."""
    if redis_client is not None:
        try:
            full_key = f"{namespace}:{key}"
            data = redis_client.get(full_key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Redis cache get error: {e}")
            # Fall back to memory cache
            pass

    # Use memory cache as fallback
    full_key = f"{namespace}:{key}"
    return _memory_cache.get(full_key)

def redis_cache_set(key, value, expiry=3600, namespace="financial_crew"):
    """Set data in Redis cache with expiry in seconds or memory fallback."""
    if redis_client is not None:
        try:
            full_key = f"{namespace}:{key}"
            redis_client.setex(full_key, expiry, json.dumps(value))
            return True
        except Exception as e:
            print(f"Redis cache set error: {e}")
            # Fall back to memory cache
            pass

    # Use memory cache as fallback
    full_key = f"{namespace}:{key}"
    _memory_cache[full_key] = value
    return True

def redis_cache_delete(key, namespace="financial_crew"):
    """Delete data from Redis cache or memory fallback."""
    if redis_client is not None:
        try:
            full_key = f"{namespace}:{key}"
            return bool(redis_client.delete(full_key))
        except Exception as e:
            print(f"Redis cache delete error: {e}")
            # Fall back to memory cache
            pass

    # Use memory cache as fallback
    full_key = f"{namespace}:{key}"
    if full_key in _memory_cache:
        del _memory_cache[full_key]
    return True

def redis_rate_limit(key, limit=100, window=60, namespace="rate_limits"):
    """
    Check if a key is rate limited with Redis or memory fallback.

    Args:
        key (str): Identifier for the client (e.g., IP address, user ID)
        limit (int): Maximum number of requests allowed in the time window
        window (int): Time window in seconds
        namespace (str): Redis namespace

    Returns:
        tuple: (is_limited, remaining, reset_time)
    """
    if redis_client is not None:
        try:
            # Get the current timestamp
            current_time = int(time.time())

            # Create a key with the current window
            window_start = current_time - (current_time % window)
            window_key = f"{namespace}:{key}:{window_start}"

            # Get the current count
            count = int(redis_client.get(window_key) or 0)

            # Check if rate limited
            is_limited = count >= limit

            # Increment the counter if not rate limited
            if not is_limited:
                # Use pipeline to ensure atomic operations
                pipe = redis_client.pipeline()
                pipe.incr(window_key)
                # Set expiry if it doesn't exist
                pipe.expire(window_key, window)
                pipe.execute()
                count += 1

            # Calculate remaining requests and reset time
            remaining = max(0, limit - count)
            reset_time = window_start + window - current_time

            return is_limited, remaining, reset_time
        except Exception as e:
            print(f"Redis rate limit error: {e}")
            # Fall back to allowing requests when Redis fails
            pass

    # Memory fallback - for simplicity, allow all requests when Redis is unavailable
    # In production, implement proper memory-based rate limiting
    return False, limit, window

# Initialize LLM
from litellm import Cache

def get_llm(model_name="groq/llama3-70b-8192"):
    """Get the LLM with optional caching enabled."""
    # Try to set up Redis caching if available, otherwise continue without caching
    try:
        if redis_client is not None:
            from langchain_community.cache import RedisCache
            from langchain.globals import set_llm_cache

            # Initialize Redis cache with client
            redis_cache = RedisCache(redis_=redis_client)

            # Set the cache globally
            set_llm_cache(redis_cache)
            print("✅ LLM caching enabled with Redis")
        else:
            print("⚠️ LLM caching disabled - Redis not available")
    except Exception as e:
        print(f"⚠️ Could not set up LLM caching: {e}")

    # Get API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if model_name.startswith("groq/"):
        if groq_api_key:
            return ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name.replace("groq/", ""),
                temperature=0.2,
                max_tokens=4000
            )
        else:
            print("⚠️ GROQ_API_KEY not found, falling back to default model")
            return ChatGroq(
                model_name="llama3-70b-8192",
                temperature=0.2,
                max_tokens=4000
            )
    elif model_name.startswith("openai/"):
        if openai_api_key:
            return ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name=model_name.replace("openai/", ""),
                temperature=0.2,
                max_tokens=4000
            )
        else:
            print("⚠️ OPENAI_API_KEY not found, falling back to Groq")
            if groq_api_key:
                return ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="llama3-70b-8192",
                    temperature=0.2,
                    max_tokens=4000
                )
    else:
        # Default to Groq
        if groq_api_key:
            return ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama3-70b-8192",
                temperature=0.2,
                max_tokens=4000
            )
        else:
            print("❌ No API keys available for LLM")
            raise ValueError("No valid API keys found for LLM initialization")

# Define state schema
class FinancialSimulationState(TypedDict):
    """State for the financial simulation workflow."""
    # Input data
    user_inputs: Dict[str, Any]
    month_number: int
    simulation_id: str

    # Simulation results
    cashflow_result: Optional[Dict[str, Any]]
    discipline_result: Optional[Dict[str, Any]]
    goal_tracking_result: Optional[Dict[str, Any]]
    behavior_result: Optional[Dict[str, Any]]
    karma_result: Optional[Dict[str, Any]]
    financial_strategy_result: Optional[Dict[str, Any]]

    # Context for agents
    cashflow_context: Optional[str]
    discipline_context: Optional[str]
    goal_tracking_context: Optional[str]
    behavior_context: Optional[str]
    karma_context: Optional[str]
    financial_strategy_context: Optional[str]

    # Economic data
    economic_context: Dict[str, float]
    market_context: str

    # Previous month data from MongoDB
    previous_month_data: Optional[Dict[str, List[Dict[str, Any]]]]

# Import enhanced functions
from functions.enhanced_task_functions import generate_enhanced_cashflow_simulation, generate_enhanced_financial_strategy, generate_enhanced_goal_tracking

# Define agent nodes
def simulate_cashflow_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Simulate cash flow for the current month using enhanced analysis."""
    print(f"🟢 Executing enhanced cashflow simulation for month {state['month_number']}")

    try:
        # Get previous month data for enhanced analysis
        user_id = state["user_inputs"].get("user_id", "default_user")
        month = state["month_number"]
        previous_data = None

        if month > 1 and state.get("previous_month_data"):
            previous_data = state["previous_month_data"].get("cashflow", [])
            print(f"📊 Using {len(previous_data) if previous_data else 0} previous months for enhanced analysis")

        # Use enhanced cashflow simulation
        result = generate_enhanced_cashflow_simulation(
            user_inputs=state["user_inputs"],
            month=month,
            economic_context=state["economic_context"],
            market_context=state["market_context"],
            previous_data=previous_data
        )

        print(f"✅ Enhanced cashflow simulation completed for month {month}")
        print(f"📈 Financial Health Score: {result.get('analysis', {}).get('financial_health_score', 'N/A')}")
        print(f"💰 Monthly Surplus: ₹{result.get('savings', {}).get('amount', 0):,.0f}")

        # Update state
        state["cashflow_result"] = result
        return state

    except Exception as e:
        print(f"❌ Error in enhanced cashflow simulation: {e}")

        # Fallback to basic simulation
        print("🔄 Falling back to basic cashflow simulation...")

        user_name = state["user_inputs"].get("user_name", "Default User")
        income = float(state["user_inputs"].get("income", 0))
        total_expenses = float(state["user_inputs"].get("total_expenses", 0))
        monthly_surplus = income - total_expenses

        fallback_result = {
            "user_name": user_name,
            "month": month,
            "income": {
                "salary": income,
                "investments": 0,
                "other": 0,
                "total": income
            },
            "expenses": {
                "housing": 0,
                "utilities": 0,
                "groceries": total_expenses,  # Put all expenses in groceries as fallback
                "transportation": 0,
                "healthcare": 0,
                "entertainment": 0,
                "dining_out": 0,
                "subscriptions": 0,
                "other": 0,
                "total": total_expenses
            },
            "savings": {
                "amount": monthly_surplus,
                "percentage_of_income": (monthly_surplus / income * 100) if income > 0 else 0,
                "target_met": monthly_surplus >= (income * 0.2)
            },
            "balance": {
                "starting": 0,
                "ending": monthly_surplus,
                "change": monthly_surplus
            },
            "notes": f"Basic simulation for month {month}. Enhanced analysis failed: {str(e)}",
            "error": str(e)
        }

        state["cashflow_result"] = fallback_result
        return state

        # Save result to file
        output_path = f"output/{user_id}_simulated_cashflow_simulation.json"

        # Get user inputs
        user_name = state["user_inputs"].get("user_name", "Default User")
        user_income = state["user_inputs"].get("income", 5000)
        user_expenses = state["user_inputs"].get("expenses", [])

        # Standardize the result format
        if isinstance(result, dict):
            # If it's a direct dictionary, ensure it has the required fields
            if "simulation_result" in result and "simulation_output" in result:
                # Create a new result structure that uses ONLY the user's actual input values
                new_result = {
                    "month": month,
                    "user_name": user_name,  # Use exact user_name from input
                    "income": {
                        "salary": user_income,  # Use exact income from input
                        "investments": 0,
                        "other": 0,
                        "total": user_income
                    },
                    "expenses": {
                        "housing": 0,
                        "utilities": 0,
                        "groceries": 0,
                        "transportation": 0,
                        "healthcare": 0,
                        "entertainment": 0,
                        "dining_out": 0,
                        "subscriptions": 0,
                        "other": 0,
                        "total": 0  # Will be calculated from actual expenses
                    },
                    "balance": {
                        "starting": 0,
                        "ending": 0,
                        "change": 0
                    },
                    "notes": "Based on your provided data."
                }

                # Always use the user's actual expenses
                # Map user expenses to categories
                total_expenses = 0
                for expense in user_expenses:
                    if isinstance(expense, dict) and "name" in expense and "amount" in expense:
                        category = expense["name"].lower()
                        amount = float(expense["amount"])
                        total_expenses += amount

                        # Map common expense categories
                        if "rent" in category or "mortgage" in category or "home" in category or "house" in category:
                            new_result["expenses"]["housing"] = amount
                        elif "util" in category or "electric" in category or "water" in category or "gas" in category:
                            new_result["expenses"]["utilities"] = amount
                        elif "groc" in category or "food" in category:
                            new_result["expenses"]["groceries"] = amount
                        elif "transport" in category or "car" in category or "fuel" in category or "gas" in category:
                            new_result["expenses"]["transportation"] = amount
                        elif "health" in category or "medical" in category or "insurance" in category:
                            new_result["expenses"]["healthcare"] = amount
                        elif "entertain" in category or "fun" in category or "recreation" in category:
                            new_result["expenses"]["entertainment"] = amount
                        elif "dining" in category or "restaurant" in category or "eat out" in category:
                            new_result["expenses"]["dining_out"] = amount
                        elif "subscript" in category or "streaming" in category or "netflix" in category:
                            new_result["expenses"]["subscriptions"] = amount
                        else:
                            # If no match, add to other
                            new_result["expenses"]["other"] += amount

                # Update total expenses
                new_result["expenses"]["total"] = total_expenses

                # Update balance
                savings = user_income - total_expenses
                new_result["balance"]["change"] = savings
                new_result["balance"]["ending"] = savings

                result = new_result

            # Ensure required fields exist
            result["month"] = month
            result["user_name"] = user_name

            # Ensure we have a complete structure with all fields for all months
            # Make sure we have income structure
            if "income" not in result:
                result["income"] = {
                    "salary": user_income,
                    "investments": 0,
                    "other": 0,
                    "total": user_income
                }

            # Make sure we have expenses structure
            if "expenses" not in result:
                result["expenses"] = {
                    "housing": 0,
                    "utilities": 0,
                    "groceries": 0,
                    "transportation": 0,
                    "healthcare": 0,
                    "entertainment": 0,
                    "dining_out": 0,
                    "subscriptions": 0,
                    "other": 0,
                    "total": 0
                }

                # Map user expenses to categories
                total_expenses = 0
                for expense in user_expenses:
                    if isinstance(expense, dict) and "name" in expense and "amount" in expense:
                        category = expense["name"].lower()
                        amount = float(expense["amount"])
                        total_expenses += amount

                        # Map to appropriate category
                        if "rent" in category or "mortgage" in category or "home" in category or "house" in category:
                            result["expenses"]["housing"] = amount
                        elif "util" in category or "electric" in category or "water" in category or "gas" in category:
                            result["expenses"]["utilities"] = amount
                        elif "groc" in category or "food" in category:
                            result["expenses"]["groceries"] = amount
                        elif "transport" in category or "car" in category or "fuel" in category or "gas" in category:
                            result["expenses"]["transportation"] = amount
                        elif "health" in category or "medical" in category or "insurance" in category:
                            result["expenses"]["healthcare"] = amount
                        elif "entertain" in category or "fun" in category or "recreation" in category:
                            result["expenses"]["entertainment"] = amount
                        elif "dining" in category or "restaurant" in category or "eat out" in category:
                            result["expenses"]["dining_out"] = amount
                        elif "subscript" in category or "streaming" in category or "netflix" in category:
                            result["expenses"]["subscriptions"] = amount
                        else:
                            # If no match, add to other
                            result["expenses"]["other"] += amount

                # Update total expenses
                result["expenses"]["total"] = total_expenses

            # Make sure we have balance structure
            if "balance" not in result:
                savings = user_income - result["expenses"]["total"]
                result["balance"] = {
                    "starting": 0,
                    "ending": savings,
                    "change": savings
                }

            # Ensure income is correct
            if "income" in result:
                if "salary" in result["income"]:
                    result["income"]["salary"] = user_income
                if "total" in result["income"]:
                    # Recalculate total income
                    result["income"]["total"] = user_income + result["income"].get("investments", 0) + result["income"].get("other", 0)

            # Wrap in a list for consistency
            result = [result]
        elif isinstance(result, list) and result:
            # Process each item in the list
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    # Convert old format to new format if needed
                    if "simulation_result" in item and "simulation_output" in item:
                        # Create a new result structure that uses ONLY the user's actual input values
                        new_item = {
                            "month": month,
                            "user_name": user_name,  # Use exact user_name from input
                            "income": {
                                "salary": user_income,  # Use exact income from input
                                "investments": 0,
                                "other": 0,
                                "total": user_income
                            },
                            "expenses": {
                                "housing": 0,
                                "utilities": 0,
                                "groceries": 0,
                                "transportation": 0,
                                "healthcare": 0,
                                "entertainment": 0,
                                "dining_out": 0,
                                "subscriptions": 0,
                                "other": 0,
                                "total": 0  # Will be calculated from actual expenses
                            },
                            "balance": {
                                "starting": 0,
                                "ending": 0,
                                "change": 0
                            },
                            "notes": "Based on your provided data."
                        }

                        # Always use the user's actual expenses
                        # Map user expenses to categories
                        total_expenses = 0
                        for expense in user_expenses:
                            if isinstance(expense, dict) and "name" in expense and "amount" in expense:
                                category = expense["name"].lower()
                                amount = float(expense["amount"])
                                total_expenses += amount

                                # Map common expense categories
                                if "rent" in category or "mortgage" in category or "home" in category or "house" in category:
                                    new_item["expenses"]["housing"] = amount
                                elif "util" in category or "electric" in category or "water" in category or "gas" in category:
                                    new_item["expenses"]["utilities"] = amount
                                elif "groc" in category or "food" in category:
                                    new_item["expenses"]["groceries"] = amount
                                elif "transport" in category or "car" in category or "fuel" in category or "gas" in category:
                                    new_item["expenses"]["transportation"] = amount
                                elif "health" in category or "medical" in category or "insurance" in category:
                                    new_item["expenses"]["healthcare"] = amount
                                elif "entertain" in category or "fun" in category or "recreation" in category:
                                    new_item["expenses"]["entertainment"] = amount
                                elif "dining" in category or "restaurant" in category or "eat out" in category:
                                    new_item["expenses"]["dining_out"] = amount
                                elif "subscript" in category or "streaming" in category or "netflix" in category:
                                    new_item["expenses"]["subscriptions"] = amount
                                else:
                                    # If no match, add to other
                                    new_item["expenses"]["other"] += amount

                        # Update total expenses
                        new_item["expenses"]["total"] = total_expenses

                        # Update balance
                        savings = user_income - total_expenses
                        new_item["balance"]["change"] = savings
                        new_item["balance"]["ending"] = savings

                        result[i] = new_item
                    else:
                        # Ensure required fields exist
                        item["month"] = month
                        item["user_name"] = user_name

                        # Ensure we have a complete structure with all fields for all months
                        # Make sure we have income structure
                        if "income" not in item:
                            item["income"] = {
                                "salary": user_income,
                                "investments": 0,
                                "other": 0,
                                "total": user_income
                            }

                        # Make sure we have expenses structure
                        if "expenses" not in item:
                            item["expenses"] = {
                                "housing": 0,
                                "utilities": 0,
                                "groceries": 0,
                                "transportation": 0,
                                "healthcare": 0,
                                "entertainment": 0,
                                "dining_out": 0,
                                "subscriptions": 0,
                                "other": 0,
                                "total": 0
                            }

                            # Map user expenses to categories
                            total_expenses = 0
                            for expense in user_expenses:
                                if isinstance(expense, dict) and "name" in expense and "amount" in expense:
                                    category = expense["name"].lower()
                                    amount = float(expense["amount"])
                                    total_expenses += amount

                                    # Map to appropriate category
                                    if "rent" in category or "mortgage" in category or "home" in category or "house" in category:
                                        item["expenses"]["housing"] = amount
                                    elif "util" in category or "electric" in category or "water" in category or "gas" in category:
                                        item["expenses"]["utilities"] = amount
                                    elif "groc" in category or "food" in category:
                                        item["expenses"]["groceries"] = amount
                                    elif "transport" in category or "car" in category or "fuel" in category or "gas" in category:
                                        item["expenses"]["transportation"] = amount
                                    elif "health" in category or "medical" in category or "insurance" in category:
                                        item["expenses"]["healthcare"] = amount
                                    elif "entertain" in category or "fun" in category or "recreation" in category:
                                        item["expenses"]["entertainment"] = amount
                                    elif "dining" in category or "restaurant" in category or "eat out" in category:
                                        item["expenses"]["dining_out"] = amount
                                    elif "subscript" in category or "streaming" in category or "netflix" in category:
                                        item["expenses"]["subscriptions"] = amount
                                    else:
                                        # If no match, add to other
                                        item["expenses"]["other"] += amount

                            # Update total expenses
                            item["expenses"]["total"] = total_expenses

                        # Make sure we have balance structure
                        if "balance" not in item:
                            savings = user_income - item["expenses"]["total"]
                            item["balance"] = {
                                "starting": 0,
                                "ending": savings,
                                "change": savings
                            }

                        # Ensure income is correct
                        if "income" in item:
                            if "salary" in item["income"]:
                                item["income"]["salary"] = user_income
                            if "total" in item["income"]:
                                # Recalculate total income
                                item["income"]["total"] = user_income + item["income"].get("investments", 0) + item["income"].get("other", 0)

        # Save to file system
        deduplicate_and_save(output_path, result)

        # Save to MongoDB
        agent_name = "cashflow"
        save_agent_output(
            user_id=user_id,
            simulation_id=state["simulation_id"],
            month=month,
            agent_name=agent_name,
            output_data=result[0] if result else {}
        )
        print(f"💾 Saved {agent_name} output to MongoDB for month {month}")

        # Update state
        return {
            **state,
            "cashflow_result": result
        }
    except Exception as e:
        print(f"❌ Error in simulate_cashflow_node: {e}")
        # Create a fallback result with user data
        # Create a customized fallback based on actual user data
        fallback = create_fallback_json(month, "cashflow", state["user_inputs"])

        return {
            **state,
            "cashflow_result": [fallback]
        }

def discipline_tracker_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Track financial discipline for the current month."""
    print(f"🟢 Executing task: discipline_tracker for month {state['month_number']}")

    # Build context from previous cashflow results
    user_id = state["user_inputs"].get("user_id", "default_user")
    month = state["month_number"]
    cashflow_context = build_simulated_cashflow_context(month, user_id)

    # Load agent config from YAML
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config", "agents.yaml")
    with open(config_path, "r") as f:
        agents_config = yaml.safe_load(f)

    # Load task config from YAML (use absolute path)
    tasks_config_path = os.path.join(base_dir, "config", "tasks.yaml")
    with open(tasks_config_path, "r") as f:
        tasks_config = yaml.safe_load(f)

    # Get the task description
    task_description = tasks_config.get("discipline_tracker_task", {}).get("description", "")

    # Get previous month data if available
    previous_month_context = ""

    if month > 1 and state.get("previous_month_data"):
        previous_discipline_data = state["previous_month_data"].get("discipline", [])
        if previous_discipline_data:
            previous_month_context = f"""
\n\nPrevious Month Discipline Data: {json.dumps(previous_discipline_data, indent=2)}

IMPORTANT LEARNING INSTRUCTIONS:
1. Compare current month's financial behavior with previous month's discipline score and violations
2. Identify if the user has improved or worsened in specific areas
3. Recognize patterns of repeated violations or improvements
4. Provide more targeted recommendations based on historical discipline issues
5. Acknowledge improvements where the user has followed previous recommendations
6. Adjust discipline scoring to reflect progressive learning and improvement
"""
            print(f"📊 Using previous month discipline data for month {month-1}")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a financial discipline tracking assistant. Always respond ONLY with valid JSON.

Your role is to track financial discipline over time and show progressive learning:
1. Compare current behavior with previous months' patterns
2. Recognize improvements or regressions in financial discipline
3. Provide increasingly personalized recommendations based on historical data
4. Adjust your scoring to reflect the user's learning journey
5. Be more strict about repeated violations and more rewarding of consistent improvements
"""),
        HumanMessage(content=task_description +
                    "\n\nUser Inputs: {user_inputs}\nCashflow Context: {cashflow_context}\nCashflow Result: {cashflow_result}" +
                    previous_month_context)
    ])

    # Get LLM
    llm = get_llm(agents_config.get("discipline_tracker_agent", {}).get("llm", "groq/llama3-70b-8192"))

    # Create chain
    chain = prompt | llm | JsonOutputParser()

    # Execute chain
    try:
        # Try to get the result from the chain
        try:
            raw_result = chain.invoke({
                "user_inputs": state["user_inputs"],
                "cashflow_context": cashflow_context,
                "cashflow_result": state["cashflow_result"]
            })
            result = raw_result  # If successful, use the result directly
        except Exception as e:
            # If the chain fails, try to extract the raw output and parse it manually
            print(f"⚠️ Warning: JSON parsing failed in discipline_tracker_node: {e}")

            # Get the raw output from the error message if possible
            error_str = str(e)
            if "Invalid json output:" in error_str:
                raw_json_str = error_str.split("Invalid json output:", 1)[1].strip()
                # Try to parse the raw JSON with our custom parser
                parsed_result = safe_parse_json(raw_json_str)
                if parsed_result:
                    print(f"✅ Successfully recovered JSON in discipline_tracker_node using custom parser")
                    result = parsed_result
                else:
                    # If parsing still fails, create a fallback JSON
                    print(f"⚠️ Could not parse JSON in discipline_tracker_node, using fallback")
                    result = create_fallback_json(month, "discipline_tracker", state["user_inputs"])
            else:
                # If we can't extract the raw JSON, create a fallback
                result = create_fallback_json(month, "discipline_tracker", state["user_inputs"])

        # Save result to file
        output_path = f"output/{user_id}_discipline_report_simulation.json"

        # Ensure it's a list with the month number
        if isinstance(result, dict):
            result["month"] = month
            result = [result]
        elif isinstance(result, list) and result:
            for item in result:
                if isinstance(item, dict):
                    item["month"] = month

        # Save to file system
        deduplicate_and_save(output_path, result)

        # Save to MongoDB
        agent_name = "discipline_tracker"
        save_agent_output(
            user_id=user_id,
            simulation_id=state["simulation_id"],
            month=month,
            agent_name=agent_name,
            output_data=result[0] if result else {}
        )
        print(f"💾 Saved {agent_name} output to MongoDB for month {month}")

        # Update state
        return {
            **state,
            "discipline_result": result,
            "discipline_context": cashflow_context
        }
    except Exception as e:
        print(f"❌ Error in discipline_tracker_node: {e}")
        # Create a fallback result
        fallback = create_fallback_json(month, "discipline_tracker", state["user_inputs"])
        return {
            **state,
            "discipline_result": [fallback],
            "discipline_context": cashflow_context
        }

def goal_tracker_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Track financial goals using enhanced analysis."""
    print(f"🟢 Executing enhanced goal tracking for month {state['month_number']}")

    try:
        # Get previous month data for enhanced analysis
        user_id = state["user_inputs"].get("user_id", "default_user")
        month = state["month_number"]
        previous_data = None

        if month > 1 and state.get("previous_month_data"):
            # Get cashflow data for savings calculation
            previous_data = state["previous_month_data"].get("cashflow", [])
            print(f"📊 Using {len(previous_data) if previous_data else 0} previous months for goal tracking")

        # Use enhanced goal tracking
        result = generate_enhanced_goal_tracking(
            user_inputs=state["user_inputs"],
            month=month,
            previous_data=previous_data
        )

        print(f"✅ Enhanced goal tracking completed for month {month}")

        # Get goal progress info for logging
        if result.get("summary"):
            progress = result["summary"].get("overall_progress", 0)
            total_saved = result["summary"].get("total_saved", 0)
            print(f"🎯 Goal Progress: {progress:.1f}% (₹{total_saved:,.0f} saved)")

        # Format result as list for consistency
        result_list = [result] if isinstance(result, dict) else result

    except Exception as e:
        print(f"❌ Error in enhanced goal tracking: {e}")

        # Fallback to basic goal tracking
        print("🔄 Falling back to basic goal tracking...")

        goal = state["user_inputs"].get("goal", "Financial Goal")
        income = float(state["user_inputs"].get("income", 0))
        total_expenses = float(state["user_inputs"].get("total_expenses", 0))
        monthly_surplus = income - total_expenses

        result_list = [{
            "month": month,
            "goals": [{
                "name": goal,
                "target_amount": 50000,  # Default target
                "saved_so_far": monthly_surplus * month,  # Estimate
                "expected_by_now": 50000 / 36 * month,  # 3-year timeline
                "status": "on_track" if monthly_surplus > 0 else "behind",
                "priority": 1,
                "adjustment_suggestion": "Continue saving consistently" if monthly_surplus > 0 else "Increase income or reduce expenses"
            }],
            "summary": {
                "on_track_goals": 1 if monthly_surplus > 0 else 0,
                "behind_goals": 0 if monthly_surplus > 0 else 1,
                "total_saved": monthly_surplus * month,
                "total_required_by_now": 50000 / 36 * month
            },
            "fallback_used": True,
            "error": str(e)
        }]

    # Save result to file
    output_path = f"output/{user_id}_goal_status_simulation.json"

    # Ensure each item has the month number
    for item in result_list:
        if isinstance(item, dict):
            item["month"] = month

    deduplicate_and_save(output_path, result_list)

    # Save to MongoDB
    agent_name = "goal_tracker"
    save_agent_output(
        user_id=user_id,
        simulation_id=state["simulation_id"],
        month=month,
        agent_name=agent_name,
        output_data=result_list[0] if result_list else {}
    )
    print(f"💾 Saved enhanced {agent_name} output to MongoDB for month {month}")

    # Update state
    return {
        **state,
        "goal_tracking_result": result_list,
        "goal_tracking_context": f"Enhanced goal tracking for month {month}"
    }

def behavior_tracker_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Track financial behavior for the current month."""
    print(f"🟢 Executing task: behavior_tracker for month {state['month_number']}")

    # Build context from previous results
    user_id = state["user_inputs"].get("user_id", "default_user")
    month = state["month_number"]
    behavior_context = build_behavior_tracker_context(month, user_id)

    # Load agent config from YAML
    import os
    import yaml
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    agents_config_path = os.path.join(config_dir, "agents.yaml")
    tasks_config_path = os.path.join(config_dir, "tasks.yaml")

    with open(agents_config_path, "r") as f:
        agents_config = yaml.safe_load(f)

    # Load task config from YAML
    with open(tasks_config_path, "r") as f:
        tasks_config = yaml.safe_load(f)

    # Get the task description
    task_description = tasks_config.get("behavior_tracker_task", {}).get("description", "")

    # Get previous month data if available
    previous_month_context = ""

    if month > 1 and state.get("previous_month_data"):
        previous_behavior_data = state["previous_month_data"].get("behavior", [])
        if previous_behavior_data:
            previous_month_context = f"""
\n\nPrevious Month Behavior Data: {json.dumps(previous_behavior_data, indent=2)}

IMPORTANT LEARNING INSTRUCTIONS:
1. Compare current financial behaviors with patterns from previous months
2. Identify behavioral improvements or regressions over time
3. Note recurring behavioral patterns that affect financial outcomes
4. Provide more targeted behavioral insights based on historical patterns
5. Recognize when the user has successfully changed problematic behaviors
6. Adjust behavioral recommendations based on what has or hasn't worked in the past
"""
            print(f"📊 Using previous month behavior data for month {month-1}")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a financial behavior tracking assistant. Always respond ONLY with valid JSON.

Your role is to analyze financial behaviors over time and demonstrate progressive learning:
1. Track behavioral patterns across multiple months, not just the current month
2. Identify trends in financial decision-making (improving, declining, or stagnant)
3. Provide increasingly personalized behavioral insights based on historical patterns
4. Recognize when behavioral patterns need special attention based on consistent issues
5. Show how current behaviors compare to previous months and highlight improvements
"""),
        HumanMessage(content=task_description +
                    "\n\nUser Inputs: {user_inputs}\nCashflow Result: {cashflow_result}\nDiscipline Result: {discipline_result}\nGoal Tracking Result: {goal_tracking_result}" +
                    previous_month_context)
    ])

    # Get LLM
    llm = get_llm(agents_config.get("behavior_tracker_agent", {}).get("llm", "groq/llama-3.1-70b-versatile"))

    # Create chain
    chain = prompt | llm | JsonOutputParser()

    # Execute chain
    try:
        # Try to get the result from the chain
        try:
            raw_result = chain.invoke({
                "user_inputs": state["user_inputs"],
                "cashflow_result": state["cashflow_result"],
                "discipline_result": state["discipline_result"],
                "goal_tracking_result": state["goal_tracking_result"]
            })
            result = raw_result  # If successful, use the result directly
        except Exception as e:
            # If the chain fails, try to extract the raw output and parse it manually
            print(f"⚠️ Warning: JSON parsing failed in behavior_tracker_node: {e}")

            # Get the raw output from the error message if possible
            error_str = str(e)
            if "Invalid json output:" in error_str:
                raw_json_str = error_str.split("Invalid json output:", 1)[1].strip()
                # Try to parse the raw JSON with our custom parser
                parsed_result = safe_parse_json(raw_json_str)
                if parsed_result:
                    print(f"✅ Successfully recovered JSON in behavior_tracker_node using custom parser")
                    result = parsed_result
                else:
                    # If parsing still fails, create a fallback JSON
                    print(f"⚠️ Could not parse JSON in behavior_tracker_node, using fallback")
                    result = create_fallback_json(month, "behavior_tracker", state["user_inputs"])
            else:
                # If we can't extract the raw JSON, create a fallback
                result = create_fallback_json(month, "behavior_tracker", state["user_inputs"])

        # Save result to file
        output_path = f"output/{user_id}_behavior_tracker_simulation.json"

        # Ensure it's a list with the month number
        if isinstance(result, dict):
            result["month"] = month
            result = [result]
        elif isinstance(result, list) and result:
            for item in result:
                if isinstance(item, dict):
                    item["month"] = month

        deduplicate_and_save(output_path, result)

        # Save to MongoDB
        agent_name = "behavior_tracker"
        save_agent_output(
            user_id=user_id,
            simulation_id=state["simulation_id"],
            month=month,
            agent_name=agent_name,
            output_data=result[0] if result else {}
        )
        print(f"💾 Saved {agent_name} output to MongoDB for month {month}")

        # Update state
        return {
            **state,
            "behavior_result": result,
            "behavior_context": behavior_context
        }
    except Exception as e:
        print(f"❌ Error in behavior_tracker_node: {e}")
        # Create a fallback result
        fallback = create_fallback_json(month, "behavior_tracker", state["user_inputs"])
        return {
            **state,
            "behavior_result": [fallback],
            "behavior_context": behavior_context
        }

def karma_tracker_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Track financial karma for the current month."""
    print(f"🟢 Executing task: karma_tracker for month {state['month_number']}")

    # Build context from previous results
    user_id = state["user_inputs"].get("user_id", "default_user")
    month = state["month_number"]
    karma_context = build_karmic_tracker_context(month, user_id)

    # Load agent config from YAML
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config", "agents.yaml")
    with open(config_path, "r") as f:
        agents_config = yaml.safe_load(f)

    # Load task config from YAML (use absolute path)
    tasks_config_path = os.path.join(base_dir, "config", "tasks.yaml")
    with open(tasks_config_path, "r") as f:
        tasks_config = yaml.safe_load(f)

    # Get the task description
    task_description = tasks_config.get("karma_tracker_task", {}).get("description", "")

    # Get previous month data if available
    previous_month_context = ""

    if month > 1 and state.get("previous_month_data"):
        previous_karma_data = state["previous_month_data"].get("karma", [])
        if previous_karma_data:
            previous_month_context = f"""
\n\nPrevious Month Karma Data: {json.dumps(previous_karma_data, indent=2)}

IMPORTANT LEARNING INSTRUCTIONS:
1. Compare current financial karma with patterns from previous months
2. Identify karmic improvements or regressions over time
3. Note recurring patterns in how financial decisions affect overall well-being
4. Provide more targeted karmic insights based on historical patterns
5. Recognize when the user has successfully improved their financial karma
6. Adjust karmic recommendations based on what has or hasn't worked in the past
"""
            print(f"📊 Using previous month karma data for month {month-1}")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a financial karma tracking assistant. Always respond ONLY with valid JSON.

Your role is to analyze financial karma over time and demonstrate progressive learning:
1. Track karmic patterns across multiple months, not just the current month
2. Identify trends in how financial decisions affect overall well-being
3. Provide increasingly personalized karmic insights based on historical patterns
4. Recognize when karmic patterns need special attention based on consistent issues
5. Show how current karma compares to previous months and highlight improvements
"""),
        HumanMessage(content=task_description +
                    "\n\nUser Inputs: {user_inputs}\nCashflow Result: {cashflow_result}\nDiscipline Result: {discipline_result}\nGoal Tracking Result: {goal_tracking_result}\nBehavior Result: {behavior_result}" +
                    previous_month_context)
    ])

    # Get LLM
    llm = get_llm(agents_config.get("karma_tracker_agent", {}).get("llm", "groq/llama-3.1-70b-versatile"))

    # Create chain
    chain = prompt | llm | JsonOutputParser()

    # Execute chain
    try:
        # Try to get the result from the chain
        try:
            raw_result = chain.invoke({
                "user_inputs": state["user_inputs"],
                "cashflow_result": state["cashflow_result"],
                "discipline_result": state["discipline_result"],
                "goal_tracking_result": state["goal_tracking_result"],
                "behavior_result": state["behavior_result"]
            })
            result = raw_result  # If successful, use the result directly
        except Exception as e:
            # If the chain fails, try to extract the raw output and parse it manually
            print(f"⚠️ Warning: JSON parsing failed in karma_tracker_node: {e}")

            # Get the raw output from the error message if possible
            error_str = str(e)
            if "Invalid json output:" in error_str:
                raw_json_str = error_str.split("Invalid json output:", 1)[1].strip()
                # Try to parse the raw JSON with our custom parser
                parsed_result = safe_parse_json(raw_json_str)
                if parsed_result:
                    print(f"✅ Successfully recovered JSON in karma_tracker_node using custom parser")
                    result = parsed_result
                else:
                    # If parsing still fails, create a fallback JSON
                    print(f"⚠️ Could not parse JSON in karma_tracker_node, using fallback")
                    result = create_fallback_json(month, "karma_tracker", state["user_inputs"])
            else:
                # If we can't extract the raw JSON, create a fallback
                result = create_fallback_json(month, "karma_tracker", state["user_inputs"])

        # Save result to file
        output_path = f"output/{user_id}_karmic_tracker_simulation.json"

        # Ensure it's a list with the month number
        if isinstance(result, dict):
            result["month"] = month
            result = [result]
        elif isinstance(result, list) and result:
            for item in result:
                if isinstance(item, dict):
                    item["month"] = month

        deduplicate_and_save(output_path, result)

        # Save to MongoDB
        agent_name = "karma_tracker"
        save_agent_output(
            user_id=user_id,
            simulation_id=state["simulation_id"],
            month=month,
            agent_name=agent_name,
            output_data=result[0] if result else {}
        )
        print(f"💾 Saved {agent_name} output to MongoDB for month {month}")

        # Update state
        return {
            **state,
            "karma_result": result,
            "karma_context": karma_context
        }
    except Exception as e:
        print(f"❌ Error in karma_tracker_node: {e}")
        # Create a fallback result
        fallback = create_fallback_json(month, "karma_tracker", state["user_inputs"])
        return {
            **state,
            "karma_result": [fallback],
            "karma_context": karma_context
        }

def financial_strategy_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Generate enhanced financial strategy for the current month."""
    print(f"🟢 Executing enhanced financial strategy for month {state['month_number']}")

    try:
        # Get previous month data for enhanced analysis
        user_id = state["user_inputs"].get("user_id", "default_user")
        month = state["month_number"]
        previous_data = None

        if month > 1 and state.get("previous_month_data"):
            # Combine all previous data types for comprehensive analysis
            previous_data = []
            for data_type in ["cashflow", "strategy", "discipline", "goal_tracking", "behavior", "karma"]:
                type_data = state["previous_month_data"].get(data_type, [])
                if type_data:
                    previous_data.extend(type_data)
            print(f"📊 Using {len(previous_data)} previous data points for enhanced strategy analysis")

        # Use enhanced financial strategy generation
        result = generate_enhanced_financial_strategy(
            user_inputs=state["user_inputs"],
            month=month,
            previous_data=previous_data
        )

        print(f"✅ Enhanced financial strategy completed for month {month}")
        print(f"📈 Generated {len(result.get('recommendations', []))} strategic recommendations")

        # Format result as list for consistency with existing system
        if isinstance(result, dict):
            result_list = [result]
        else:
            result_list = result if isinstance(result, list) else [result]

    except Exception as e:
        print(f"❌ Error in enhanced financial strategy: {e}")

        # Fallback to basic strategy generation
        print("🔄 Falling back to basic financial strategy...")

        # Basic strategy recommendations based on user profile
        income = float(state["user_inputs"].get("income", 0))
        total_expenses = float(state["user_inputs"].get("total_expenses", 0))
        monthly_surplus = income - total_expenses
        financial_type = state["user_inputs"].get("financial_type", "moderate")
        goal = state["user_inputs"].get("goal", "")

        basic_recommendations = []

        if monthly_surplus > 0:
            basic_recommendations.append({
                "type": "savings_optimization",
                "description": f"Continue saving ₹{monthly_surplus:,.0f} monthly to build financial security",
                "reasoning": "Positive cash flow provides opportunity for wealth building"
            })

            if "emergency" in goal.lower():
                basic_recommendations.append({
                    "type": "emergency_fund",
                    "description": "Prioritize building emergency fund before other investments",
                    "reasoning": "Emergency fund provides financial stability and peace of mind"
                })

            if financial_type.lower() == "moderate":
                basic_recommendations.append({
                    "type": "balanced_investment",
                    "description": "Consider balanced mutual funds for moderate risk investment",
                    "reasoning": "Balanced approach aligns with moderate risk tolerance"
                })
        else:
            basic_recommendations.append({
                "type": "expense_reduction",
                "description": "Focus on reducing expenses to create positive cash flow",
                "reasoning": "Negative cash flow prevents wealth building and goal achievement"
            })

        result_list = [{
            "month": month,
            "recommendations": basic_recommendations,
            "financial_health_score": 50,  # Neutral score
            "key_insights": [f"Basic strategy generated for month {month}"],
            "fallback_used": True,
            "error": str(e)
        }]

    # Save result to file
    output_path = f"output/{user_id}_financial_strategy_simulation.json"

    # Ensure each item has the month number
    for item in result_list:
        if isinstance(item, dict):
            item["month"] = month

    deduplicate_and_save(output_path, result_list)

    # Save to MongoDB
    agent_name = "financial_strategy"
    save_agent_output(
        user_id=user_id,
        simulation_id=state["simulation_id"],
        month=month,
        agent_name=agent_name,
        output_data=result_list[0] if result_list else {}
    )
    print(f"💾 Saved enhanced {agent_name} output to MongoDB for month {month}")

    # Update state
    return {
        **state,
        "financial_strategy_result": result_list,
        "financial_strategy_context": f"Enhanced financial strategy for month {month}"
    }

# Define the LangGraph workflow
def create_financial_simulation_graph():
    """Create the financial simulation workflow with adaptive scenarios."""
    workflow = StateGraph(FinancialSimulationState)

    # Add nodes
    workflow.add_node("simulate_cashflow", simulate_cashflow_node)
    workflow.add_node("discipline_tracker", discipline_tracker_node)
    workflow.add_node("goal_tracker", goal_tracker_node)
    workflow.add_node("behavior_tracker", behavior_tracker_node)
    workflow.add_node("karma_tracker", karma_tracker_node)
    workflow.add_node("financial_strategy", financial_strategy_node)

    # Define the edges (sequential workflow)
    workflow.add_edge("simulate_cashflow", "discipline_tracker")
    workflow.add_edge("discipline_tracker", "goal_tracker")
    workflow.add_edge("goal_tracker", "behavior_tracker")
    workflow.add_edge("behavior_tracker", "karma_tracker")
    workflow.add_edge("karma_tracker", "financial_strategy")
    workflow.add_edge("financial_strategy", END)

    # Set the entry point
    workflow.set_entry_point("simulate_cashflow")

    # Compile the graph
    return workflow.compile()

# Main simulation function
def simulate_timeline_langgraph(n_months: int, simulation_unit: str, user_inputs: dict, task_id: str = None, simulation_id: str = None, use_cache: bool = True):
    """Run the financial simulation for multiple months using LangGraph.

    Args:
        n_months: Number of months to simulate
        simulation_unit: Unit of simulation (e.g., "Months")
        user_inputs: User input data
        task_id: Optional task ID for status updates
        simulation_id: Optional simulation ID (if not provided, a new one will be generated)
        use_cache: Whether to use Redis cache for simulation results (default: True)
    """
    print(f"🚀 Starting LangGraph Financial Simulation for {n_months} {simulation_unit}...")

    # Create the workflow graph
    workflow = create_financial_simulation_graph()

    # Ensure user_id exists
    if "user_id" not in user_inputs:
        user_inputs["user_id"] = "default_user"

    # Ensure user_name exists
    if "user_name" not in user_inputs:
        user_inputs["user_name"] = "Default User"

    # Generate a unique simulation ID if not provided
    if not simulation_id:
        simulation_id = generate_simulation_id()
    print(f"📝 Simulation ID: {simulation_id}")

    # Check if simulation results are already in Redis cache
    if use_cache:
        cache_key = f"simulation:{simulation_id}"
        cached_results = redis_cache_get(cache_key)
        if cached_results and len(cached_results.get("months", [])) >= n_months:
            print(f"🔄 Using cached simulation results for simulation_id: {simulation_id}")
            return cached_results

    # Initialize cache structure if using cache
    if use_cache:
        redis_cache_set(
            f"simulation:{simulation_id}",
            {
                "simulation_id": simulation_id,
                "user_inputs": user_inputs,
                "n_months": n_months,
                "simulation_unit": simulation_unit,
                "status": "running",
                "start_time": time.time(),
                "months": []
            },
            expiry=86400  # Cache for 24 hours
        )

    # Save user input to MongoDB
    save_user_input(user_inputs, simulation_id)
    print(f"💾 Saved user input to MongoDB")

    # Run simulation for each month
    for month in range(1, n_months + 1):
        print(f"\n🔄 Simulating Month {month} of {n_months}")

        # Update task status if task_id is provided
        if task_id:
            print(f"📊 Updating task status: {month}/{n_months} months completed")

        # Simulate economic environment
        eco_env = EconomicEnvironment(unit=simulation_unit)
        eco_env.simulate_step()
        eco_context = eco_env.get_context()

        # Simulate market conditions
        _, market_context_summary = simulate_monthly_market()  # Using _ to ignore unused variable

        # Update user inputs with economic data
        month_inputs = user_inputs.copy()
        month_inputs["Month"] = month
        month_inputs["market_context"] = market_context_summary
        month_inputs["inflation"] = eco_context["inflation_rate"]
        month_inputs["interest_rate"] = eco_context["interest_rate"]
        month_inputs["cost_of_living_index"] = eco_context["cost_of_living_index"]

        # Fetch previous month data from MongoDB if not month 1
        previous_month_data = None
        if month > 1:
            user_id = user_inputs["user_id"]
            previous_month = month - 1

            # Get cashflow data
            cashflow_data = get_agent_outputs_for_month(user_id, previous_month, "cashflow")

            # Get discipline data
            discipline_data = get_agent_outputs_for_month(user_id, previous_month, "discipline_tracker")

            # Get goal tracking data
            goal_data = get_agent_outputs_for_month(user_id, previous_month, "goal_tracker")

            # Get behavior data
            behavior_data = get_agent_outputs_for_month(user_id, previous_month, "behavior_tracker")

            # Get karma data
            karma_data = get_agent_outputs_for_month(user_id, previous_month, "karma_tracker")

            # Get financial strategy data
            strategy_data = get_agent_outputs_for_month(user_id, previous_month, "financial_strategy")

            # Compile all data
            previous_month_data = {
                "cashflow": [item["data"] for item in cashflow_data],
                "discipline": [item["data"] for item in discipline_data],
                "goal": [item["data"] for item in goal_data],
                "behavior": [item["data"] for item in behavior_data],
                "karma": [item["data"] for item in karma_data],
                "strategy": [item["data"] for item in strategy_data]
            }

            print(f"📊 Fetched previous month data from MongoDB for month {previous_month}")

        # Initialize state
        initial_state = {
            "user_inputs": month_inputs,
            "month_number": month,
            "simulation_id": simulation_id,
            "cashflow_result": None,
            "discipline_result": None,
            "goal_tracking_result": None,
            "behavior_result": None,
            "karma_result": None,
            "financial_strategy_result": None,
            "cashflow_context": None,
            "discipline_context": None,
            "goal_tracking_context": None,
            "behavior_context": None,
            "karma_context": None,
            "financial_strategy_context": None,
            "economic_context": eco_context,
            "market_context": market_context_summary,
            "previous_month_data": previous_month_data
        }

        # Run the workflow
        try:
            # Execute the workflow directly instead of streaming
            result = workflow.invoke(initial_state)

            # Get the final state with all results
            final_state = result

            # Generate monthly reflection report
            user_id = user_inputs["user_id"]
            # Use user_id for file naming consistency
            assign_persona(user_id, month)
            generate_monthly_reflection_report(user_id, month)
            print(f"📝 Generated monthly reflection report for user_id: {user_id}, month: {month}")

            # Cache the month's results in Redis if using cache
            if use_cache:
                # Get current cache
                cache_key = f"simulation:{simulation_id}"
                cached_data = redis_cache_get(cache_key) or {
                    "simulation_id": simulation_id,
                    "user_inputs": user_inputs,
                    "n_months": n_months,
                    "simulation_unit": simulation_unit,
                    "status": "running",
                    "start_time": time.time(),
                    "months": []
                }

                # Add this month's results
                month_data = {
                    "month": month,
                    "cashflow_result": final_state.get("cashflow_result"),
                    "discipline_result": final_state.get("discipline_result"),
                    "goal_tracking_result": final_state.get("goal_tracking_result"),
                    "behavior_result": final_state.get("behavior_result"),
                    "karma_result": final_state.get("karma_result"),
                    "financial_strategy_result": final_state.get("financial_strategy_result"),
                    "economic_context": final_state.get("economic_context"),
                    "market_context": final_state.get("market_context")
                }

                # Add to months array
                cached_data["months"].append(month_data)

                # Update cache
                redis_cache_set(cache_key, cached_data, expiry=86400)  # Cache for 24 hours

            print(f"✅ Month {month} simulation completed successfully")

        except Exception as e:
            print(f"❌ Error in month {month} simulation: {e}")

            # Update cache with error status if using cache
            if use_cache:
                cache_key = f"simulation:{simulation_id}"
                cached_data = redis_cache_get(cache_key)
                if cached_data:
                    cached_data["error"] = str(e)
                    cached_data["status"] = "error"
                    redis_cache_set(cache_key, cached_data, expiry=86400)

            import traceback
            traceback.print_exc()

        # Add delay between months to avoid rate limits
        if month < n_months:
            print(f"⏳ Waiting before starting next month simulation...")
            time.sleep(15)

    # Update final status in cache if using cache
    if use_cache:
        cache_key = f"simulation:{simulation_id}"
        cached_data = redis_cache_get(cache_key)
        if cached_data:
            cached_data["status"] = "completed"
            cached_data["end_time"] = time.time()
            cached_data["duration"] = cached_data["end_time"] - cached_data["start_time"]
            redis_cache_set(cache_key, cached_data, expiry=86400)

    print(f"🎉 Financial simulation completed for {n_months} {simulation_unit}")

    # Return the cached data if using cache, otherwise return True
    if use_cache:
        return redis_cache_get(f"simulation:{simulation_id}")
    return True

# For testing
if __name__ == "__main__":
    # Test inputs
    test_inputs = {
        "user_id": "test_user",
        "user_name": "Test User",
        "age": 30,
        "occupation": "Software Engineer",
        "income_level": "50,000-100,000",
        "goal": "Save $10,000 for emergency fund",
        "starting_balance": 5000,
        "monthly_earning": 6000,
        "monthly_expenses": 4500,
        "savings_target": 1500
    }

    # Run simulation for 2 months
    simulate_timeline_langgraph(2, "Months", test_inputs)
