"""
Example usage of the Memory Management API.

This module demonstrates how to use the Memory Management API
for storing and retrieving memories and interactions.
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

import httpx

# Configuration
API_BASE_URL = "http://localhost:8003"
API_KEY = "memory_api_key_dev"  # Use your actual API key

# Headers for API requests
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


class MemoryAPIClient:
    """Client for interacting with the Memory Management API."""
    
    def __init__(self, base_url: str = API_BASE_URL, api_key: str = API_KEY):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/memory/health")
            return response.json()
    
    async def store_memory(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store a memory chunk."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/memory",
                json=memory_data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def store_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store an interaction."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/memory/interaction",
                json=interaction_data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def get_memories_by_persona(
        self, 
        persona_id: str, 
        user_id: str = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get memories for a persona."""
        params = {"persona": persona_id, "limit": limit}
        if user_id:
            params["user_id"] = user_id
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory",
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def get_recent_interactions(
        self, 
        limit: int = 10,
        user_id: str = None,
        persona_id: str = None
    ) -> Dict[str, Any]:
        """Get recent interactions."""
        params = {"limit": limit, "recent_interactions": "true"}
        if user_id:
            params["user_id"] = user_id
        if persona_id:
            params["persona"] = persona_id
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory",
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def search_memories(
        self, 
        query: str,
        persona_id: str = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Search memories."""
        params = {"query": query, "limit": limit}
        if persona_id:
            params["persona_id"] = persona_id
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory/search",
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()


async def example_financial_advisor_scenario():
    """Example scenario: Financial advisor persona with user interactions."""
    client = MemoryAPIClient()
    
    print("🏦 Financial Advisor Memory Management Example")
    print("=" * 50)
    
    # Check API health
    try:
        health = await client.health_check()
        print(f"✅ API Health: {health['status']}")
    except Exception as e:
        print(f"❌ API Health Check Failed: {e}")
        return
    
    # User and persona information
    user_id = "user_john_doe"
    persona_id = "financial_advisor"
    
    # 1. Store user preferences as memories
    print("\n📝 Storing user preferences...")
    
    preferences = [
        {
            "user_id": user_id,
            "persona_id": persona_id,
            "content": "User prefers conservative investment strategies with low risk",
            "content_type": "preference",
            "metadata": {
                "tags": ["investment", "conservative", "low-risk"],
                "importance": 8,
                "topic": "investment_strategy",
                "source": "user_input"
            }
        },
        {
            "user_id": user_id,
            "persona_id": persona_id,
            "content": "User has a monthly income of $5000 and wants to save $1000 per month",
            "content_type": "fact",
            "metadata": {
                "tags": ["income", "savings", "budget"],
                "importance": 9,
                "topic": "financial_profile",
                "source": "user_input"
            }
        },
        {
            "user_id": user_id,
            "persona_id": persona_id,
            "content": "User is 35 years old and planning for retirement in 30 years",
            "content_type": "fact",
            "metadata": {
                "tags": ["age", "retirement", "planning"],
                "importance": 7,
                "topic": "retirement_planning",
                "source": "user_input"
            }
        }
    ]
    
    stored_memories = []
    for pref in preferences:
        try:
            result = await client.store_memory(pref)
            memory_id = result["data"]["memory_id"]
            stored_memories.append(memory_id)
            print(f"  ✅ Stored memory: {memory_id}")
        except Exception as e:
            print(f"  ❌ Failed to store memory: {e}")
    
    # 2. Store some interactions
    print("\n💬 Storing user interactions...")
    
    interactions = [
        {
            "user_id": user_id,
            "persona_id": persona_id,
            "user_message": "What's the best investment strategy for someone my age?",
            "agent_response": "For a 35-year-old with 30 years until retirement, I recommend a balanced portfolio with 70% stocks and 30% bonds. Given your conservative preference, we can adjust this to 60% stocks and 40% bonds.",
            "context": {
                "session_id": "session_001",
                "conversation_turn": 1,
                "domain": "finance",
                "intent": "investment_advice"
            },
            "metadata": {
                "response_time": 1.5,
                "confidence": 0.92,
                "model_used": "financial_advisor_v1"
            }
        },
        {
            "user_id": user_id,
            "persona_id": persona_id,
            "user_message": "How much should I save for retirement?",
            "agent_response": "Based on your $5000 monthly income and goal to save $1000 per month, you're on track to save $360,000 over 30 years. With compound interest at 7% annual return, this could grow to approximately $1 million.",
            "context": {
                "session_id": "session_001",
                "conversation_turn": 2,
                "domain": "finance",
                "intent": "retirement_planning"
            },
            "metadata": {
                "response_time": 2.1,
                "confidence": 0.89,
                "model_used": "financial_advisor_v1"
            }
        }
    ]
    
    stored_interactions = []
    for interaction in interactions:
        try:
            result = await client.store_interaction(interaction)
            interaction_id = result["data"]["interaction_id"]
            stored_interactions.append(interaction_id)
            print(f"  ✅ Stored interaction: {interaction_id}")
        except Exception as e:
            print(f"  ❌ Failed to store interaction: {e}")
    
    # 3. Retrieve memories for the persona
    print(f"\n🔍 Retrieving memories for persona: {persona_id}")
    
    try:
        memories = await client.get_memories_by_persona(persona_id, user_id, limit=10)
        print(f"  📊 Found {memories['total_count']} memories")
        
        for memory in memories['memories'][:3]:  # Show first 3
            content = memory['content'][:60] + "..." if len(memory['content']) > 60 else memory['content']
            print(f"    - {memory['content_type']}: {content}")
            print(f"      Tags: {', '.join(memory['metadata']['tags'])}")
            print(f"      Importance: {memory['metadata']['importance']}/10")
    except Exception as e:
        print(f"  ❌ Failed to retrieve memories: {e}")
    
    # 4. Get recent interactions for context
    print(f"\n🕒 Retrieving recent interactions...")
    
    try:
        interactions = await client.get_recent_interactions(limit=5, user_id=user_id)
        print(f"  📊 Found {interactions['total_count']} recent interactions")
        
        for interaction in interactions['interactions']:
            user_msg = interaction['user_message'][:50] + "..." if len(interaction['user_message']) > 50 else interaction['user_message']
            agent_resp = interaction['agent_response'][:50] + "..." if len(interaction['agent_response']) > 50 else interaction['agent_response']
            print(f"    👤 User: {user_msg}")
            print(f"    🤖 Agent: {agent_resp}")
            print(f"    ⏱️  Response time: {interaction['metadata']['response_time']}s")
            print()
    except Exception as e:
        print(f"  ❌ Failed to retrieve interactions: {e}")
    
    # 5. Search for specific topics
    print(f"\n🔎 Searching for 'investment' related memories...")
    
    try:
        search_results = await client.search_memories("investment", persona_id, limit=5)
        print(f"  📊 Found {search_results['total_results']} results in {search_results['search_time']:.3f}s")
        
        for result in search_results['results']:
            content = result['content'][:80] + "..." if len(result['content']) > 80 else result['content']
            print(f"    - {content}")
            print(f"      Topic: {result['metadata'].get('topic', 'N/A')}")
    except Exception as e:
        print(f"  ❌ Failed to search memories: {e}")
    
    print("\n✅ Financial advisor scenario completed!")


async def example_gurukul_tutor_scenario():
    """Example scenario: Gurukul tutor persona with educational interactions."""
    client = MemoryAPIClient()
    
    print("\n📚 Gurukul Tutor Memory Management Example")
    print("=" * 50)
    
    user_id = "student_alice"
    persona_id = "math_tutor"
    
    # Store learning preferences and progress
    learning_memories = [
        {
            "user_id": user_id,
            "persona_id": persona_id,
            "content": "Student struggles with algebra but excels in geometry",
            "content_type": "fact",
            "metadata": {
                "tags": ["algebra", "geometry", "learning_style"],
                "importance": 8,
                "topic": "math_skills",
                "source": "assessment"
            }
        },
        {
            "user_id": user_id,
            "persona_id": persona_id,
            "content": "Student prefers visual learning methods over textual explanations",
            "content_type": "preference",
            "metadata": {
                "tags": ["visual", "learning_style", "preference"],
                "importance": 7,
                "topic": "teaching_method",
                "source": "observation"
            }
        }
    ]
    
    print("📝 Storing learning profile...")
    for memory in learning_memories:
        try:
            result = await client.store_memory(memory)
            print(f"  ✅ Stored: {result['data']['memory_id']}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Store tutoring interactions
    tutoring_interactions = [
        {
            "user_id": user_id,
            "persona_id": persona_id,
            "user_message": "I don't understand how to solve quadratic equations",
            "agent_response": "Let me show you a visual method using the quadratic formula. Think of it as finding where a parabola crosses the x-axis.",
            "context": {
                "session_id": "lesson_001",
                "conversation_turn": 1,
                "domain": "gurukul",
                "intent": "help_request"
            },
            "metadata": {
                "response_time": 2.0,
                "confidence": 0.95,
                "model_used": "math_tutor_v2"
            }
        }
    ]
    
    print("\n💬 Storing tutoring interactions...")
    for interaction in tutoring_interactions:
        try:
            result = await client.store_interaction(interaction)
            print(f"  ✅ Stored: {result['data']['interaction_id']}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n✅ Gurukul tutor scenario completed!")


async def main():
    """Run all example scenarios."""
    print("🚀 Memory Management API Examples")
    print("=" * 60)
    
    try:
        await example_financial_advisor_scenario()
        await example_gurukul_tutor_scenario()
        
        print("\n🎉 All examples completed successfully!")
        print("\n📖 Next steps:")
        print("  - Check the API documentation at http://localhost:8003/memory/docs")
        print("  - Explore the stored data in your MongoDB database")
        print("  - Try the search and filtering capabilities")
        print("  - Implement your own persona-specific memory patterns")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        print("Make sure the Memory Management API is running on http://localhost:8003")


if __name__ == "__main__":
    asyncio.run(main())
