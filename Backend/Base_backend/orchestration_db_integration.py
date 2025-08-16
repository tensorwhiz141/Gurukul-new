"""
Database Integration Layer for Orchestration System
Manages MongoDB integration between Base_backend and Orchestration system
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from db import user_collection, subjects_collection, lectures_collection, tests_collection
from orchestration_config import config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestrationDBIntegration:
    """Database integration layer for orchestration system"""
    
    def __init__(self):
        self.config = config
    
    def store_enhanced_lesson(self, lesson_data: Dict[str, Any]) -> str:
        """Store enhanced lesson in MongoDB with orchestration metadata"""
        try:
            # Prepare lesson document for storage
            lesson_doc = {
                **lesson_data,
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "collection_type": "enhanced_lesson",
                "orchestration_version": "1.0.0",
                "integration_source": "base_backend_orchestration"
            }
            
            # Insert into lectures collection
            result = lectures_collection.insert_one(lesson_doc)
            lesson_id = str(result.inserted_id)
            
            # Log the storage
            if self.config.LOG_ORCHESTRATION_CALLS:
                logger.info(f"Enhanced lesson stored: {lesson_id} for user {lesson_data.get('user_id')}")
            
            return lesson_id
            
        except Exception as e:
            logger.error(f"Failed to store enhanced lesson: {e}")
            raise
    
    def get_user_lesson_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's lesson history from MongoDB"""
        try:
            # Query lessons for the user
            lessons = list(lectures_collection.find(
                {"user_id": user_id},
                {"_id": 0}  # Exclude MongoDB _id field
            ).sort("generated_at", -1).limit(limit))
            
            return lessons
            
        except Exception as e:
            logger.error(f"Failed to get user lesson history: {e}")
            return []
    
    def store_user_progress(self, user_id: str, progress_data: Dict[str, Any]) -> bool:
        """Store user progress data in MongoDB"""
        try:
            # Prepare progress document
            progress_doc = {
                "user_id": user_id,
                "progress_data": progress_data,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "source": "orchestration_integration"
            }
            
            # Upsert user progress (update if exists, insert if not)
            user_collection.update_one(
                {"user_id": user_id, "type": "progress_tracking"},
                {"$set": progress_doc},
                upsert=True
            )
            
            if self.config.LOG_ORCHESTRATION_CALLS:
                logger.info(f"User progress stored for: {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store user progress: {e}")
            return False
    
    def get_user_progress(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user progress data from MongoDB"""
        try:
            progress_doc = user_collection.find_one(
                {"user_id": user_id, "type": "progress_tracking"},
                {"_id": 0}
            )
            
            return progress_doc.get("progress_data") if progress_doc else None
            
        except Exception as e:
            logger.error(f"Failed to get user progress: {e}")
            return None
    
    def store_trigger_event(self, user_id: str, trigger_data: Dict[str, Any]) -> bool:
        """Store trigger event in MongoDB for analysis"""
        try:
            if not self.config.LOG_TRIGGER_EVENTS:
                return True  # Skip if logging disabled
            
            # Prepare trigger event document
            trigger_doc = {
                "user_id": user_id,
                "trigger_data": trigger_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "trigger_event",
                "source": "orchestration_integration"
            }
            
            # Store in user collection
            user_collection.insert_one(trigger_doc)
            
            logger.info(f"Trigger event logged for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store trigger event: {e}")
            return False
    
    def get_user_trigger_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's trigger history from MongoDB"""
        try:
            triggers = list(user_collection.find(
                {"user_id": user_id, "type": "trigger_event"},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit))
            
            return triggers
            
        except Exception as e:
            logger.error(f"Failed to get user trigger history: {e}")
            return []
    
    def sync_orchestration_user_data(self, user_id: str, orchestration_session: Dict[str, Any]) -> bool:
        """Sync user data between orchestration system and MongoDB"""
        try:
            # Extract relevant data from orchestration session
            sync_data = {
                "user_id": user_id,
                "educational_progress": orchestration_session.get("educational_progress", {}),
                "wellness_metrics": orchestration_session.get("wellness_metrics", {}),
                "spiritual_journey": orchestration_session.get("spiritual_journey", {}),
                "interaction_count": orchestration_session.get("interaction_count", 0),
                "last_active": orchestration_session.get("last_active"),
                "synced_at": datetime.now(timezone.utc).isoformat(),
                "type": "orchestration_sync"
            }
            
            # Upsert sync data
            user_collection.update_one(
                {"user_id": user_id, "type": "orchestration_sync"},
                {"$set": sync_data},
                upsert=True
            )
            
            logger.info(f"Orchestration user data synced for: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync orchestration user data: {e}")
            return False
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user analytics from MongoDB"""
        try:
            analytics = {
                "user_id": user_id,
                "lesson_count": 0,
                "trigger_count": 0,
                "progress_data": None,
                "recent_activity": [],
                "performance_metrics": {}
            }
            
            # Count lessons
            analytics["lesson_count"] = lectures_collection.count_documents({"user_id": user_id})
            
            # Count triggers
            analytics["trigger_count"] = user_collection.count_documents({
                "user_id": user_id, 
                "type": "trigger_event"
            })
            
            # Get progress data
            analytics["progress_data"] = self.get_user_progress(user_id)
            
            # Get recent activity (last 10 items)
            recent_lessons = list(lectures_collection.find(
                {"user_id": user_id},
                {"subject": 1, "topic": 1, "generated_at": 1, "_id": 0}
            ).sort("generated_at", -1).limit(5))
            
            recent_triggers = list(user_collection.find(
                {"user_id": user_id, "type": "trigger_event"},
                {"trigger_data.type": 1, "timestamp": 1, "_id": 0}
            ).sort("timestamp", -1).limit(5))
            
            analytics["recent_activity"] = {
                "lessons": recent_lessons,
                "triggers": recent_triggers
            }
            
            # Calculate performance metrics
            if analytics["progress_data"]:
                progress = analytics["progress_data"]
                educational_progress = progress.get("educational_progress", {})
                quiz_scores = educational_progress.get("quiz_scores", [])
                
                if quiz_scores:
                    analytics["performance_metrics"] = {
                        "average_quiz_score": sum(quiz_scores) / len(quiz_scores),
                        "latest_quiz_score": quiz_scores[-1],
                        "quiz_trend": "improving" if len(quiz_scores) > 1 and quiz_scores[-1] > quiz_scores[-2] else "stable",
                        "total_quizzes": len(quiz_scores)
                    }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get user analytics: {e}")
            return {"user_id": user_id, "error": str(e)}
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old orchestration data from MongoDB"""
        try:
            cutoff_date = datetime.now(timezone.utc).replace(
                day=datetime.now().day - days_old
            ).isoformat()
            
            # Clean up old trigger events
            trigger_result = user_collection.delete_many({
                "type": "trigger_event",
                "timestamp": {"$lt": cutoff_date}
            })
            
            # Clean up old sync data (keep latest for each user)
            # This is more complex - we'll keep it simple for now
            
            cleanup_stats = {
                "trigger_events_deleted": trigger_result.deleted_count,
                "cleanup_date": cutoff_date
            }
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return {"error": str(e)}

# Global instance
db_integration = OrchestrationDBIntegration()

def get_db_integration() -> OrchestrationDBIntegration:
    """Get the global database integration instance"""
    return db_integration

# Convenience functions for easy import
def store_enhanced_lesson(lesson_data: Dict[str, Any]) -> str:
    """Store enhanced lesson - convenience function"""
    return db_integration.store_enhanced_lesson(lesson_data)

def get_user_analytics(user_id: str) -> Dict[str, Any]:
    """Get user analytics - convenience function"""
    return db_integration.get_user_analytics(user_id)

def sync_user_data(user_id: str, orchestration_session: Dict[str, Any]) -> bool:
    """Sync user data - convenience function"""
    return db_integration.sync_orchestration_user_data(user_id, orchestration_session)

if __name__ == "__main__":
    # Test the database integration
    print("🗄️ Testing Database Integration")
    print("=" * 40)
    
    test_user_id = "test_db_integration_001"
    
    # Test storing progress
    test_progress = {
        "educational_progress": {
            "quiz_scores": [75, 80, 85],
            "learning_topics": ["math", "science"],
            "last_activity": datetime.now().isoformat()
        }
    }
    
    success = db_integration.store_user_progress(test_user_id, test_progress)
    print(f"✅ Store progress: {success}")
    
    # Test getting analytics
    analytics = db_integration.get_user_analytics(test_user_id)
    print(f"✅ Get analytics: {analytics.get('user_id') == test_user_id}")
    
    print("Database integration test completed!")
