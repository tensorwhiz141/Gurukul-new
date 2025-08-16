"""
BHIV Session Management and Logging
Comprehensive session management, interaction logging, reward scoring, and error handling

Features:
- Session lifecycle management
- Interaction logging and analytics
- Reward scoring and feedback loops
- Error tracking and recovery
- Performance monitoring
- User behavior analysis
- Data persistence and retrieval
"""

import os
import json
import logging
import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import uuid
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading

# Local imports
from bhiv_core_schema import (
    BHIVCoreRequest, BHIVCoreResponse, UserInfo, SystemLogs, LogEntry,
    ClassificationType, UrgencyLevel, OutputFormat
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SessionMetrics:
    """Session performance metrics"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    avg_response_time: float = 0.0
    total_processing_time: float = 0.0
    classifications_used: Dict[str, int] = None
    agents_used: Dict[str, int] = None
    reward_scores: List[float] = None
    user_satisfaction: Optional[float] = None
    
    def __post_init__(self):
        if self.classifications_used is None:
            self.classifications_used = {}
        if self.agents_used is None:
            self.agents_used = {}
        if self.reward_scores is None:
            self.reward_scores = []


@dataclass
class InteractionRecord:
    """Individual interaction record"""
    interaction_id: str
    session_id: str
    user_id: str
    timestamp: datetime
    request_data: Dict[str, Any]
    response_data: Dict[str, Any]
    classification: str
    agent_used: str
    processing_time_ms: float
    reward_score: float
    error_flag: bool
    error_details: Optional[str] = None
    user_feedback: Optional[Dict[str, Any]] = None


class BHIVDatabaseManager:
    """Database manager for BHIV session and interaction data"""
    
    def __init__(self, db_path: str = "bhiv_sessions.db"):
        self.db_path = db_path
        self.connection_pool = {}
        self.lock = threading.Lock()
        
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection for current thread"""
        thread_id = threading.get_ident()
        
        with self.lock:
            if thread_id not in self.connection_pool:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self.connection_pool[thread_id] = conn
            
            return self.connection_pool[thread_id]
    
    def initialize_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        
        # Sessions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                total_interactions INTEGER DEFAULT 0,
                successful_interactions INTEGER DEFAULT 0,
                failed_interactions INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0.0,
                total_processing_time REAL DEFAULT 0.0,
                classifications_used TEXT,
                agents_used TEXT,
                reward_scores TEXT,
                user_satisfaction REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Interactions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                request_data TEXT NOT NULL,
                response_data TEXT NOT NULL,
                classification TEXT NOT NULL,
                agent_used TEXT NOT NULL,
                processing_time_ms REAL NOT NULL,
                reward_score REAL NOT NULL,
                error_flag BOOLEAN NOT NULL,
                error_details TEXT,
                user_feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        
        # User analytics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_analytics (
                user_id TEXT PRIMARY KEY,
                total_sessions INTEGER DEFAULT 0,
                total_interactions INTEGER DEFAULT 0,
                avg_session_duration REAL DEFAULT 0.0,
                preferred_classifications TEXT,
                preferred_agents TEXT,
                avg_reward_score REAL DEFAULT 0.0,
                last_active TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                metric_id TEXT PRIMARY KEY,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_data TEXT,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions (start_time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_session_id ON interactions (session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions (user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions (timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_analytics_last_active ON user_analytics (last_active)")
        
        conn.commit()
        logger.info("Database initialized successfully")
    
    def save_session(self, session_metrics: SessionMetrics):
        """Save session metrics to database"""
        conn = self.get_connection()
        
        conn.execute("""
            INSERT OR REPLACE INTO sessions (
                session_id, user_id, start_time, end_time, total_interactions,
                successful_interactions, failed_interactions, avg_response_time,
                total_processing_time, classifications_used, agents_used,
                reward_scores, user_satisfaction, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            session_metrics.session_id,
            session_metrics.user_id,
            session_metrics.start_time,
            session_metrics.end_time,
            session_metrics.total_interactions,
            session_metrics.successful_interactions,
            session_metrics.failed_interactions,
            session_metrics.avg_response_time,
            session_metrics.total_processing_time,
            json.dumps(session_metrics.classifications_used),
            json.dumps(session_metrics.agents_used),
            json.dumps(session_metrics.reward_scores),
            session_metrics.user_satisfaction
        ))
        
        conn.commit()
    
    def save_interaction(self, interaction: InteractionRecord):
        """Save interaction record to database"""
        conn = self.get_connection()
        
        conn.execute("""
            INSERT INTO interactions (
                interaction_id, session_id, user_id, timestamp, request_data,
                response_data, classification, agent_used, processing_time_ms,
                reward_score, error_flag, error_details, user_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction.interaction_id,
            interaction.session_id,
            interaction.user_id,
            interaction.timestamp,
            json.dumps(interaction.request_data),
            json.dumps(interaction.response_data),
            interaction.classification,
            interaction.agent_used,
            interaction.processing_time_ms,
            interaction.reward_score,
            interaction.error_flag,
            interaction.error_details,
            json.dumps(interaction.user_feedback) if interaction.user_feedback else None
        ))
        
        conn.commit()
    
    def get_session(self, session_id: str) -> Optional[SessionMetrics]:
        """Get session metrics by ID"""
        conn = self.get_connection()
        
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        
        if row:
            return SessionMetrics(
                session_id=row['session_id'],
                user_id=row['user_id'],
                start_time=datetime.fromisoformat(row['start_time']),
                end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                total_interactions=row['total_interactions'],
                successful_interactions=row['successful_interactions'],
                failed_interactions=row['failed_interactions'],
                avg_response_time=row['avg_response_time'],
                total_processing_time=row['total_processing_time'],
                classifications_used=json.loads(row['classifications_used']) if row['classifications_used'] else {},
                agents_used=json.loads(row['agents_used']) if row['agents_used'] else {},
                reward_scores=json.loads(row['reward_scores']) if row['reward_scores'] else [],
                user_satisfaction=row['user_satisfaction']
            )
        
        return None
    
    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[SessionMetrics]:
        """Get recent sessions for a user"""
        conn = self.get_connection()
        
        rows = conn.execute("""
            SELECT * FROM sessions 
            WHERE user_id = ? 
            ORDER BY start_time DESC 
            LIMIT ?
        """, (user_id, limit)).fetchall()
        
        sessions = []
        for row in rows:
            sessions.append(SessionMetrics(
                session_id=row['session_id'],
                user_id=row['user_id'],
                start_time=datetime.fromisoformat(row['start_time']),
                end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                total_interactions=row['total_interactions'],
                successful_interactions=row['successful_interactions'],
                failed_interactions=row['failed_interactions'],
                avg_response_time=row['avg_response_time'],
                total_processing_time=row['total_processing_time'],
                classifications_used=json.loads(row['classifications_used']) if row['classifications_used'] else {},
                agents_used=json.loads(row['agents_used']) if row['agents_used'] else {},
                reward_scores=json.loads(row['reward_scores']) if row['reward_scores'] else [],
                user_satisfaction=row['user_satisfaction']
            ))
        
        return sessions
    
    def get_interactions(self, session_id: str) -> List[InteractionRecord]:
        """Get all interactions for a session"""
        conn = self.get_connection()
        
        rows = conn.execute("""
            SELECT * FROM interactions 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (session_id,)).fetchall()
        
        interactions = []
        for row in rows:
            interactions.append(InteractionRecord(
                interaction_id=row['interaction_id'],
                session_id=row['session_id'],
                user_id=row['user_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                request_data=json.loads(row['request_data']),
                response_data=json.loads(row['response_data']),
                classification=row['classification'],
                agent_used=row['agent_used'],
                processing_time_ms=row['processing_time_ms'],
                reward_score=row['reward_score'],
                error_flag=bool(row['error_flag']),
                error_details=row['error_details'],
                user_feedback=json.loads(row['user_feedback']) if row['user_feedback'] else None
            ))
        
        return interactions


class BHIVSessionManager:
    """
    Main BHIV Session Manager
    Handles session lifecycle, logging, and analytics
    """

    def __init__(self, db_path: str = "bhiv_sessions.db"):
        self.db_manager = BHIVDatabaseManager(db_path)
        self.active_sessions = {}
        self.session_metrics = {}
        self.reward_calculator = RewardScoreCalculator()
        self.analytics_engine = AnalyticsEngine(self.db_manager)

    async def initialize(self):
        """Initialize session manager"""
        logger.info("Initializing BHIV Session Manager...")

        # Initialize database
        self.db_manager.initialize_database()

        # Load active sessions from database
        await self._load_active_sessions()

        logger.info("BHIV Session Manager initialized successfully")

    async def start_session(self, user_info: UserInfo) -> str:
        """Start a new session"""
        session_id = str(uuid.uuid4())

        # Create session metrics
        session_metrics = SessionMetrics(
            session_id=session_id,
            user_id=user_info.user_id,
            start_time=datetime.now()
        )

        # Store in memory
        self.active_sessions[session_id] = {
            'user_info': user_info,
            'start_time': datetime.now(),
            'last_activity': datetime.now(),
            'status': 'active'
        }

        self.session_metrics[session_id] = session_metrics

        # Save to database
        self.db_manager.save_session(session_metrics)

        logger.info(f"Started new session: {session_id} for user: {user_info.user_id}")
        return session_id

    async def end_session(self, session_id: str):
        """End a session"""
        if session_id in self.active_sessions:
            # Update session metrics
            if session_id in self.session_metrics:
                metrics = self.session_metrics[session_id]
                metrics.end_time = datetime.now()

                # Calculate final metrics
                if metrics.total_interactions > 0:
                    metrics.avg_response_time = metrics.total_processing_time / metrics.total_interactions

                # Save final metrics
                self.db_manager.save_session(metrics)

            # Remove from active sessions
            self.active_sessions[session_id]['status'] = 'ended'
            self.active_sessions[session_id]['end_time'] = datetime.now()

            logger.info(f"Ended session: {session_id}")

    async def log_interaction(
        self,
        session_id: str,
        request: BHIVCoreRequest,
        response: BHIVCoreResponse
    ):
        """Log an interaction"""
        try:
            # Create interaction record
            interaction = InteractionRecord(
                interaction_id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=request.user_info.user_id,
                timestamp=datetime.now(),
                request_data=self._serialize_request(request),
                response_data=self._serialize_response(response),
                classification=response.agent_directives.classification.value,
                agent_used=response.agent_directives.agent_name,
                processing_time_ms=response.logs.processing_time_ms or 0.0,
                reward_score=response.logs.reward_score,
                error_flag=response.logs.error_flag,
                error_details=self._extract_error_details(response.logs)
            )

            # Save interaction
            self.db_manager.save_interaction(interaction)

            # Update session metrics
            await self._update_session_metrics(session_id, interaction)

            # Update user analytics
            await self.analytics_engine.update_user_analytics(
                request.user_info.user_id, interaction
            )

            logger.info(f"Logged interaction for session: {session_id}")

        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    async def _update_session_metrics(self, session_id: str, interaction: InteractionRecord):
        """Update session metrics with new interaction"""
        if session_id not in self.session_metrics:
            return

        metrics = self.session_metrics[session_id]

        # Update counters
        metrics.total_interactions += 1
        if not interaction.error_flag:
            metrics.successful_interactions += 1
        else:
            metrics.failed_interactions += 1

        # Update processing time
        metrics.total_processing_time += interaction.processing_time_ms
        metrics.avg_response_time = metrics.total_processing_time / metrics.total_interactions

        # Update classification usage
        if interaction.classification not in metrics.classifications_used:
            metrics.classifications_used[interaction.classification] = 0
        metrics.classifications_used[interaction.classification] += 1

        # Update agent usage
        if interaction.agent_used not in metrics.agents_used:
            metrics.agents_used[interaction.agent_used] = 0
        metrics.agents_used[interaction.agent_used] += 1

        # Update reward scores
        metrics.reward_scores.append(interaction.reward_score)

        # Update last activity
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['last_activity'] = datetime.now()

        # Save updated metrics
        self.db_manager.save_session(metrics)

    def _serialize_request(self, request: BHIVCoreRequest) -> Dict[str, Any]:
        """Serialize request for storage"""
        return {
            'session_id': request.session_id,
            'timestamp': request.timestamp.isoformat() if request.timestamp else None,
            'user_id': request.user_info.user_id,
            'raw_text': request.input.raw_text,
            'voice_enabled': request.input.voice_enabled,
            'mode': request.input.mode.value if request.input.mode else None,
            'lang': request.user_info.lang
        }

    def _serialize_response(self, response: BHIVCoreResponse) -> Dict[str, Any]:
        """Serialize response for storage"""
        return {
            'session_id': response.session_id,
            'timestamp': response.timestamp,
            'response_text': response.output.response_text,
            'confidence_score': response.output.confidence_score,
            'agent_name': response.agent_directives.agent_name,
            'classification': response.agent_directives.classification.value,
            'processing_time_ms': response.logs.processing_time_ms,
            'reward_score': response.logs.reward_score,
            'error_flag': response.logs.error_flag
        }

    def _extract_error_details(self, logs: SystemLogs) -> Optional[str]:
        """Extract error details from logs"""
        if not logs.error_flag:
            return None

        error_logs = [
            log for log in logs.detailed_logs
            if log.level.lower() == 'error'
        ]

        if error_logs:
            return "; ".join([log.message for log in error_logs])

        return "Unknown error"

    async def _load_active_sessions(self):
        """Load active sessions from database"""
        # This would load sessions that were active when the system shut down
        # For now, we'll start fresh
        pass

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        # Get from memory first
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id].copy()

            # Add metrics if available
            if session_id in self.session_metrics:
                metrics = self.session_metrics[session_id]
                session_info['metrics'] = asdict(metrics)

            return session_info

        # Get from database
        metrics = self.db_manager.get_session(session_id)
        if metrics:
            return {
                'session_id': session_id,
                'status': 'completed',
                'metrics': asdict(metrics)
            }

        return None

    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user analytics"""
        return await self.analytics_engine.get_user_analytics(user_id)

    async def cleanup_old_sessions(self, days_old: int = 30):
        """Clean up old sessions"""
        cutoff_date = datetime.now() - timedelta(days=days_old)

        # Remove from active sessions
        to_remove = []
        for session_id, session_info in self.active_sessions.items():
            if session_info['start_time'] < cutoff_date:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self.active_sessions[session_id]
            if session_id in self.session_metrics:
                del self.session_metrics[session_id]

        logger.info(f"Cleaned up {len(to_remove)} old sessions")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'active_sessions': len(self.active_sessions),
            'total_sessions_in_memory': len(self.session_metrics),
            'timestamp': datetime.now().isoformat()
        }


class RewardScoreCalculator:
    """Calculate reward scores for interactions"""

    def calculate_reward_score(
        self,
        classification_confidence: float,
        response_confidence: float,
        processing_time_ms: float,
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate reward score based on multiple factors"""

        # Base score from confidence levels
        confidence_score = (classification_confidence + response_confidence) / 2

        # Processing time penalty (prefer faster responses)
        time_penalty = min(processing_time_ms / 10000, 0.3)  # Max 0.3 penalty

        # User feedback bonus/penalty
        feedback_adjustment = 0.0
        if user_feedback:
            rating = user_feedback.get('rating', 3)  # 1-5 scale
            feedback_adjustment = (rating - 3) * 0.1  # -0.2 to +0.2

        # Calculate final score
        final_score = confidence_score - time_penalty + feedback_adjustment

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, final_score))


class AnalyticsEngine:
    """Analytics engine for user behavior analysis"""

    def __init__(self, db_manager: BHIVDatabaseManager):
        self.db_manager = db_manager

    async def update_user_analytics(self, user_id: str, interaction: InteractionRecord):
        """Update user analytics with new interaction"""
        # This would update comprehensive user analytics
        # For now, we'll implement basic tracking
        pass

    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user analytics"""
        sessions = self.db_manager.get_user_sessions(user_id)

        if not sessions:
            return {
                'user_id': user_id,
                'total_sessions': 0,
                'total_interactions': 0,
                'avg_session_duration': 0.0,
                'preferred_classifications': {},
                'preferred_agents': {},
                'avg_reward_score': 0.0
            }

        # Calculate analytics
        total_sessions = len(sessions)
        total_interactions = sum(s.total_interactions for s in sessions)

        # Calculate average session duration
        completed_sessions = [s for s in sessions if s.end_time]
        if completed_sessions:
            durations = [
                (s.end_time - s.start_time).total_seconds()
                for s in completed_sessions
            ]
            avg_duration = sum(durations) / len(durations)
        else:
            avg_duration = 0.0

        # Aggregate classifications and agents
        all_classifications = {}
        all_agents = {}
        all_rewards = []

        for session in sessions:
            for classification, count in session.classifications_used.items():
                all_classifications[classification] = all_classifications.get(classification, 0) + count

            for agent, count in session.agents_used.items():
                all_agents[agent] = all_agents.get(agent, 0) + count

            all_rewards.extend(session.reward_scores)

        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

        return {
            'user_id': user_id,
            'total_sessions': total_sessions,
            'total_interactions': total_interactions,
            'avg_session_duration': avg_duration,
            'preferred_classifications': all_classifications,
            'preferred_agents': all_agents,
            'avg_reward_score': avg_reward,
            'last_active': sessions[0].start_time.isoformat() if sessions else None
        }
