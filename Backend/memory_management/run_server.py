#!/usr/bin/env python3
"""
Memory Management API Server Startup Script.

This script starts the Memory Management API server with proper configuration
and error handling.
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from memory_management.api import app
from memory_management.database import get_memory_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('memory_api.log')
    ]
)

logger = logging.getLogger(__name__)


def check_environment():
    """Check if all required environment variables are set."""
    required_vars = [
        "MONGODB_URL",
        "MEMORY_API_KEYS"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Using default values. For production, please set these variables.")
    
    return len(missing_vars) == 0


def test_database_connection():
    """Test database connection before starting the server."""
    try:
        db = get_memory_database()
        # Test connection by attempting to ping
        db._client.admin.command('ping')
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error("Please check your MongoDB connection settings")
        return False


def main():
    """Main function to start the server."""
    logger.info("🚀 Starting Memory Management API Server")
    
    # Check environment
    env_ok = check_environment()
    if not env_ok:
        logger.warning("⚠️  Some environment variables are missing")
    
    # Test database connection
    if not test_database_connection():
        logger.error("❌ Cannot start server without database connection")
        sys.exit(1)
    
    # Server configuration
    host = os.getenv("MEMORY_API_HOST", "0.0.0.0")
    port = int(os.getenv("MEMORY_API_PORT", "8003"))
    reload = os.getenv("MEMORY_API_RELOAD", "false").lower() == "true"
    workers = int(os.getenv("MEMORY_API_WORKERS", "1"))
    
    logger.info(f"🌐 Server configuration:")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info(f"   Reload: {reload}")
    logger.info(f"   Workers: {workers}")
    
    # Start server
    try:
        if workers > 1 and not reload:
            # Production mode with multiple workers
            uvicorn.run(
                "memory_management.api:app",
                host=host,
                port=port,
                workers=workers,
                log_level="info",
                access_log=True
            )
        else:
            # Development mode or single worker
            uvicorn.run(
                app,
                host=host,
                port=port,
                reload=reload,
                log_level="info",
                access_log=True
            )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
