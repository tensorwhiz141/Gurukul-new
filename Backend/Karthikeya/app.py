"""
Karthikeya Multilingual Reporting Engine - Orchestrator
Minimal orchestrator that delegates to modular nudge_engine components
"""

import os
import sys
import logging
from pathlib import Path

# Add nudge_engine to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nudge_engine.api_endpoints import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point - orchestrator only"""
    logger.info("Starting Karthikeya Multilingual Reporting Engine v2.0")
    
    # Configuration paths
    config_dir = os.getenv('KARTHIKEYA_CONFIG_DIR', 'config')
    templates_dir = os.getenv('KARTHIKEYA_TEMPLATES_DIR', 'templates')
    
    # Create Flask app using modular components
    app = create_app(config_dir=config_dir, templates_dir=templates_dir)
    
    # Get configuration from environment
    host = os.getenv('KARTHIKEYA_HOST', '0.0.0.0')
    port = int(os.getenv('KARTHIKEYA_PORT', 5000))
    debug = os.getenv('KARTHIKEYA_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Config directory: {config_dir}")
    logger.info(f"Templates directory: {templates_dir}")
    logger.info(f"Debug mode: {debug}")
    
    # Run the application
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
