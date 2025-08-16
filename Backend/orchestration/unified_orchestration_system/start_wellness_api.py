#!/usr/bin/env python3
"""
Wellness API Startup Script
Starts the unified orchestration system with wellness capabilities
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import google.generativeai
        print("✓ Required packages found")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has required keys"""
    env_path = Path(".env")
    if not env_path.exists():
        print("✗ .env file not found")
        print("Creating sample .env file...")
        with open(".env", "w") as f:
            f.write("# Gemini API Keys\n")
            f.write("GEMINI_API_KEY=your_primary_key_here\n")
            f.write("GEMINI_API_KEY_BACKUP=your_backup_key_here\n")
            f.write("\n# Sub-agent URLs (optional)\n")
            f.write("EMOTIONAL_WELLNESS_BOT_URL=http://localhost:8002\n")
            f.write("FINANCIAL_WELLNESS_BOT_URL=http://localhost:8003\n")
            f.write("TUTORBOT_URL=http://localhost:8001\n")
            f.write("QUIZBOT_URL=http://localhost:8004\n")
        print("✓ Created .env file. Please add your Gemini API keys.")
        return False
    
    # Check if API keys are set
    from dotenv import load_dotenv
    load_dotenv()
    
    primary_key = os.getenv("GEMINI_API_KEY")
    backup_key = os.getenv("GEMINI_API_KEY_BACKUP")
    
    if not primary_key or primary_key == "your_primary_key_here":
        print("✗ GEMINI_API_KEY not set in .env file")
        return False
    
    print("✓ Environment configuration found")
    return True

def start_simple_api():
    """Start the simple API server"""
    print("\n" + "="*60)
    print("  STARTING WELLNESS API SERVER")
    print("="*60)
    print("Starting simple orchestration API...")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nEndpoints:")
    print("  POST /ask-wellness - Wellness advice with emotional support")
    print("  POST /wellness - Simple wellness advice")
    print("  POST /ask-vedas - Spiritual wisdom")
    print("  POST /edumentor - Educational content")
    print("="*60)
    
    try:
        # Try to start the full orchestration API first
        if Path("orchestration_api.py").exists():
            print("Starting full orchestration system...")
            subprocess.run([sys.executable, "orchestration_api.py"], check=True)
        else:
            print("Starting simple API...")
            subprocess.run([sys.executable, "simple_api.py"], check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error starting server: {e}")
        print("Trying simple API as fallback...")
        try:
            subprocess.run([sys.executable, "simple_api.py"], check=True)
        except Exception as fallback_error:
            print(f"✗ Fallback also failed: {fallback_error}")

def main():
    """Main startup function"""
    print("Wellness API Startup Script")
    print("=" * 30)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check environment
    if not check_env_file():
        print("\nPlease configure your .env file with valid API keys and run again.")
        return
    
    # Start the API
    start_simple_api()

if __name__ == "__main__":
    main()
