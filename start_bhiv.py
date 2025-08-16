#!/usr/bin/env python3
"""
BHIV Core Startup Script
Run this from the Gurukul root directory to start BHIV Core
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🧠 Starting BHIV Core...")
    
    # Navigate to the BHIV directory
    bhiv_dir = Path("Backend/orchestration/unified_orchestration_system")
    
    if not bhiv_dir.exists():
        print("❌ BHIV directory not found!")
        print("   Make sure you're running this from the Gurukul root directory")
        sys.exit(1)
    
    # Change to BHIV directory
    os.chdir(bhiv_dir)
    print(f"📁 Changed to directory: {bhiv_dir.absolute()}")
    
    # Check if setup is needed
    if not Path("bhiv_core_api.py").exists():
        print("❌ BHIV Core files not found!")
        sys.exit(1)
    
    # Run setup first
    print("\n🔧 Running BHIV setup...")
    try:
        subprocess.run([sys.executable, "setup_bhiv.py"], check=True)
    except subprocess.CalledProcessError:
        print("⚠️ Setup had issues, but continuing...")
    except FileNotFoundError:
        print("⚠️ Setup script not found, continuing without setup...")
    
    # Start the API
    print("\n🚀 Starting BHIV Core API...")
    try:
        subprocess.run([sys.executable, "bhiv_core_api.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 BHIV Core stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start BHIV Core: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
