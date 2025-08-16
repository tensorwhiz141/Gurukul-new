#!/usr/bin/env python3
"""
Gurukul Backend Services Health Check
Checks if all backend services are running and accessible
"""

import requests
import time
import sys
from typing import Dict, List, Tuple

# Service configurations
SERVICES = {
    "Base Backend": {
        "url": "http://localhost:8000/docs",
        "port": 8000,
        "description": "Main API with orchestration"
    },
    "API Data Service": {
        "url": "http://localhost:8001/health",
        "port": 8001,
        "description": "Data processing service"
    },
    "Financial Simulator": {
        "url": "http://localhost:8002/docs",
        "port": 8002,
        "description": "Financial forecasting"
    },
    "Memory Management": {
        "url": "http://localhost:8003/memory/health",
        "port": 8003,
        "description": "User memory service"
    },
    "Akash Service": {
        "url": "http://localhost:8004/health",
        "port": 8004,
        "description": "Additional AI services"
    },
    "Subject Generation": {
        "url": "http://localhost:8005/",
        "port": 8005,
        "description": "Subject content generation"
    },
    "Wellness API": {
        "url": "http://localhost:8006/",
        "port": 8006,
        "description": "Wellness + Forecasting"
    },
    "TTS Service": {
        "url": "http://localhost:8007/",
        "port": 8007,
        "description": "Text-to-speech service"
    }
}

def check_service(name: str, config: Dict) -> Tuple[bool, str]:
    """Check if a service is running and accessible"""
    try:
        response = requests.get(config["url"], timeout=5)
        if response.status_code == 200:
            return True, f"✅ {name} (Port {config['port']}) - Running"
        else:
            return False, f"❌ {name} (Port {config['port']}) - HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"❌ {name} (Port {config['port']}) - Connection refused"
    except requests.exceptions.Timeout:
        return False, f"⏰ {name} (Port {config['port']}) - Timeout"
    except Exception as e:
        return False, f"❌ {name} (Port {config['port']}) - Error: {str(e)}"

def main():
    """Main health check function"""
    print("🔍 Gurukul Backend Services Health Check")
    print("=" * 50)
    
    results = []
    running_count = 0
    
    for service_name, config in SERVICES.items():
        is_running, message = check_service(service_name, config)
        results.append((is_running, message, config["description"]))
        if is_running:
            running_count += 1
        
        print(message)
        print(f"   📝 {config['description']}")
        print()
    
    print("=" * 50)
    print(f"📊 Summary: {running_count}/{len(SERVICES)} services running")
    
    if running_count == len(SERVICES):
        print("🎉 All services are running successfully!")
        print("\n🚀 Next steps:")
        print("   1. Start the frontend: cd 'new frontend' && start_frontend.bat")
        print("   2. Open http://localhost:3000 or http://localhost:5174")
        return 0
    else:
        print("⚠️  Some services are not running. Check the startup script output.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure all dependencies are installed")
        print("   2. Check if ports are available")
        print("   3. Review service logs for errors")
        print("   4. Try restarting the services")
        return 1

if __name__ == "__main__":
    sys.exit(main())
