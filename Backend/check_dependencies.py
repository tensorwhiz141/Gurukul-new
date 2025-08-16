#!/usr/bin/env python3
"""
Gurukul Platform Dependency Checker
Checks and installs missing dependencies for all services
"""

import subprocess
import sys
import importlib
import os

# Service-specific dependency mapping
SERVICE_DEPENDENCIES = {
    "Base Backend (Port 8000)": [
        "fastapi", "uvicorn", "easyocr", "opencv-python", "pillow", 
        "pytesseract", "requests", "python-dotenv"
    ],
    "Dedicated Chatbot (Port 8001)": [
        "fastapi", "uvicorn", "groq", "openai", "transformers", 
        "torch", "requests", "python-dotenv"
    ],
    "Financial Simulator (Port 8002)": [
        "fastapi", "uvicorn", "redis", "celery", "yfinance",
        "pandas", "numpy", "networkx", "langchain", "langchain_groq",
        "langgraph", "python-dotenv"
    ],
    "Memory Management (Port 8003)": [
        "fastapi", "uvicorn", "pymongo", "chromadb", "sentence-transformers",
        "requests", "python-dotenv"
    ],
    "Akash Service (Port 8004)": [
        "fastapi", "uvicorn", "transformers", "torch", "requests", 
        "python-dotenv"
    ],
    "Subject Generation (Port 8005)": [
        "fastapi", "uvicorn", "requests", "beautifulsoup4", "lxml",
        "python-dotenv"
    ],
    "Wellness + Forecasting (Port 8006)": [
        "fastapi", "uvicorn", "prophet", "statsmodels", "scikit-learn",
        "pandas", "numpy", "faiss", "chromadb", "sentence-transformers",
        "google-generativeai", "python-dotenv"
    ],
    "TTS Service (Port 8007)": [
        "fastapi", "uvicorn", "pyttsx3", "gTTS", "pygame", "soundfile",
        "librosa", "speechrecognition", "python-dotenv"
    ]
}

# Package name mapping (import name -> pip name)
PACKAGE_MAPPING = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv"
}

def check_package(package_name):
    """Check if a package is installed"""
    try:
        # Handle package name mapping
        import_name = package_name
        if package_name in PACKAGE_MAPPING.values():
            # Find the import name for this pip package
            for imp_name, pip_name in PACKAGE_MAPPING.items():
                if pip_name == package_name:
                    import_name = imp_name
                    break
        
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        print(f"  📦 Installing {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name, "--quiet"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ {package_name} installed successfully")
            return True
        else:
            print(f"  ❌ Failed to install {package_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Error installing {package_name}: {e}")
        return False

def check_service_dependencies(service_name, dependencies):
    """Check dependencies for a specific service"""
    print(f"\n🔍 Checking {service_name}...")
    
    missing_packages = []
    for package in dependencies:
        if not check_package(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"  ⚠️  Missing packages: {', '.join(missing_packages)}")
        
        # Ask user if they want to install
        install_all = input(f"  Install missing packages for {service_name}? (y/n): ").lower().strip()
        
        if install_all == 'y':
            success_count = 0
            for package in missing_packages:
                if install_package(package):
                    success_count += 1
            
            print(f"  📊 Installed {success_count}/{len(missing_packages)} packages")
            return success_count == len(missing_packages)
        else:
            print(f"  ⏭️  Skipping installation for {service_name}")
            return False
    else:
        print(f"  ✅ All dependencies satisfied")
        return True

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Gurukul requires Python 3.8+")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def main():
    """Main dependency checker"""
    print("🔍 GURUKUL PLATFORM DEPENDENCY CHECKER")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Please upgrade Python and try again")
        return 1
    
    # Check each service
    all_services_ready = True
    
    for service_name, dependencies in SERVICE_DEPENDENCIES.items():
        service_ready = check_service_dependencies(service_name, dependencies)
        if not service_ready:
            all_services_ready = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_services_ready:
        print("🎉 ALL SERVICES READY!")
        print("✅ All dependencies are satisfied")
        print("\n🚀 You can now run: start_gurukul_with_forecasting.bat")
    else:
        print("⚠️  SOME SERVICES HAVE MISSING DEPENDENCIES")
        print("❌ Please install missing packages before starting services")
        print("\n💡 Quick fix: Run Backend/install_all_dependencies.bat")
    
    print("\n📋 Service Status Summary:")
    for service_name, dependencies in SERVICE_DEPENDENCIES.items():
        missing = [pkg for pkg in dependencies if not check_package(pkg)]
        if missing:
            print(f"  ❌ {service_name}: Missing {len(missing)} packages")
        else:
            print(f"  ✅ {service_name}: Ready")
    
    return 0 if all_services_ready else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        input("\nPress Enter to exit...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Dependency check cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
