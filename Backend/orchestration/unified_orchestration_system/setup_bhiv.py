#!/usr/bin/env python3
"""
BHIV Core Setup Script
Helps set up and run the BHIV Core system
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def print_header():
    print("🧠 BHIV Core v0.9 Setup")
    print("=" * 50)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install core dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "fastapi", "uvicorn[standard]", "pydantic>=2.0", 
            "aiohttp", "aiofiles", "pytest", "pytest-asyncio"
        ])
        print("✅ Core dependencies installed")
        
        # Try to install optional dependencies
        optional_deps = [
            "qdrant-client",
            "redis", 
            "pyjwt",
            "sentence-transformers",
            "langchain",
            "langchain-community"
        ]
        
        for dep in optional_deps:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                    capture_output=True)
                print(f"✅ {dep} installed")
            except subprocess.CalledProcessError:
                print(f"⚠️ {dep} installation failed (optional)")
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependency installation failed: {e}")
        return False
    
    return True

def create_env_file():
    """Create environment file with default values"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    print("\n🔧 Creating .env file...")
    
    env_content = """# BHIV Core Configuration
# Core Settings
JWT_SECRET_KEY=bhiv-development-secret-change-in-production
REDIS_URL=redis://localhost:6379
QDRANT_URL=localhost:6333

# Database
DATABASE_URL=sqlite:///bhiv_sessions.db

# Audio Configuration
AUDIO_BASE_URL=http://localhost:8010/audio
TTS_SERVICE_URL=http://localhost:8005

# External APIs (Optional - add your keys)
# OPENAI_API_KEY=your-openai-key-here
# ELEVENLABS_API_KEY=your-elevenlabs-key-here
# GEMINI_API_KEY=your-gemini-key-here

# Development Settings
DEBUG=true
LOG_LEVEL=INFO
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created with default values")

def check_file_structure():
    """Check if all required files exist"""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "bhiv_core_schema.py",
        "bhiv_core_orchestrator.py", 
        "bhiv_core_api.py",
        "bhiv_alpha_router.py",
        "bhiv_knowledgebase.py",
        "bhiv_voice_video_pipeline.py",
        "bhiv_session_manager.py",
        "bhiv_api_gateway.py",
        "test_bhiv_core.py",
        "integration_test_bhiv.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def run_basic_test():
    """Run a basic import test"""
    print("\n🧪 Running basic import test...")
    
    try:
        # Test basic imports
        import bhiv_core_schema
        print("✅ bhiv_core_schema imported successfully")
        
        # Test schema creation
        from bhiv_core_schema import create_bhiv_request, UserPersona, InputMode
        
        request = create_bhiv_request(
            user_id="test_user",
            raw_text="Hello BHIV!",
            voice_enabled=False,
            mode=InputMode.CHAT,
            persona=UserPersona.STUDENT
        )
        
        print("✅ BHIV request created successfully")
        print(f"   User: {request.user_info.user_id}")
        print(f"   Query: {request.input.raw_text}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

async def run_integration_test():
    """Run a simple integration test"""
    print("\n🔄 Running integration test...")
    
    try:
        # Import and test orchestrator
        from bhiv_core_orchestrator import BHIVCoreOrchestrator
        from bhiv_core_schema import create_bhiv_request, InputMode, UserPersona
        
        # Create orchestrator (without full initialization for quick test)
        orchestrator = BHIVCoreOrchestrator()
        
        # Test health check
        health = await orchestrator.health_check()
        print(f"✅ Health check completed: {health['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print("   This is expected if external services (Redis, Qdrant) are not running")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n🚀 Next Steps:")
    print("=" * 50)
    print("1. Start the BHIV Core API:")
    print("   python bhiv_core_api.py")
    print()
    print("2. Run unit tests:")
    print("   python -m pytest test_bhiv_core.py -v")
    print()
    print("3. Run integration tests:")
    print("   python integration_test_bhiv.py")
    print()
    print("4. Access the API documentation:")
    print("   http://localhost:8010/docs")
    print()
    print("5. Optional: Start external services for full functionality:")
    print("   - Redis: docker run -p 6379:6379 redis")
    print("   - Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    print()
    print("📚 See BHIV_CORE_README.md for detailed documentation")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    check_python_version()
    
    # Check file structure
    if not check_file_structure():
        print("\n❌ Setup failed: Missing required files")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Run basic test
    if not run_basic_test():
        print("\n⚠️ Basic test failed, but setup can continue")
    
    # Run integration test
    try:
        asyncio.run(run_integration_test())
    except Exception as e:
        print(f"\n⚠️ Integration test failed: {e}")
        print("   This is normal if external services are not running")
    
    print("\n✅ BHIV Core setup completed!")
    print_next_steps()

if __name__ == "__main__":
    main()
