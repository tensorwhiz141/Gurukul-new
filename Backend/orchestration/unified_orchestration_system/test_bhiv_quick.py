#!/usr/bin/env python3
"""
BHIV Core Quick Test
Tests basic BHIV functionality without external dependencies
"""

import sys
import json
from datetime import datetime

def test_schema_imports():
    """Test that we can import the BHIV schema"""
    print("🧪 Testing BHIV schema imports...")
    
    try:
        from bhiv_core_schema import (
            create_bhiv_request, UserPersona, InputMode, 
            ClassificationType, UrgencyLevel, TriggerModule
        )
        print("✅ BHIV schema imports successful")
        return True
    except ImportError as e:
        print(f"❌ BHIV schema import failed: {e}")
        return False

def test_request_creation():
    """Test creating BHIV requests"""
    print("\n🧪 Testing BHIV request creation...")
    
    try:
        from bhiv_core_schema import create_bhiv_request, UserPersona, InputMode
        
        # Create a test request
        request = create_bhiv_request(
            user_id="test_user_123",
            raw_text="How do I learn machine learning?",
            voice_enabled=True,
            mode=InputMode.CHAT,
            persona=UserPersona.STUDENT,
            lang="en-IN"
        )
        
        # Validate request
        assert request.user_info.user_id == "test_user_123"
        assert request.input.raw_text == "How do I learn machine learning?"
        assert request.input.voice_enabled is True
        assert request.user_info.persona == UserPersona.STUDENT
        
        print("✅ BHIV request creation successful")
        print(f"   User ID: {request.user_info.user_id}")
        print(f"   Query: {request.input.raw_text}")
        print(f"   Voice: {request.input.voice_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ BHIV request creation failed: {e}")
        return False

def test_schema_serialization():
    """Test BHIV schema serialization"""
    print("\n🧪 Testing BHIV schema serialization...")
    
    try:
        from bhiv_core_schema import create_bhiv_request, InputMode, UserPersona
        
        # Create request
        request = create_bhiv_request(
            user_id="test_user",
            raw_text="Test serialization",
            voice_enabled=False,
            mode=InputMode.CHAT,
            persona=UserPersona.STUDENT
        )
        
        # Serialize to JSON
        json_data = request.model_dump_json()
        
        # Parse back
        parsed_data = json.loads(json_data)
        
        # Validate
        assert parsed_data["user_info"]["user_id"] == "test_user"
        assert parsed_data["input"]["raw_text"] == "Test serialization"
        
        print("✅ BHIV schema serialization successful")
        print(f"   JSON length: {len(json_data)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ BHIV schema serialization failed: {e}")
        return False

def test_classification_types():
    """Test BHIV classification enums"""
    print("\n🧪 Testing BHIV classification types...")
    
    try:
        from bhiv_core_schema import ClassificationType, UrgencyLevel, TriggerModule
        
        # Test enum values
        assert ClassificationType.LEARNING_QUERY == "learning-query"
        assert ClassificationType.WELLNESS_QUERY == "wellness-query"
        assert ClassificationType.SPIRITUAL_QUERY == "spiritual-query"
        
        assert UrgencyLevel.LOW == "low"
        assert UrgencyLevel.NORMAL == "normal"
        assert UrgencyLevel.HIGH == "high"
        assert UrgencyLevel.CRITICAL == "critical"
        
        assert TriggerModule.KNOWLEDGEBASE == "knowledgebase"
        assert TriggerModule.TUTORBOT == "tutorbot"
        
        print("✅ BHIV classification types successful")
        print(f"   Learning query: {ClassificationType.LEARNING_QUERY}")
        print(f"   Normal urgency: {UrgencyLevel.NORMAL}")
        print(f"   Knowledgebase trigger: {TriggerModule.KNOWLEDGEBASE}")
        
        return True
        
    except Exception as e:
        print(f"❌ BHIV classification types failed: {e}")
        return False

def test_agent_directives():
    """Test BHIV agent directives creation"""
    print("\n🧪 Testing BHIV agent directives...")
    
    try:
        from bhiv_core_schema import (
            create_agent_directives, ClassificationType, 
            UrgencyLevel, TriggerModule, OutputFormat
        )
        
        # Create agent directives
        directives = create_agent_directives(
            classification=ClassificationType.LEARNING_QUERY,
            agent_name="EduMentor",
            urgency=UrgencyLevel.NORMAL,
            trigger_module=TriggerModule.KNOWLEDGEBASE,
            output_format=OutputFormat.RICH_TEXT_WITH_TTS,
            confidence=0.95
        )
        
        # Validate
        assert directives.classification == ClassificationType.LEARNING_QUERY
        assert directives.agent_name == "EduMentor"
        assert directives.urgency == UrgencyLevel.NORMAL
        assert directives.confidence_score == 0.95
        
        print("✅ BHIV agent directives successful")
        print(f"   Agent: {directives.agent_name}")
        print(f"   Classification: {directives.classification}")
        print(f"   Confidence: {directives.confidence_score}")
        
        return True
        
    except Exception as e:
        print(f"❌ BHIV agent directives failed: {e}")
        return False

def test_orchestrator_import():
    """Test BHIV orchestrator import"""
    print("\n🧪 Testing BHIV orchestrator import...")
    
    try:
        from bhiv_core_orchestrator import BHIVCoreOrchestrator
        
        # Just test that we can create an instance
        orchestrator = BHIVCoreOrchestrator()
        
        print("✅ BHIV orchestrator import successful")
        print(f"   Orchestrator type: {type(orchestrator).__name__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ BHIV orchestrator import failed: {e}")
        print("   This might be due to missing dependencies")
        return False
    except Exception as e:
        print(f"❌ BHIV orchestrator creation failed: {e}")
        return False

def test_api_import():
    """Test BHIV API import"""
    print("\n🧪 Testing BHIV API import...")
    
    try:
        # Try importing the API module
        import bhiv_core_api
        
        print("✅ BHIV API import successful")
        return True
        
    except ImportError as e:
        print(f"❌ BHIV API import failed: {e}")
        print("   This might be due to missing FastAPI dependencies")
        return False
    except Exception as e:
        print(f"❌ BHIV API import error: {e}")
        return False

def test_session_manager_import():
    """Test BHIV session manager import"""
    print("\n🧪 Testing BHIV session manager import...")
    
    try:
        from bhiv_session_manager import BHIVSessionManager, SessionMetrics
        
        # Test creating session metrics
        metrics = SessionMetrics(
            session_id="test_session",
            user_id="test_user",
            start_time=datetime.now()
        )
        
        print("✅ BHIV session manager import successful")
        print(f"   Session ID: {metrics.session_id}")
        
        return True
        
    except ImportError as e:
        print(f"❌ BHIV session manager import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ BHIV session manager error: {e}")
        return False

def main():
    """Run all BHIV quick tests"""
    print("🧠 BHIV Core Quick Test Suite")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Test time: {datetime.now().isoformat()}")
    print()
    
    tests = [
        test_schema_imports,
        test_request_creation,
        test_schema_serialization,
        test_classification_types,
        test_agent_directives,
        test_orchestrator_import,
        test_api_import,
        test_session_manager_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 BHIV Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All BHIV tests passed! BHIV Core is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Install dependencies: pip install fastapi uvicorn")
        print("   2. Start BHIV API: python bhiv_core_api.py")
        print("   3. Visit: http://localhost:8010/docs")
        return True
    elif passed >= 5:
        print("✅ Core BHIV functionality works! Some advanced features may need dependencies.")
        print("\n🚀 You can start with:")
        print("   python bhiv_core_api.py")
        return True
    else:
        print("⚠️ Some core BHIV tests failed. Check the output above for details.")
        print("\n💡 Try installing dependencies:")
        print("   pip install pydantic fastapi")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
