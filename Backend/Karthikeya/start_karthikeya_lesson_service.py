#!/usr/bin/env python3
"""
Startup script for Karthikeya Lesson Service with new format support
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('karthikeya_lesson_service.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'flask',
        'pyyaml',
        'requests'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {missing_modules}")
        print("Please install them using: pip install " + " ".join(missing_modules))
        return False
    
    return True

def check_files():
    """Check if all required files are present"""
    required_files = [
        'app.py',
        'nudge_engine/api_endpoints.py',
        'nudge_engine/report_generator.py',
        'templates/report_templates.json',
        'config/language_config.yaml',
        'config/nudge_config.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (current_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    return True

def validate_lesson_templates():
    """Validate that lesson templates are properly configured"""
    try:
        import json
        template_path = current_dir / 'templates' / 'report_templates.json'
        
        with open(template_path, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        
        # Check if lesson_content templates exist
        if 'lesson_content' not in templates:
            print("❌ lesson_content templates not found in report_templates.json")
            return False
        
        # Check supported languages
        lesson_templates = templates['lesson_content']
        supported_languages = list(lesson_templates.keys())
        
        print(f"✅ Lesson templates found for languages: {supported_languages}")
        return True
        
    except Exception as e:
        print(f"❌ Error validating lesson templates: {e}")
        return False

def start_service():
    """Start the Karthikeya lesson service"""
    try:
        print("🚀 Starting Karthikeya Lesson Service...")
        print("=" * 50)
        
        # Import and start the app
        from app import app
        
        print("✅ Karthikeya Lesson Service initialized successfully")
        print("📚 New features available:")
        print("   - Multilingual lesson generation")
        print("   - Standardized lesson format")
        print("   - Interactive quiz generation")
        print("   - TTS-ready content")
        print("   - Sentiment analysis integration")
        print("")
        print("🌐 API Endpoints:")
        print("   POST /generate-lesson - Generate multilingual lessons")
        print("   GET  /health        - Service health check")
        print("   POST /generate      - Generate reports (existing)")
        print("")
        print("📖 Documentation: KARTHIKEYA_LESSON_INTEGRATION.md")
        print("🧪 Test Script: test_lesson_integration.py")
        print("")
        print("🔗 Service running on: http://localhost:5000")
        print("=" * 50)
        
        # Start the Flask app
        app.run(
            debug=True,
            host="0.0.0.0",
            port=5000,
            use_reloader=False  # Disable reloader to avoid double startup
        )
        
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🎓 Karthikeya Lesson Service Startup")
    print("=" * 40)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Pre-flight checks
    print("🔍 Running pre-flight checks...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    print("✅ Dependencies check passed")
    
    # Check required files
    if not check_files():
        sys.exit(1)
    print("✅ Required files check passed")
    
    # Validate lesson templates
    if not validate_lesson_templates():
        sys.exit(1)
    print("✅ Lesson templates validation passed")
    
    print("✅ All pre-flight checks passed!")
    print("")
    
    # Start the service
    try:
        start_service()
    except KeyboardInterrupt:
        print("\n🛑 Service stopped by user")
        logger.info("Karthikeya Lesson Service stopped by user")
    except Exception as e:
        print(f"\n❌ Service error: {e}")
        logger.error(f"Service error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
