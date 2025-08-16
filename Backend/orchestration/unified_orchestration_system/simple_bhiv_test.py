#!/usr/bin/env python3
"""
Simple BHIV Test - Minimal dependencies
Just tests if the core BHIV schema works
"""

def main():
    print("🧠 Simple BHIV Core Test")
    print("=" * 40)
    
    try:
        # Test 1: Import the schema
        print("📦 Testing imports...")
        from bhiv_core_schema import create_bhiv_request
        print("✅ BHIV schema imported")
        
        # Test 2: Create a simple request
        print("\n🔧 Testing request creation...")
        request = create_bhiv_request(
            user_id="test_user",
            raw_text="Hello BHIV!"
        )
        print("✅ BHIV request created")
        print(f"   User: {request.user_info.user_id}")
        print(f"   Query: {request.input.raw_text}")
        
        # Test 3: Test serialization
        print("\n💾 Testing serialization...")
        json_data = request.model_dump_json()
        print("✅ Request serialized to JSON")
        print(f"   JSON size: {len(json_data)} characters")
        
        print("\n🎉 BHIV Core basic functionality works!")
        print("\n🚀 Ready to start BHIV API:")
        print("   python bhiv_core_api.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Try installing: pip install pydantic")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the right directory")
        print("2. Install pydantic: pip install pydantic")
        print("3. Check if bhiv_core_schema.py exists")
