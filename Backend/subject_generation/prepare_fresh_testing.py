#!/usr/bin/env python3
"""
Prepare the system for fresh testing by clearing all caches and ensuring clean state
"""

import os
import sys
import subprocess

def clear_caches():
    """Clear all caches"""
    print("🧹 Clearing all caches for fresh testing...")
    
    try:
        # Run the clear cache script
        result = subprocess.run([sys.executable, "clear_cache.py"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Caches cleared successfully")
            print(result.stdout)
        else:
            print("⚠️ Cache clearing had issues:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error clearing caches: {e}")

def verify_service_running():
    """Check if the subject generation service is running"""
    print("\n🔍 Checking if subject generation service is running...")
    
    import requests
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Subject generation service is running on port 8000")
            return True
        else:
            print(f"⚠️ Service responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ Subject generation service is not running on port 8000")
        print("💡 Please start it with: cd Backend/subject_generation && python app.py")
        return False

def run_tests():
    """Run the comprehensive tests"""
    print("\n🧪 Running comprehensive tests...")
    
    test_scripts = [
        "test_strict_separation.py",
        "test_wikipedia_hardcode_fix.py", 
        "test_knowledge_base_fix.py"
    ]
    
    for script in test_scripts:
        if os.path.exists(script):
            print(f"\n🔄 Running {script}...")
            try:
                result = subprocess.run([sys.executable, script], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                
                print(result.stdout)
                if result.stderr:
                    print("Errors/Warnings:")
                    print(result.stderr)
                    
            except Exception as e:
                print(f"❌ Error running {script}: {e}")
        else:
            print(f"⚠️ Test script {script} not found")

def main():
    """Main function"""
    print("🚀 Preparing Fresh Testing Environment")
    print("=" * 50)
    
    # Step 1: Clear caches
    clear_caches()
    
    # Step 2: Verify service is running
    service_running = verify_service_running()
    
    if service_running:
        # Step 3: Run tests
        run_tests()
        
        print("\n" + "=" * 50)
        print("✅ Fresh testing preparation completed!")
        print("\n💡 What was done:")
        print("   - Cleared all cached lesson data")
        print("   - Cleared Wikipedia cache")
        print("   - Verified service is running")
        print("   - Ran comprehensive tests")
        print("\n🎯 Expected behavior now:")
        print("   - Each request generates fresh content")
        print("   - Knowledge base and Wikipedia are strictly separated")
        print("   - No repetitive responses")
        print("   - Different parameters produce different content")
        
    else:
        print("\n❌ Cannot proceed with testing - service not running")
        print("Please start the subject generation service first:")
        print("   cd Backend/subject_generation")
        print("   python app.py")

if __name__ == "__main__":
    main()
