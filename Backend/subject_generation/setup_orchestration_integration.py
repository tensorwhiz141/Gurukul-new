#!/usr/bin/env python3
"""
Setup script to ensure orchestration system data is properly integrated
This script will create vector stores from the orchestration system data if they don't exist
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_orchestration_data():
    """Check if orchestration system data files exist"""
    
    print("🔍 Checking Orchestration System Data")
    print("=" * 50)
    
    # Path to orchestration system data
    orchestration_path = os.path.join(os.path.dirname(__file__), '..', 'orchestration', 'unified_orchestration_system')
    data_path = os.path.join(orchestration_path, 'data')
    vector_stores_path = os.path.join(orchestration_path, 'vector_stores')
    
    print(f"📂 Data Directory: {data_path}")
    print(f"📂 Vector Stores Directory: {vector_stores_path}")
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        print("❌ Data directory does not exist")
        return False, orchestration_path
    
    # Check for expected data files
    expected_files = [
        "Four-Vedas-English-Translation.pdf",
        "108upanishads.pdf", 
        "Gita.pdf",
        "ramayan.pdf",
        "Plant_8-12.csv",
        "Seed_1-7.csv",
        "Tree.csv"
    ]
    
    found_files = []
    missing_files = []
    
    for file in expected_files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            found_files.append(file)
            print(f"✅ {file} - {file_size:,} bytes")
        else:
            missing_files.append(file)
            print(f"❌ {file} - NOT FOUND")
    
    print(f"\n📊 Summary: {len(found_files)}/{len(expected_files)} files found")
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        print("💡 Please ensure all data files are in the orchestration/unified_orchestration_system/data/ directory")
    
    return len(found_files) > 0, orchestration_path

def check_vector_stores(orchestration_path):
    """Check if vector stores exist"""
    
    print("\n🗃️ Checking Vector Stores")
    print("=" * 30)
    
    vector_stores_path = os.path.join(orchestration_path, 'vector_stores')
    
    expected_stores = [
        "vedas_index",
        "educational_index", 
        "wellness_index",
        "unified_index"
    ]
    
    existing_stores = []
    missing_stores = []
    
    for store in expected_stores:
        store_path = os.path.join(vector_stores_path, store)
        if os.path.exists(store_path):
            existing_stores.append(store)
            print(f"✅ {store}")
        else:
            missing_stores.append(store)
            print(f"❌ {store} - NOT FOUND")
    
    print(f"\n📊 Summary: {len(existing_stores)}/{len(expected_stores)} vector stores found")
    
    return len(existing_stores) > 0, missing_stores

def create_vector_stores(orchestration_path):
    """Create vector stores from the orchestration system data"""
    
    print("\n🔧 Creating Vector Stores from Orchestration Data")
    print("=" * 60)
    
    try:
        # Change to orchestration directory
        original_cwd = os.getcwd()
        os.chdir(orchestration_path)
        
        print(f"📂 Working directory: {orchestration_path}")
        print("🚀 Running data ingestion...")
        
        # Run the data ingestion script
        result = subprocess.run([sys.executable, "data_ingestion.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Data ingestion completed successfully!")
            print("\n📄 Output:")
            print(result.stdout)
            
            if result.stderr:
                print("\n⚠️  Warnings:")
                print(result.stderr)
                
            return True
        else:
            print("❌ Data ingestion failed!")
            print(f"Exit code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Data ingestion timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running data ingestion: {e}")
        return False
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def test_integration():
    """Test the integration by making a sample request"""
    
    print("\n🧪 Testing Integration")
    print("=" * 30)
    
    try:
        import requests
        
        # Test request
        params = {
            "subject": "ved",
            "topic": "sound",
            "include_wikipedia": False,
            "use_knowledge_store": True
        }
        
        print("📡 Making test request...")
        response = requests.get("http://localhost:8000/generate_lesson", params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            detailed_sources = data.get('detailed_sources', [])
            
            print(f"✅ Test successful!")
            print(f"📚 Knowledge Base Used: {data.get('knowledge_base_used', False)}")
            print(f"🔍 Detailed Sources: {len(detailed_sources)}")
            
            if detailed_sources:
                print("📋 Source Types Found:")
                source_types = set(source.get('source_type', 'unknown') for source in detailed_sources)
                for source_type in source_types:
                    count = sum(1 for s in detailed_sources if s.get('source_type') == source_type)
                    print(f"   - {source_type}: {count}")
                return True
            else:
                print("⚠️  No detailed sources found")
                return False
        else:
            print(f"❌ Test failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to subject generation service")
        print("💡 Please start the service: cd Backend/subject_generation && python app.py")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main setup function"""
    
    print("🚀 Orchestration System Integration Setup")
    print("=" * 60)
    print("This script will:")
    print("  1. Check for orchestration system data files")
    print("  2. Check for existing vector stores")
    print("  3. Create vector stores if needed")
    print("  4. Test the integration")
    print("=" * 60)
    
    # Step 1: Check data files
    has_data, orchestration_path = check_orchestration_data()
    
    if not has_data:
        print("\n❌ Setup failed: No data files found")
        print("💡 Please ensure data files are in Backend/orchestration/unified_orchestration_system/data/")
        return False
    
    # Step 2: Check vector stores
    has_stores, missing_stores = check_vector_stores(orchestration_path)
    
    # Step 3: Create vector stores if needed
    if not has_stores or missing_stores:
        print(f"\n🔧 Need to create vector stores: {missing_stores}")
        
        user_input = input("\nProceed with vector store creation? (y/n): ").lower().strip()
        if user_input != 'y':
            print("❌ Setup cancelled by user")
            return False
        
        success = create_vector_stores(orchestration_path)
        if not success:
            print("\n❌ Setup failed: Could not create vector stores")
            return False
    else:
        print("\n✅ All vector stores already exist")
    
    # Step 4: Test integration
    print("\n🧪 Testing integration...")
    test_success = test_integration()
    
    if test_success:
        print("\n🎉 Setup completed successfully!")
        print("✅ Orchestration system data is now integrated with knowledge store")
        print("\n💡 You can now use the knowledge store option to access:")
        print("   📖 Vedic texts (Four Vedas, Upanishads, Gita, Ramayana)")
        print("   🗃️ Educational databases (Plant, Seed, Tree data)")
        print("   🎯 Categorized content with detailed source attribution")
        return True
    else:
        print("\n⚠️  Setup completed but integration test failed")
        print("💡 Vector stores were created but the service may need to be restarted")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 Ready to test!")
        print("Run: python test_orchestration_integration.py")
    else:
        print("\n🔧 Setup incomplete - please check the issues above")
    
    sys.exit(0 if success else 1)
