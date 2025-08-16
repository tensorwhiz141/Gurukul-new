#!/usr/bin/env python3
"""
Script to clear cached lesson data to ensure fresh generation
"""

import os
import shutil
import glob

def clear_knowledge_store_cache():
    """Clear all cached lessons from knowledge store"""
    
    print("🧹 Clearing Knowledge Store Cache")
    print("=" * 40)
    
    # Knowledge store directory
    knowledge_store_dir = "knowledge_store"
    
    if os.path.exists(knowledge_store_dir):
        try:
            # Get list of cached files
            cached_files = glob.glob(os.path.join(knowledge_store_dir, "*.json"))
            
            if cached_files:
                print(f"Found {len(cached_files)} cached lesson files:")
                for file in cached_files:
                    filename = os.path.basename(file)
                    print(f"  - {filename}")
                
                # Remove all cached files
                for file in cached_files:
                    os.remove(file)
                    print(f"✅ Removed: {os.path.basename(file)}")
                
                print(f"\n🎉 Successfully cleared {len(cached_files)} cached lessons")
            else:
                print("ℹ️  No cached lesson files found")
                
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
    else:
        print("ℹ️  Knowledge store directory doesn't exist")

def clear_wikipedia_cache():
    """Clear Wikipedia cache"""
    
    print("\n🌐 Clearing Wikipedia Cache")
    print("=" * 40)
    
    # Wikipedia cache directory
    wikipedia_cache_dir = "wikipedia_cache"
    
    if os.path.exists(wikipedia_cache_dir):
        try:
            # Get list of cached files
            cached_files = glob.glob(os.path.join(wikipedia_cache_dir, "*.json"))
            
            if cached_files:
                print(f"Found {len(cached_files)} Wikipedia cache files:")
                for file in cached_files:
                    filename = os.path.basename(file)
                    print(f"  - {filename}")
                
                # Remove all cached files
                for file in cached_files:
                    os.remove(file)
                    print(f"✅ Removed: {os.path.basename(file)}")
                
                print(f"\n🎉 Successfully cleared {len(cached_files)} Wikipedia cache files")
            else:
                print("ℹ️  No Wikipedia cache files found")
                
        except Exception as e:
            print(f"❌ Error clearing Wikipedia cache: {e}")
    else:
        print("ℹ️  Wikipedia cache directory doesn't exist")

def clear_all_caches():
    """Clear all caches"""
    
    print("🚀 Clearing All Caches for Fresh Generation")
    print("=" * 50)
    
    clear_knowledge_store_cache()
    clear_wikipedia_cache()
    
    print("\n" + "=" * 50)
    print("✅ All caches cleared!")
    print("💡 Next lesson generation will be completely fresh")
    print("🔄 Different parameter combinations will now produce different results")

if __name__ == "__main__":
    clear_all_caches()
