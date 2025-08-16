/**
 * IndexedDB Storage Utility for Large Files (1GB+ capacity)
 * Specifically designed for storing .glb 3D models and other large binary data
 */

class IndexedDBStorage {
  constructor() {
    this.dbName = 'GurukulStorage';
    this.dbVersion = 1;
    this.db = null;
    this.stores = {
      customModels: 'customModels',
      metadata: 'metadata'
    };
  }

  /**
   * Initialize IndexedDB connection
   */
  async init() {
    if (this.db) {
      console.log('🔄 IndexedDB: Using existing connection');
      return this.db;
    }

    console.log('🔄 IndexedDB: Opening database...', this.dbName, 'version', this.dbVersion);

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => {
        console.error('❌ IndexedDB: Database open error:', request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        console.log('✅ IndexedDB: Database opened successfully');
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        console.log('🔄 IndexedDB: Database upgrade needed');
        const db = event.target.result;

        // Create object stores
        if (!db.objectStoreNames.contains(this.stores.customModels)) {
          console.log('🔄 IndexedDB: Creating customModels store');
          const modelsStore = db.createObjectStore(this.stores.customModels, { keyPath: 'id' });
          modelsStore.createIndex('name', 'name', { unique: false });
          modelsStore.createIndex('uploadDate', 'uploadDate', { unique: false });
        }

        if (!db.objectStoreNames.contains(this.stores.metadata)) {
          console.log('🔄 IndexedDB: Creating metadata store');
          const metadataStore = db.createObjectStore(this.stores.metadata, { keyPath: 'key' });
        }

        console.log('✅ IndexedDB: Stores created/upgraded');
      };

      request.onblocked = () => {
        console.warn('⚠️ IndexedDB: Database upgrade blocked by another connection');
      };
    });
  }

  /**
   * Store a .glb file with metadata
   */
  async saveGlbFile(id, fileData, metadata) {
    try {
      console.log('🔄 IndexedDB: Initializing database...');
      await this.init();

      console.log('🔄 IndexedDB: Creating transaction...');
      const transaction = this.db.transaction([this.stores.customModels], 'readwrite');
      const store = transaction.objectStore(this.stores.customModels);

      const modelData = {
        id,
        ...metadata,
        fileData, // Store the actual file data (base64 or blob)
        storedAt: new Date().toISOString()
      };

      console.log('🔄 IndexedDB: Putting data...', {
        id,
        dataSize: fileData.length,
        metadataKeys: Object.keys(metadata)
      });

      const request = store.put(modelData);

      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          console.log(`✅ IndexedDB: Model ${id} saved successfully`);
          resolve(true);
        };

        request.onerror = () => {
          console.error('❌ IndexedDB: Error saving model:', request.error);
          reject(request.error);
        };

        transaction.onerror = () => {
          console.error('❌ IndexedDB: Transaction error:', transaction.error);
          reject(transaction.error);
        };

        transaction.onabort = () => {
          console.error('❌ IndexedDB: Transaction aborted:', transaction.error);
          reject(new Error('Transaction aborted: ' + transaction.error));
        };
      });
    } catch (error) {
      console.error('❌ IndexedDB: Error in saveGlbFile:', error);
      throw error;
    }
  }

  /**
   * Load a .glb file by ID
   */
  async loadGlbFile(id) {
    try {
      await this.init();
      
      const transaction = this.db.transaction([this.stores.customModels], 'readonly');
      const store = transaction.objectStore(this.stores.customModels);
      const request = store.get(id);
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          if (request.result) {
            console.log(`Model ${id} loaded from IndexedDB`);
            resolve({
              data: request.result.fileData,
              metadata: request.result
            });
          } else {
            console.warn(`Model ${id} not found in IndexedDB`);
            resolve(null);
          }
        };
        
        request.onerror = () => {
          console.error('Error loading model:', request.error);
          reject(request.error);
        };
      });
    } catch (error) {
      console.error('Error in loadGlbFile:', error);
      throw error;
    }
  }

  /**
   * Delete a .glb file by ID
   */
  async deleteGlbFile(id) {
    try {
      await this.init();
      
      const transaction = this.db.transaction([this.stores.customModels], 'readwrite');
      const store = transaction.objectStore(this.stores.customModels);
      const request = store.delete(id);
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          console.log(`Model ${id} deleted from IndexedDB`);
          resolve(true);
        };
        
        request.onerror = () => {
          console.error('Error deleting model:', request.error);
          reject(request.error);
        };
      });
    } catch (error) {
      console.error('Error in deleteGlbFile:', error);
      throw error;
    }
  }

  /**
   * Get all custom models metadata
   */
  async getAllModels() {
    try {
      await this.init();
      
      const transaction = this.db.transaction([this.stores.customModels], 'readonly');
      const store = transaction.objectStore(this.stores.customModels);
      const request = store.getAll();
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          const models = request.result.map(model => ({
            id: model.id,
            name: model.name,
            fileName: model.fileName,
            fileSize: model.fileSize,
            uploadDate: model.uploadDate,
            thumbnailUrl: model.thumbnailUrl,
            type: model.type,
            isCustom: model.isCustom,
            storedAt: model.storedAt
          }));
          resolve(models);
        };
        
        request.onerror = () => {
          console.error('Error getting all models:', request.error);
          reject(request.error);
        };
      });
    } catch (error) {
      console.error('Error in getAllModels:', error);
      throw error;
    }
  }

  /**
   * Get storage usage information
   */
  async getStorageInfo() {
    try {
      await this.init();
      
      // Get storage estimate (if supported)
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        const estimate = await navigator.storage.estimate();
        const usedMB = (estimate.usage / (1024 * 1024)).toFixed(2);
        const quotaMB = (estimate.quota / (1024 * 1024)).toFixed(0);
        const availableMB = ((estimate.quota - estimate.usage) / (1024 * 1024)).toFixed(2);
        const percentUsed = ((estimate.usage / estimate.quota) * 100).toFixed(1);
        
        return {
          usedMB,
          quotaMB,
          availableMB,
          percentUsed,
          supported: true
        };
      } else {
        // Fallback for browsers without storage estimate
        return {
          usedMB: '0',
          quotaMB: '1000', // Assume 1GB
          availableMB: '1000',
          percentUsed: '0',
          supported: false
        };
      }
    } catch (error) {
      console.error('Error getting storage info:', error);
      return {
        usedMB: '0',
        quotaMB: '1000',
        availableMB: '1000',
        percentUsed: '0',
        supported: false
      };
    }
  }

  /**
   * Clear all custom models
   */
  async clearAllModels() {
    try {
      await this.init();
      
      const transaction = this.db.transaction([this.stores.customModels], 'readwrite');
      const store = transaction.objectStore(this.stores.customModels);
      const request = store.clear();
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          console.log('All models cleared from IndexedDB');
          resolve(true);
        };
        
        request.onerror = () => {
          console.error('Error clearing models:', request.error);
          reject(request.error);
        };
      });
    } catch (error) {
      console.error('Error in clearAllModels:', error);
      throw error;
    }
  }

  /**
   * Check if IndexedDB is supported
   */
  static isSupported() {
    return 'indexedDB' in window;
  }
}

// Create singleton instance
const indexedDBStorage = new IndexedDBStorage();

export default indexedDBStorage;
