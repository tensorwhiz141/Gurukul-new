import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { toast } from 'react-hot-toast';
import { 
  Upload, 
  Trash2, 
  Heart, 
  Download,
  FileText,
  Calendar,
  HardDrive,
  AlertCircle,
  CheckCircle
} from 'lucide-react';
import {
  selectCustomModels,
  selectCustomModelsLoading,
  selectCustomModelsError,
  selectSelectedAvatar,
  addCustomModel,
  removeCustomModel,
  setCustomModelsLoading,
  setCustomModelsError,
  setSelectedAvatar,
  loadAvatarSettings,
  addFavorite,
} from '../store/avatarSlice';
import { storage } from '../utils/storageUtils';
import indexedDBStorage from '../utils/indexedDBStorage';
import AvatarViewer from './AvatarViewer';

const CustomModelsTab = () => {
  const dispatch = useDispatch();
  const customModels = useSelector(selectCustomModels);
  const loading = useSelector(selectCustomModelsLoading);
  const error = useSelector(selectCustomModelsError);
  const selectedAvatar = useSelector(selectSelectedAvatar);

  const [dragOver, setDragOver] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [storageInfo, setStorageInfo] = useState({ usedMB: '0', quotaMB: '1000', availableMB: '1000', percentUsed: '0' });
  const fileInputRef = useRef(null);

  // Load storage info on component mount and after operations
  useEffect(() => {
    const loadStorageInfo = async () => {
      console.log('🔄 Loading storage info...');
      const info = await checkStorageSpace();
      console.log('📊 Storage info:', info);
      setStorageInfo(info);
    };

    loadStorageInfo();
  }, [customModels.length]); // Reload when models change

  // Add warning about potential data loss (only for unsaved changes)
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (uploadProgress) {
        const message = 'Upload in progress. Leaving now will cancel the upload.';
        e.preventDefault();
        e.returnValue = message;
        return message;
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [uploadProgress]);

  // Check IndexedDB storage space (much larger capacity)
  const checkStorageSpace = async () => {
    try {
      const storageInfo = await indexedDBStorage.getStorageInfo();
      return storageInfo;
    } catch (error) {
      console.error('Error checking IndexedDB storage:', error);
      return {
        usedMB: '0',
        quotaMB: '1000',
        availableMB: '1000',
        percentUsed: '0',
        supported: false
      };
    }
  };

  // File validation with IndexedDB (much larger capacity)
  const validateFile = async (file) => {
    console.log('🔍 Validating file:', file.name, 'Size:', file.size);

    const allowedTypes = ['.glb'];

    // Much more generous file size limits with IndexedDB
    const maxSize = 100 * 1024 * 1024; // 100MB per file (reasonable for 3D models)

    if (file.size > maxSize) {
      const maxSizeMB = (maxSize / (1024 * 1024)).toFixed(0);
      throw new Error(`File size must be less than ${maxSizeMB}MB. Try compressing your model if it's larger.`);
    }

    const extension = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
    if (!allowedTypes.includes(extension)) {
      throw new Error('Only .glb files are supported');
    }

    // Skip storage space check for now to isolate the issue
    console.log('✅ File validation passed');
    return true;
  };

  // Convert file to base64 with timeout
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      console.log('🔄 Starting FileReader for:', file.name);
      const reader = new FileReader();

      // Add timeout to prevent hanging
      const timeout = setTimeout(() => {
        reader.abort();
        reject(new Error('File reading timed out after 30 seconds'));
      }, 30000);

      reader.onloadstart = () => {
        console.log('🔄 FileReader: Load started');
      };

      reader.onprogress = (e) => {
        if (e.lengthComputable) {
          const progress = (e.loaded / e.total) * 100;
          console.log(`🔄 FileReader: Progress ${progress.toFixed(1)}%`);
        }
      };

      reader.onload = () => {
        clearTimeout(timeout);
        console.log('✅ FileReader: Load complete, result length:', reader.result?.length);
        resolve(reader.result);
      };

      reader.onerror = (error) => {
        clearTimeout(timeout);
        console.error('❌ FileReader: Error:', error);
        reject(error);
      };

      reader.onabort = () => {
        clearTimeout(timeout);
        console.error('❌ FileReader: Aborted');
        reject(new Error('File reading was aborted'));
      };

      reader.readAsDataURL(file);
    });
  };

  // Generate thumbnail (placeholder for now)
  const generateThumbnail = async (file) => {
    // TODO: Implement actual 3D model thumbnail generation
    // For now, return the file data itself as the preview URL
    const base64Data = await fileToBase64(file);
    return base64Data;
  };

  // Handle file upload
  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return;

    console.log('🔄 Starting file upload for:', files.length, 'files');
    dispatch(setCustomModelsLoading(true));
    dispatch(setCustomModelsError(null));

    try {
      for (const file of files) {
        console.log('📁 Processing file:', file.name, 'Size:', file.size);
        setUploadProgress({ fileName: file.name, progress: 0 });

        // Add a small delay to ensure UI updates
        await new Promise(resolve => setTimeout(resolve, 100));

        // Validate file
        console.log('✅ Validating file...');
        await validateFile(file);
        console.log('✅ File validation complete');

        setUploadProgress({ fileName: file.name, progress: 25 });
        await new Promise(resolve => setTimeout(resolve, 100));

        // Convert to base64
        console.log('🔄 Converting to base64...');
        const base64Data = await fileToBase64(file);
        console.log('✅ Base64 conversion complete, length:', base64Data.length);

        setUploadProgress({ fileName: file.name, progress: 50 });
        await new Promise(resolve => setTimeout(resolve, 100));

        // Generate thumbnail
        console.log('🖼️ Generating thumbnail...');
        const thumbnailUrl = await generateThumbnail(file);
        console.log('✅ Thumbnail generation complete');

        setUploadProgress({ fileName: file.name, progress: 75 });
        await new Promise(resolve => setTimeout(resolve, 100));

        // Create model metadata
        const modelId = `custom_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const metadata = {
          id: modelId,
          name: file.name.replace(/\.[^/.]+$/, ""), // Remove extension
          fileName: file.name,
          fileSize: file.size,
          uploadDate: new Date().toISOString(),
          thumbnailUrl,
          type: 'custom',
          isCustom: true,
        };

        console.log('💾 Saving to IndexedDB...', modelId);

        // Save file data to IndexedDB with error handling
        try {
          const success = await indexedDBStorage.saveGlbFile(modelId, base64Data, metadata);
          if (!success) {
            throw new Error('Failed to save file to IndexedDB');
          }
          console.log('✅ Successfully saved to IndexedDB');
        } catch (storageError) {
          console.error('❌ IndexedDB storage error:', storageError);
          if (storageError.name === 'QuotaExceededError' || storageError.message.includes('quota')) {
            throw new Error('Storage quota exceeded. Please delete some models or try a smaller file.');
          }
          throw new Error('Failed to save file to storage. Please try again.');
        }

        // Add to custom models
        console.log('📝 Adding to Redux store...');
        dispatch(addCustomModel(metadata));

        setUploadProgress({ fileName: file.name, progress: 100 });
        await new Promise(resolve => setTimeout(resolve, 100));

        console.log('🎉 Upload complete for:', file.name);
        toast.success(`${file.name} uploaded successfully!`);
      }
    } catch (error) {
      console.error('❌ Upload error:', error);
      dispatch(setCustomModelsError(error.message));
      toast.error(`Upload failed: ${error.message}`);
    } finally {
      dispatch(setCustomModelsLoading(false));
      setUploadProgress(null);
    }
  };

  // Handle drag and drop
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    handleFileUpload(files);
  }, []);

  // Handle file input change
  const handleFileInputChange = (e) => {
    const files = Array.from(e.target.files);
    handleFileUpload(files);
    e.target.value = ''; // Reset input
  };

  // Handle model selection
  const handleSelectModel = async (model) => {
    try {
      // Load the file data from IndexedDB
      const fileData = await indexedDBStorage.loadGlbFile(model.id);
      if (!fileData) {
        toast.error('Failed to load model file');
        return;
      }

      // Create blob URL from base64 data for better performance
      const base64Data = fileData.data.split(',')[1]; // Remove data:application/octet-stream;base64, prefix
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: 'application/octet-stream' });
      const blobUrl = URL.createObjectURL(blob);

      // Create avatar object with blob URL
      const avatarWithData = {
        ...model,
        fileData: fileData.data, // Keep original base64 for storage
        previewUrl: blobUrl, // Use blob URL for Three.js
        isBlobUrl: true, // Flag to clean up later
        isCustomModel: true, // Flag for settings panel
      };

      dispatch(setSelectedAvatar(avatarWithData));
      dispatch(loadAvatarSettings(avatarWithData));
      toast.success(`Selected ${model.name}`);
    } catch (error) {
      console.error('Error selecting model:', error);
      toast.error('Failed to load model');
    }
  };

  // Handle model deletion
  const handleDeleteModel = async (model) => {
    if (window.confirm(`Are you sure you want to delete "${model.name}"?`)) {
      try {
        // Delete file from IndexedDB
        await indexedDBStorage.deleteGlbFile(model.id);

        // Remove from Redux state
        dispatch(removeCustomModel(model.id));

        // If this was the selected avatar, clear selection
        if (selectedAvatar?.id === model.id) {
          dispatch(setSelectedAvatar(null));
        }

        // Show storage info after deletion
        const storageInfo = await checkStorageSpace();
        toast.success(`${model.name} deleted. Storage: ${storageInfo.percentUsed}% used`);
      } catch (error) {
        console.error('Error deleting model:', error);
        toast.error('Failed to delete model');
      }
    }
  };



  // Clear all custom models
  const handleClearAllModels = async () => {
    if (customModels.length === 0) {
      toast.error('No models to delete');
      return;
    }

    if (window.confirm(`Are you sure you want to delete all ${customModels.length} custom models? This cannot be undone.`)) {
      try {
        // Clear all files from IndexedDB
        await indexedDBStorage.clearAllModels();

        // Clear Redux state
        customModels.forEach(model => {
          dispatch(removeCustomModel(model.id));
        });

        // Clear selection if it was a custom model
        if (selectedAvatar?.isCustomModel) {
          dispatch(setSelectedAvatar(null));
        }

        const storageInfo = await checkStorageSpace();
        toast.success(`All custom models deleted. Storage: ${storageInfo.percentUsed}% used`);
      } catch (error) {
        console.error('Error clearing all models:', error);
        toast.error('Failed to clear all models');
      }
    }
  };

  // Handle add to favorites
  const handleAddToFavorites = async (model) => {
    try {
      // Load the file data from IndexedDB to get the blob URL
      const fileData = await indexedDBStorage.loadGlbFile(model.id);
      if (!fileData) {
        toast.error('Failed to load model file');
        return;
      }

      // Create blob URL from base64 data
      const base64Data = fileData.data.split(',')[1];
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: 'application/octet-stream' });
      const blobUrl = URL.createObjectURL(blob);

      const favorite = {
        ...model,
        id: `fav_${model.id}`,
        originalId: model.id,
        timestamp: new Date().toISOString(),
        activeTab: 'custom-models',
        previewUrl: blobUrl, // Use blob URL for preview
        fileData: fileData.data, // Keep original base64 for storage
        isCustomModel: true,
        // Add default positioning
        gridPosition: { x: 0, y: 0, z: 0 },
        gridRotation: { x: 0, y: 0, z: 0 },
        gridScale: 1,
        pinPosition: { x: 0, y: 0, z: 0 },
        pinRotation: { x: 0, y: 0, z: 0 },
        pinScale: 1,
      };

      dispatch(addFavorite(favorite));
      toast.success(`${model.name} added to favorites!`);
    } catch (error) {
      console.error('Error adding to favorites:', error);
      toast.error('Failed to add to favorites');
    }
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Format date
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-white">Custom Models</h2>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="text-sm text-white/60">
              {customModels.length} model{customModels.length !== 1 ? 's' : ''}
            </div>
            <div className="text-xs text-white/40">
              Storage: {storageInfo.usedMB}MB / {storageInfo.quotaMB}MB ({storageInfo.percentUsed}%)
            </div>
            <div className="text-xs text-white/30">
              Available: {storageInfo.availableMB}MB
            </div>
          </div>
          {customModels.length > 0 && (
            <button
              onClick={handleClearAllModels}
              className="px-3 py-1 bg-red-500/20 border border-red-500/30 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors text-sm"
              title="Delete all custom models"
            >
              Clear All
            </button>
          )}
        </div>
      </div>

      {/* Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-6 mb-6 transition-all ${
          dragOver
            ? 'border-orange-400 bg-orange-500/10'
            : 'border-white/20 hover:border-white/40'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="text-center">
          <Upload className="w-8 h-8 text-white/40 mx-auto mb-3" />
          <h3 className="text-lg font-medium text-white mb-2">
            Upload 3D Models
          </h3>
          <p className="text-white/60 mb-3">
            Drag and drop .glb files here, or click to browse
          </p>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded-lg transition-colors text-sm"
            disabled={loading}
          >
            {loading ? 'Uploading...' : 'Choose Files'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".glb"
            multiple
            onChange={handleFileInputChange}
            className="hidden"
          />
        </div>
      </div>

      {/* Upload Progress */}
      {uploadProgress && (
        <div className="mb-4 p-3 bg-white/5 rounded-lg border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-white text-sm">{uploadProgress.fileName}</span>
            <span className="text-white/60 text-sm">{uploadProgress.progress}%</span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2">
            <div
              className="bg-orange-500 h-2 rounded-full transition-all"
              style={{ width: `${uploadProgress.progress}%` }}
            />
          </div>
        </div>
      )}





      {/* Storage Warning */}
      {parseFloat(storageInfo.percentUsed) > 80 && (
        <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg flex items-center gap-3">
          <AlertCircle className="w-4 h-4 text-yellow-400" />
          <div className="text-yellow-400 text-sm">
            <div>Storage is {storageInfo.percentUsed}% full</div>
            <div className="text-xs text-yellow-400/70 mt-1">
              Consider deleting unused models to free up space
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-3">
          <AlertCircle className="w-4 h-4 text-red-400" />
          <span className="text-red-400 text-sm">{error}</span>
        </div>
      )}

      {/* Models Grid */}
      <div className="flex-1 overflow-y-auto">
        {customModels.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-white/40">
            <FileText className="w-16 h-16 mb-4" />
            <p className="text-lg">No custom models uploaded yet</p>
            <p className="text-sm">Upload your first .glb model to get started</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {customModels.map((model) => (
              <div
                key={model.id}
                className={`bg-white/5 rounded-lg border overflow-hidden hover:border-orange-500/30 transition-all group cursor-pointer relative ${
                  selectedAvatar?.id === model.id
                    ? 'border-orange-500/50 bg-orange-500/10'
                    : 'border-white/10'
                }`}
                onClick={() => handleSelectModel(model)}
                style={{ zIndex: 1 }}
              >
                {/* Model Preview */}
                <div className="aspect-square bg-black/20 relative overflow-hidden">
                  <div className="w-full h-full flex items-center justify-center">
                    <FileText className="w-8 h-8 text-white/40" />
                    <span className="text-xs text-white/40 ml-2">GLB</span>
                  </div>

                  {/* Hover overlay */}
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all flex items-center justify-center opacity-0 group-hover:opacity-100 z-10">
                    <div className="text-white text-sm font-medium">Select</div>
                  </div>

                  {/* Delete button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteModel(model);
                    }}
                    className="absolute top-2 right-2 p-1 bg-black/50 backdrop-blur-sm rounded-full border border-white/20 hover:bg-red-500/50 transition-all opacity-0 group-hover:opacity-100 z-20"
                  >
                    <Trash2 className="w-3 h-3 text-white" />
                  </button>

                  {/* Add to favorites button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleAddToFavorites(model);
                    }}
                    className="absolute top-2 left-2 p-1 bg-black/50 backdrop-blur-sm rounded-full border border-white/20 hover:bg-orange-500/50 transition-all opacity-0 group-hover:opacity-100 z-20"
                    title="Add to Favorites"
                  >
                    <Heart className="w-3 h-3 text-white" />
                  </button>
                </div>

                {/* Model Info */}
                <div className="p-3">
                  <h4 className="text-white font-medium text-sm truncate text-center mb-1">
                    {model.name}
                  </h4>
                  <div className="text-xs text-white/60 text-center">
                    {formatFileSize(model.fileSize)}
                  </div>
                  <div className="text-xs text-white/40 text-center">
                    {formatDate(model.uploadDate)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default CustomModelsTab;
