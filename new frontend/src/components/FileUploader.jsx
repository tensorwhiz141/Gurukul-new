import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Image, FileImage, AlertCircle } from 'lucide-react';
import { toast } from 'react-hot-toast';

/**
 * File Uploader Component with drag-and-drop support
 * Optimized for image uploads for 3D model generation
 */
export default function FileUploader({
  onFileSelect,
  onFileRemove,
  acceptedFileTypes = {
    'image/*': ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
  },
  maxFileSize = 10 * 1024 * 1024, // 10MB
  multiple = false,
  className = '',
  disabled = false,
}) {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [previewUrls, setPreviewUrls] = useState({});

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    // Handle rejected files
    if (rejectedFiles.length > 0) {
      rejectedFiles.forEach(({ file, errors }) => {
        errors.forEach((error) => {
          if (error.code === 'file-too-large') {
            toast.error(`File "${file.name}" is too large. Maximum size is ${maxFileSize / (1024 * 1024)}MB`);
          } else if (error.code === 'file-invalid-type') {
            toast.error(`File "${file.name}" is not a supported image format`);
          } else {
            toast.error(`Error with file "${file.name}": ${error.message}`);
          }
        });
      });
    }

    // Handle accepted files
    if (acceptedFiles.length > 0) {
      const newFiles = acceptedFiles.map(file => ({
        file,
        id: `${file.name}-${Date.now()}`,
        name: file.name,
        size: file.size,
        type: file.type,
      }));

      if (multiple) {
        setUploadedFiles(prev => [...prev, ...newFiles]);
      } else {
        setUploadedFiles(newFiles);
        // Clear previous preview URLs
        Object.values(previewUrls).forEach(url => URL.revokeObjectURL(url));
        setPreviewUrls({});
      }

      // Create preview URLs for images
      newFiles.forEach(({ file, id }) => {
        if (file.type.startsWith('image/')) {
          const url = URL.createObjectURL(file);
          setPreviewUrls(prev => ({ ...prev, [id]: url }));
        }
      });

      // Notify parent component
      if (onFileSelect) {
        if (multiple) {
          onFileSelect(newFiles.map(f => f.file));
        } else {
          onFileSelect(newFiles[0].file);
        }
      }

      toast.success(`${newFiles.length} file(s) uploaded successfully`);
    }
  }, [maxFileSize, multiple, onFileSelect, previewUrls]);

  const removeFile = useCallback((fileId) => {
    setUploadedFiles(prev => {
      const newFiles = prev.filter(f => f.id !== fileId);
      
      // Revoke preview URL
      if (previewUrls[fileId]) {
        URL.revokeObjectURL(previewUrls[fileId]);
        setPreviewUrls(prev => {
          const newUrls = { ...prev };
          delete newUrls[fileId];
          return newUrls;
        });
      }

      // Notify parent component
      if (onFileRemove) {
        const removedFile = prev.find(f => f.id === fileId);
        if (removedFile) {
          onFileRemove(removedFile.file);
        }
      }

      return newFiles;
    });

    toast.success('File removed');
  }, [onFileRemove, previewUrls]);

  const clearAllFiles = useCallback(() => {
    // Revoke all preview URLs
    Object.values(previewUrls).forEach(url => URL.revokeObjectURL(url));
    
    setUploadedFiles([]);
    setPreviewUrls({});
    
    if (onFileRemove) {
      onFileRemove(null);
    }
    
    toast.success('All files cleared');
  }, [onFileRemove, previewUrls]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: acceptedFileTypes,
    maxSize: maxFileSize,
    multiple,
    disabled,
  });

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className={`w-full ${className}`}>
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200
          ${isDragActive && !isDragReject 
            ? 'border-blue-400 bg-blue-500/10' 
            : isDragReject 
            ? 'border-red-400 bg-red-500/10' 
            : 'border-white/20 bg-white/5 hover:border-white/40 hover:bg-white/10'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center space-y-3">
          {isDragReject ? (
            <AlertCircle className="w-12 h-12 text-red-400" />
          ) : (
            <Upload className={`w-12 h-12 ${isDragActive ? 'text-blue-400' : 'text-white/60'}`} />
          )}
          
          <div>
            <p className={`text-lg font-medium ${isDragActive ? 'text-blue-400' : 'text-white'}`}>
              {isDragActive 
                ? isDragReject 
                  ? 'Invalid file type' 
                  : 'Drop files here'
                : 'Drag & drop images here'
              }
            </p>
            <p className="text-white/60 text-sm mt-1">
              or click to browse files
            </p>
          </div>
          
          <div className="text-xs text-white/50">
            <p>Supported formats: PNG, JPG, JPEG, WebP, BMP</p>
            <p>Maximum file size: {formatFileSize(maxFileSize)}</p>
          </div>
        </div>
      </div>

      {/* Uploaded Files List */}
      {uploadedFiles.length > 0 && (
        <div className="mt-4 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-white font-medium">
              Uploaded Files ({uploadedFiles.length})
            </h4>
            {uploadedFiles.length > 1 && (
              <button
                onClick={clearAllFiles}
                className="text-red-400 hover:text-red-300 text-sm transition-colors"
              >
                Clear All
              </button>
            )}
          </div>
          
          <div className="space-y-2">
            {uploadedFiles.map((fileData) => (
              <div
                key={fileData.id}
                className="flex items-center space-x-3 p-3 bg-white/5 rounded-lg border border-white/10"
              >
                {/* File Preview */}
                <div className="flex-shrink-0">
                  {previewUrls[fileData.id] ? (
                    <img
                      src={previewUrls[fileData.id]}
                      alt={fileData.name}
                      className="w-12 h-12 object-cover rounded border border-white/20"
                    />
                  ) : (
                    <div className="w-12 h-12 bg-white/10 rounded border border-white/20 flex items-center justify-center">
                      <FileImage className="w-6 h-6 text-white/60" />
                    </div>
                  )}
                </div>
                
                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium truncate">
                    {fileData.name}
                  </p>
                  <p className="text-white/60 text-sm">
                    {formatFileSize(fileData.size)} • {fileData.type}
                  </p>
                </div>
                
                {/* Remove Button */}
                <button
                  onClick={() => removeFile(fileData.id)}
                  className="flex-shrink-0 p-1 text-red-400 hover:text-red-300 transition-colors"
                  title="Remove file"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
