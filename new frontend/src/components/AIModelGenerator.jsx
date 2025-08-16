import React, { useState, useCallback } from 'react';
import { 
  Wand2, 
  Type, 
  Image as ImageIcon, 
  Settings, 
  Play, 
  Square, 
  Download,
  Trash2,
  Eye,
  Loader2
} from 'lucide-react';
import { useModelGeneration } from '../hooks/useModelGeneration';
import FileUploader from './FileUploader';
import SimpleModelViewer from './SimpleModelViewer';

/**
 * AI Model Generator Component
 * Handles both text-to-3D and image-to-3D generation
 */
export default function AIModelGenerator({
  onModelSelect,
  className = '',
  modelType = 'avatar',
  placeholder = 'e.g., a futuristic robot with glowing blue eyes, a medieval castle with towers, a cute cartoon cat...'
}) {
  const [activeTab, setActiveTab] = useState('text'); // 'text' or 'image'
  const [textPrompt, setTextPrompt] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [generationOptions, setGenerationOptions] = useState({
    format: 'glb',
    quality: 'medium',
    style: 'realistic'
  });
  const [previewModel, setPreviewModel] = useState(null);

  const {
    isGenerating,
    generationProgress,
    generatedModels,
    error,
    generateFromText,
    generateFromImage,
    cancelGeneration,
    removeModel,
    downloadModel,
    supportedProviders
  } = useModelGeneration();

  const handleTextGeneration = useCallback(async () => {
    if (!textPrompt.trim()) {
      return;
    }

    const result = await generateFromText(textPrompt, generationOptions);
    if (result) {
      setPreviewModel(result);
    }
  }, [textPrompt, generationOptions, generateFromText]);

  const handleImageGeneration = useCallback(async () => {
    if (!selectedImage) {
      return;
    }

    const result = await generateFromImage(selectedImage, generationOptions);
    if (result) {
      setPreviewModel(result);
    }
  }, [selectedImage, generationOptions, generateFromImage]);

  const handleModelSelect = useCallback((model) => {
    setPreviewModel(model);
    if (onModelSelect) {
      onModelSelect(model);
    }
  }, [onModelSelect]);

  const tabs = [
    {
      id: 'text',
      name: modelType === 'background' ? 'Text to Environment' : 'Text to 3D',
      icon: Type,
      description: modelType === 'background'
        ? 'Generate 3D environments from text descriptions'
        : 'Generate 3D models from text descriptions'
    },
    {
      id: 'image',
      name: modelType === 'background' ? 'Image to Environment' : 'Image to 3D',
      icon: ImageIcon,
      description: modelType === 'background'
        ? 'Convert images into 3D environments'
        : 'Convert images into 3D models'
    }
  ];

  const qualityOptions = [
    { value: 'low', label: 'Low (Fast)' },
    { value: 'medium', label: 'Medium (Balanced)' },
    { value: 'high', label: 'High (Slow)' }
  ];

  const styleOptions = [
    { value: 'realistic', label: 'Realistic' },
    { value: 'cartoon', label: 'Cartoon' },
    { value: 'stylized', label: 'Stylized' },
    { value: 'low-poly', label: 'Low Poly' }
  ];

  return (
    <div className={`w-full ${className}`}>
      {/* Tab Navigation */}
      <div className="flex space-x-1 mb-6 bg-white/5 rounded-lg p-1">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-md transition-all ${
                activeTab === tab.id
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'text-white/70 hover:text-white hover:bg-white/5'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{tab.name}</span>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Generation Panel */}
        <div className="space-y-6">
          {/* Generation Options */}
          <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Settings className="mr-2 text-purple-400" />
              Generation Settings
            </h3>
            
            <div className="space-y-4">
              {/* Quality Setting */}
              <div>
                <label className="block text-white/80 text-sm font-medium mb-2">
                  Quality
                </label>
                <select
                  value={generationOptions.quality}
                  onChange={(e) => setGenerationOptions(prev => ({ ...prev, quality: e.target.value }))}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                >
                  {qualityOptions.map(option => (
                    <option key={option.value} value={option.value} className="bg-gray-800">
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Style Setting */}
              <div>
                <label className="block text-white/80 text-sm font-medium mb-2">
                  Style
                </label>
                <select
                  value={generationOptions.style}
                  onChange={(e) => setGenerationOptions(prev => ({ ...prev, style: e.target.value }))}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                >
                  {styleOptions.map(option => (
                    <option key={option.value} value={option.value} className="bg-gray-800">
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Format Setting */}
              <div>
                <label className="block text-white/80 text-sm font-medium mb-2">
                  Format
                </label>
                <select
                  value={generationOptions.format}
                  onChange={(e) => setGenerationOptions(prev => ({ ...prev, format: e.target.value }))}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                >
                  <option value="glb" className="bg-gray-800">GLB (Recommended)</option>
                  <option value="gltf" className="bg-gray-800">GLTF</option>
                  <option value="obj" className="bg-gray-800">OBJ</option>
                </select>
              </div>
            </div>
          </div>

          {/* Input Panel */}
          <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Wand2 className="mr-2 text-yellow-400" />
              {tabs.find(t => t.id === activeTab)?.name}
            </h3>

            {activeTab === 'text' ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    {modelType === 'background' ? 'Describe your environment' : 'Describe your 3D model'}
                  </label>
                  <textarea
                    value={textPrompt}
                    onChange={(e) => setTextPrompt(e.target.value)}
                    placeholder={placeholder}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:border-blue-400 focus:outline-none resize-none"
                    rows={4}
                    maxLength={500}
                  />
                  <p className="text-white/50 text-xs mt-1">
                    {textPrompt.length}/500 characters
                  </p>
                </div>

                <button
                  onClick={handleTextGeneration}
                  disabled={!textPrompt.trim() || isGenerating}
                  className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg font-medium hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-2"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Generating... {generationProgress}%</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      <span>{modelType === 'background' ? 'Generate Environment' : 'Generate 3D Model'}</span>
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <FileUploader
                  onFileSelect={setSelectedImage}
                  onFileRemove={() => setSelectedImage(null)}
                  multiple={false}
                  maxFileSize={10 * 1024 * 1024}
                />

                <button
                  onClick={handleImageGeneration}
                  disabled={!selectedImage || isGenerating}
                  className="w-full px-6 py-3 bg-gradient-to-r from-green-500 to-blue-600 text-white rounded-lg font-medium hover:from-green-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-2"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Generating... {generationProgress}%</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      <span>{modelType === 'background' ? 'Generate Environment' : 'Generate 3D Model'}</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Cancel Button */}
            {isGenerating && (
              <button
                onClick={cancelGeneration}
                className="w-full mt-3 px-6 py-2 bg-red-500/20 text-red-400 rounded-lg border border-red-500/30 hover:bg-red-500/30 transition-colors flex items-center justify-center space-x-2"
              >
                <Square className="w-4 h-4" />
                <span>Cancel Generation</span>
              </button>
            )}

            {/* Error Display */}
            {error && (
              <div className="mt-4 p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-red-400 text-sm">
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Preview Panel */}
        <div className="space-y-6">
          {/* 3D Model Preview */}
          <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Eye className="mr-2 text-green-400" />
              3D Preview
            </h3>
            
            <div className="aspect-square bg-black/20 rounded-lg border border-white/10 overflow-hidden">
              {previewModel ? (
                <SimpleModelViewer
                  modelPath={previewModel.url}
                  enableControls={true}
                  autoRotate={false}
                  showEnvironment={true}
                  fallbackMessage="Loading Model..."
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-white/50">
                  <div className="text-center">
                    <Eye className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>Generate a model to see preview</p>
                  </div>
                </div>
              )}
            </div>

            {previewModel && (
              <div className="mt-4 flex space-x-2">
                <button
                  onClick={() => downloadModel(previewModel)}
                  className="flex-1 px-4 py-2 bg-blue-500/20 text-blue-400 rounded-lg border border-blue-500/30 hover:bg-blue-500/30 transition-colors flex items-center justify-center space-x-2"
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </button>
                <button
                  onClick={() => handleModelSelect(previewModel)}
                  className="flex-1 px-4 py-2 bg-green-500/20 text-green-400 rounded-lg border border-green-500/30 hover:bg-green-500/30 transition-colors"
                >
                  {modelType === 'background' ? 'Select Background' : 'Select as Avatar'}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
