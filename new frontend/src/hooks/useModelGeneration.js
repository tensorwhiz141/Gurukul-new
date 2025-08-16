import { useState, useCallback, useRef, useEffect } from 'react';
import { aiModelService } from '../services/aiModelService';
import { toast } from 'react-hot-toast';

/**
 * Custom hook for AI 3D model generation
 * Handles text-to-3D and image-to-3D generation with progress tracking
 */
export function useModelGeneration() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [generatedModels, setGeneratedModels] = useState([]);
  const [currentTask, setCurrentTask] = useState(null);
  const [error, setError] = useState(null);
  
  const pollIntervalRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  /**
   * Generate 3D model from text prompt
   */
  const generateFromText = useCallback(async (prompt, options = {}) => {
    // Validate prompt
    const validation = aiModelService.constructor.validatePrompt(prompt);
    if (!validation.valid) {
      setError(validation.error);
      toast.error(validation.error);
      return null;
    }

    setIsGenerating(true);
    setError(null);
    setGenerationProgress(0);
    
    // Create abort controller for this generation
    abortControllerRef.current = new AbortController();

    try {
      toast.loading('Starting text-to-3D generation...', { id: 'generation-progress' });
      
      // Start generation
      const response = await aiModelService.generateFromText(prompt, options);
      
      if (response.task_id) {
        setCurrentTask(response.task_id);
        await pollGenerationStatus(response.task_id, 'text-to-3D');
      } else if (response.model_url) {
        // Direct response with model URL
        const newModel = {
          id: Date.now().toString(),
          type: 'text-to-3d',
          prompt,
          url: response.model_url,
          thumbnail: response.thumbnail_url,
          createdAt: new Date().toISOString(),
          format: options.format || 'glb',
        };
        
        setGeneratedModels(prev => [newModel, ...prev]);
        toast.success('3D model generated successfully!', { id: 'generation-progress' });
        return newModel;
      }
    } catch (err) {
      console.error('Text-to-3D generation error:', err);
      setError(err.message);
      toast.error(`Generation failed: ${err.message}`, { id: 'generation-progress' });
      return null;
    } finally {
      setIsGenerating(false);
      setCurrentTask(null);
      setGenerationProgress(0);
    }
  }, []);

  /**
   * Generate 3D model from image
   */
  const generateFromImage = useCallback(async (imageFile, options = {}) => {
    if (!imageFile) {
      const errorMsg = 'Image file is required';
      setError(errorMsg);
      toast.error(errorMsg);
      return null;
    }

    setIsGenerating(true);
    setError(null);
    setGenerationProgress(0);
    
    // Create abort controller for this generation
    abortControllerRef.current = new AbortController();

    try {
      toast.loading('Starting image-to-3D generation...', { id: 'generation-progress' });
      
      // Start generation
      const response = await aiModelService.generateFromImage(imageFile, options);
      
      if (response.task_id) {
        setCurrentTask(response.task_id);
        await pollGenerationStatus(response.task_id, 'image-to-3D');
      } else if (response.model_url) {
        // Direct response with model URL
        const newModel = {
          id: Date.now().toString(),
          type: 'image-to-3d',
          sourceImage: URL.createObjectURL(imageFile),
          url: response.model_url,
          thumbnail: response.thumbnail_url,
          createdAt: new Date().toISOString(),
          format: options.format || 'glb',
        };
        
        setGeneratedModels(prev => [newModel, ...prev]);
        toast.success('3D model generated successfully!', { id: 'generation-progress' });
        return newModel;
      }
    } catch (err) {
      console.error('Image-to-3D generation error:', err);
      setError(err.message);
      toast.error(`Generation failed: ${err.message}`, { id: 'generation-progress' });
      return null;
    } finally {
      setIsGenerating(false);
      setCurrentTask(null);
      setGenerationProgress(0);
    }
  }, []);

  /**
   * Poll generation status
   */
  const pollGenerationStatus = useCallback(async (taskId, generationType) => {
    let attempts = 0;
    const maxAttempts = 60; // 5 minutes with 5-second intervals
    
    pollIntervalRef.current = setInterval(async () => {
      try {
        attempts++;
        
        if (attempts > maxAttempts) {
          clearInterval(pollIntervalRef.current);
          throw new Error('Generation timeout - please try again');
        }

        const status = await aiModelService.checkStatus(taskId);
        
        // Update progress based on status
        if (status.progress) {
          setGenerationProgress(status.progress);
          toast.loading(`Generating... ${status.progress}%`, { id: 'generation-progress' });
        }

        if (status.status === 'completed' && status.model_url) {
          clearInterval(pollIntervalRef.current);
          
          const newModel = {
            id: taskId,
            type: generationType,
            url: status.model_url,
            thumbnail: status.thumbnail_url,
            createdAt: new Date().toISOString(),
            format: status.format || 'glb',
          };
          
          setGeneratedModels(prev => [newModel, ...prev]);
          setGenerationProgress(100);
          toast.success('3D model generated successfully!', { id: 'generation-progress' });
          
        } else if (status.status === 'failed') {
          clearInterval(pollIntervalRef.current);
          throw new Error(status.error || 'Generation failed');
        }
        
      } catch (err) {
        clearInterval(pollIntervalRef.current);
        console.error('Status polling error:', err);
        setError(err.message);
        toast.error(`Generation failed: ${err.message}`, { id: 'generation-progress' });
      }
    }, 5000); // Poll every 5 seconds
  }, []);

  /**
   * Cancel current generation
   */
  const cancelGeneration = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
    
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    setIsGenerating(false);
    setCurrentTask(null);
    setGenerationProgress(0);
    toast.dismiss('generation-progress');
    toast.success('Generation cancelled');
  }, []);

  /**
   * Remove a generated model
   */
  const removeModel = useCallback((modelId) => {
    setGeneratedModels(prev => prev.filter(model => model.id !== modelId));
    toast.success('Model removed');
  }, []);

  /**
   * Clear all generated models
   */
  const clearAllModels = useCallback(() => {
    setGeneratedModels([]);
    toast.success('All models cleared');
  }, []);

  /**
   * Download a model
   */
  const downloadModel = useCallback(async (model) => {
    try {
      toast.loading('Downloading model...', { id: 'download-progress' });
      
      const filename = `${model.type}-${model.id}.${model.format}`;
      await aiModelService.downloadModel(model.url, filename);
      
      toast.success('Model downloaded successfully!', { id: 'download-progress' });
    } catch (err) {
      console.error('Download error:', err);
      toast.error(`Download failed: ${err.message}`, { id: 'download-progress' });
    }
  }, []);

  /**
   * Set AI provider
   */
  const setProvider = useCallback((providerName) => {
    try {
      aiModelService.setProvider(providerName);
      toast.success(`Switched to ${providerName}`);
    } catch (err) {
      toast.error(err.message);
    }
  }, []);

  /**
   * Set API key
   */
  const setApiKey = useCallback((apiKey) => {
    aiModelService.setApiKey(apiKey);
    toast.success('API key updated');
  }, []);

  return {
    // State
    isGenerating,
    generationProgress,
    generatedModels,
    currentTask,
    error,
    
    // Actions
    generateFromText,
    generateFromImage,
    cancelGeneration,
    removeModel,
    clearAllModels,
    downloadModel,
    setProvider,
    setApiKey,
    
    // Utilities
    supportedProviders: aiModelService.constructor.getSupportedProviders(),
  };
}
