import React, { useState, useEffect } from "react";
import GlassContainer from "../components/GlassContainer";
import GlassInput from "../components/GlassInput";
import GlassButton from "../components/GlassButton";
import { useJupiterTTS } from "../hooks/useTTS";
import {
  useLazyGetLessonDataQuery,
  useCreateLessonMutation,
  useLazyGenerateLessonQuery,
} from "../api/subjectsApiSlice";
import {
  useGenerateEnhancedLessonMutation,
  useGetUserProgressQuery,
  useGetUserAnalyticsQuery,
  useTriggerInterventionMutation,
  useGetIntegrationStatusQuery,
  formatEnhancedLessonData,
  formatUserProgressData
} from "../api/orchestrationApiSlice";
import { Book, BookOpen, Search, AlertCircle, BarChart3, Settings } from "lucide-react";
import { useSelector } from "react-redux";
import { selectUserId } from "../store/authSlice";
import { toast } from "react-hot-toast";
import { API_BASE_URL, CHAT_API_BASE_URL } from "../config";
import { useVideo } from "../context/VideoContext";
import UserProgressDashboard from "../components/UserProgressDashboard";

export default function Subjects() {
  // Get user ID first (needed for hooks)
  const userId = useSelector(selectUserId) || "guest-user";

  // API hooks for lesson data
  const [
    getLessonData,
    {
      data: existingLessonData,
      isLoading: isLoadingExisting,
      isError: isErrorExisting,
    },
  ] = useLazyGetLessonDataQuery();

  const [
    createLesson,
    { data: newLessonData, isLoading: isLoadingNew, isError: isErrorNew },
  ] = useCreateLessonMutation();

  // Direct lesson generation hook
  const [
    generateLesson,
    { data: generatedLessonData, isLoading: isLoadingGenerated, isError: isErrorGenerated },
  ] = useLazyGenerateLessonQuery();

  // Orchestration API hooks
  const [
    generateEnhancedLesson,
    { data: enhancedLessonData, isLoading: isLoadingEnhanced, isError: isErrorEnhanced },
  ] = useGenerateEnhancedLessonMutation();

  const [
    triggerIntervention,
    { isLoading: isTriggeringIntervention },
  ] = useTriggerInterventionMutation();

  // Get integration status to check if orchestration is available
  const { data: integrationStatus, isLoading: isLoadingIntegration } = useGetIntegrationStatusQuery();

  // Get user progress if orchestration is available
  const { data: userProgress, isLoading: isLoadingProgress } = useGetUserProgressQuery(userId, {
    skip: !integrationStatus?.integration_status?.overall_valid || !userId || userId === "guest-user"
  });

  // Get user analytics if orchestration is available
  const { data: userAnalytics, isLoading: isLoadingAnalytics } = useGetUserAnalyticsQuery(userId, {
    skip: !integrationStatus?.integration_status?.overall_valid || !userId || userId === "guest-user"
  });

  // Video context
  const { generatedVideo, showVideoInSidebar, showVideo, hideVideo } = useVideo();

  // Component state
  const [selectedSubject, setSelectedSubject] = useState("");
  const [topic, setTopic] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [isButtonHovered, setIsButtonHovered] = useState(false);
  const [lessonData, setLessonData] = useState(null);
  const [includeWikipedia, setIncludeWikipedia] = useState(true);
  const [useKnowledgeStore, setUseKnowledgeStore] = useState(true);

  // Orchestration-specific state
  const [useOrchestration, setUseOrchestration] = useState(true);
  const [showProgressDashboard, setShowProgressDashboard] = useState(false);
  const [lastQuizScore, setLastQuizScore] = useState(null);
  const [showInterventionPanel, setShowInterventionPanel] = useState(false);

  // Edge case handling states
  const [retryCount, setRetryCount] = useState(0);
  const [extendedWaitMode, setExtendedWaitMode] = useState(false);
  const [fallbackContentAvailable, setFallbackContentAvailable] = useState(false);
  const [isOfflineMode, setIsOfflineMode] = useState(false);
  const [lastError, setLastError] = useState(null);
  const maxRetries = 3;
  const extendedWaitThreshold = 300; // 5 minutes

  // TTS integration for Jupiter model responses
  const { handleJupiterResponse, serviceHealthy, isPlaying, stopTTS } = useJupiterTTS({
    onPlayStart: (text) => {
      console.log("🔊 Jupiter TTS: Started playing lesson content");
    },
    onPlayEnd: (text) => {
      console.log("🔊 Jupiter TTS: Finished playing lesson content");
    },
    onError: (error) => {
      console.warn("🔊 Jupiter TTS: Auto-play failed:", error.message);
    }
  });

  // Handler for triggering interventions
  const handleTriggerIntervention = async () => {
    if (!userId || userId === "guest-user") {
      toast.error("Please log in to access personalized interventions.");
      return;
    }

    try {
      const result = await triggerIntervention({
        user_id: userId,
        quiz_score: lastQuizScore
      }).unwrap();

      if (result.interventions && result.interventions.length > 0) {
        toast.success(
          `✅ ${result.interventions.length} intervention(s) triggered successfully!`,
          {
            duration: 4000,
            style: {
              background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
              color: '#fff',
              border: '1px solid rgba(34, 197, 94, 0.3)',
              borderRadius: '12px',
            },
          }
        );
        setShowInterventionPanel(true);
      } else {
        toast.info("No interventions needed at this time. Keep up the good work!");
      }
    } catch (error) {
      console.error("Failed to trigger intervention:", error);
      toast.error("Failed to trigger intervention. Please try again.");
    }
  };

  // Cleanup blob URLs to prevent memory leaks
  useEffect(() => {
    return () => {
      if (generatedVideo?.url) {
        URL.revokeObjectURL(generatedVideo.url);
      }
    };
  }, [generatedVideo]);

  // Computed values for loading and error states
  const isLoadingData = isLoadingExisting || isLoadingNew;
  const isErrorData = isErrorExisting || isErrorNew;
  const subjectData = lessonData;

  // Reset results when subject or topic changes
  useEffect(() => {
    setShowResults(false);
    setLessonData(null);
  }, [selectedSubject, topic]);

  // Ensure isSubmitting is properly reset when API request completes
  useEffect(() => {
    if (!isLoadingData && isSubmitting) {
      // Small delay to ensure smooth transition
      const timer = setTimeout(() => {
        setIsSubmitting(false);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isLoadingData, isSubmitting]);

  // Helper function to check if both fields are valid (non-empty after trimming)
  const isFormValid = () => {
    return selectedSubject.trim().length > 0 && topic.trim().length > 0;
  };

  // Helper function to determine if button should be disabled
  const isButtonDisabled = () => {
    return isSubmitting || isLoadingData;
  };

  // Helper function to determine button visual state
  const getButtonVisualState = () => {
    if (isButtonDisabled()) {
      return "disabled"; // Fully disabled during submission/loading
    }
    if (!isFormValid()) {
      return "invalid"; // Visually dimmed but clickable when form is invalid
    }
    return "valid"; // Normal state when form is valid and not submitting
  };

  // Helper function to reset form and return to search
  const handleNewSearch = () => {
    setShowResults(false);
    setSelectedSubject("");
    setTopic("");
    setLessonData(null);
    setIncludeWikipedia(true);
    setUseKnowledgeStore(true);
    setRetryCount(0);
    setExtendedWaitMode(false);
    setFallbackContentAvailable(false);
    setIsOfflineMode(false);
    setLastError(null);
    hideVideo(); // Clear any generated video and hide from sidebar
  };

  // Retry logic with exponential backoff for lesson generation
  const retryLessonGeneration = async (apiCall, maxRetries = 3, baseDelay = 2000) => {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await apiCall();
      } catch (error) {
        setLastError(error.message);

        if (attempt === maxRetries - 1) {
          throw error; // Last attempt failed
        }

        const delay = baseDelay * Math.pow(2, attempt); // Exponential backoff
        console.log(`🔄 Lesson generation retry attempt ${attempt + 1}/${maxRetries} in ${delay}ms`);

        toast.loading(
          `🔄 Retrying lesson generation... (Attempt ${attempt + 2}/${maxRetries})\n⏱️ Please wait ${delay/1000} seconds`,
          {
            id: "lesson-retry-notification",
            duration: delay,
            style: {
              background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
              color: '#fff',
              border: '1px solid rgba(255, 153, 51, 0.3)',
              borderRadius: '12px',
              fontSize: '14px',
              maxWidth: '450px',
              boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
            },
          }
        );

        // After the retry delay, update the main toast to continue showing progress
        setTimeout(() => {
          toast.loading(
            "📚 Continuing lesson generation... Please remain patient.\n🗄️ Knowledge Store processing may take 5-10 minutes.",
            {
              id: "lesson-generation",
              duration: Infinity,
              style: {
                background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
                color: '#fff',
                border: '1px solid rgba(255, 153, 51, 0.3)',
                borderRadius: '12px',
                fontSize: '14px',
                maxWidth: '450px',
                fontWeight: '500',
                boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
              },
            }
          );
        }, delay);

        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  };

  // Check for cached lesson content
  const getCachedLessonContent = (subject, topic) => {
    try {
      const cacheKey = `lesson_${subject.toLowerCase()}_${topic.toLowerCase()}`;
      const cached = localStorage.getItem(cacheKey);
      if (cached) {
        const parsedCache = JSON.parse(cached);
        const cacheAge = Date.now() - parsedCache.timestamp;
        const maxAge = 24 * 60 * 60 * 1000; // 24 hours

        if (cacheAge < maxAge) {
          return parsedCache.data;
        }
      }
    } catch (error) {
      console.error("Error reading cached lesson content:", error);
    }
    return null;
  };

  // Save lesson content to cache
  const cacheLessonContent = (subject, topic, data) => {
    try {
      const cacheKey = `lesson_${subject.toLowerCase()}_${topic.toLowerCase()}`;
      const cacheData = {
        data: data,
        timestamp: Date.now()
      };
      localStorage.setItem(cacheKey, JSON.stringify(cacheData));
    } catch (error) {
      console.error("Error caching lesson content:", error);
    }
  };



  // Send lesson content to generate-video endpoint via proxy
  const sendToVisionAPI = async (subject, topic, lessonData) => {
    try {
      console.log("🎬 Sending lesson content to AnimateDiff video generation API...");

      if (!lessonData?.explanation) {
        console.log("⚠️ No explanation available to send to video generation API");
        return;
      }

      // Enhanced payload structure with AnimateDiff parameters
      const payload = {
        prompt: lessonData.explanation,
        negative_prompt: "blurry, low quality, distorted, text, watermark",
        num_frames: 16,
        guidance_scale: 7.5,
        steps: 25,
        seed: 333,
        fps: 8
      };

      console.log("🎬 AnimateDiff API Payload:", payload);
      console.log("🎬 Target endpoint: http://192.168.0.121:8501/generate-video");
      console.log("🎬 Ngrok endpoint: https://369595c6ef7f.ngrok-free.app/generate-video");

      let response;
      let usingProxy = false;

      // Standard headers with API key for all requests
      const headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': 'shashank_ka_vision786'
      };

      try {
        // First attempt: Use backend proxy (most reliable - avoids CORS)
        console.log("🎬 Attempting backend proxy request via localhost:8001/proxy/vision...");
        const backendProxyResponse = await fetch(`${CHAT_API_BASE_URL}/proxy/vision`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          mode: 'cors',
          credentials: 'same-origin',
          body: JSON.stringify({
            ...payload,
            target_endpoint: "http://192.168.0.121:8501/generate-video"
          })
        });

        // Check if backend proxy request was successful
        if (backendProxyResponse.ok) {
          response = backendProxyResponse;
          usingProxy = true;
          console.log("🎬 Backend proxy request successful");
        } else {
          console.log(`🎬 Backend proxy failed with status ${backendProxyResponse.status}: ${backendProxyResponse.statusText}`);
          throw new Error(`Backend proxy returned ${backendProxyResponse.status}`);
        }
      } catch (backendProxyError) {
        console.log("🎬 Backend proxy failed, trying test proxy...");
        console.log("🎬 Backend proxy error:", backendProxyError.message);

        try {
          // Second attempt: Use backend test proxy
          console.log("🎬 Attempting backend test proxy via localhost:8001/test-generate-video...");
          const testProxyResponse = await fetch(`${CHAT_API_BASE_URL}/test-generate-video`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            mode: 'cors',
            credentials: 'same-origin',
            body: JSON.stringify(payload)
          });

          // Check if test proxy request was successful
          if (testProxyResponse.ok) {
            response = testProxyResponse;
            usingProxy = true;
            console.log("🎬 Backend test proxy request successful");
          } else {
            console.log(`🎬 Backend test proxy failed with status ${testProxyResponse.status}: ${testProxyResponse.statusText}`);
            throw new Error(`Backend test proxy returned ${testProxyResponse.status}`);
          }
        } catch (testProxyError) {
          console.log("🎬 Backend test proxy failed, trying ngrok endpoint...");
          console.log("🎬 Test proxy error:", testProxyError.message);

          try {
            // Third attempt: Use ngrok public endpoint with API key
            console.log("🎬 Attempting ngrok endpoint with API key...");
            const ngrokResponse = await fetch("https://369595c6ef7f.ngrok-free.app/generate-video", {
              method: 'POST',
              headers: {
                ...headers,
                'ngrok-skip-browser-warning': 'true' // Skip ngrok browser warning
              },
              mode: 'cors',
              credentials: 'omit',
              body: JSON.stringify(payload)
            });

            // Check if ngrok request was successful
            if (ngrokResponse.ok) {
              response = ngrokResponse;
              usingProxy = false;
              console.log("🎬 Ngrok endpoint request successful");
            } else {
              console.log(`🎬 Ngrok endpoint failed with status ${ngrokResponse.status}: ${ngrokResponse.statusText}`);
              throw new Error(`Ngrok endpoint returned ${ngrokResponse.status}`);
            }
          } catch (ngrokError) {
            console.log("🎬 All connection methods failed");
            console.log("🎬 Ngrok error:", ngrokError.message);
            throw new Error("All connection methods failed - backend proxy, test proxy, and ngrok all failed");
          }
        }
      }

      if (!response.ok) {
        // Try to get error details from response
        let errorMessage = `AnimateDiff API request failed: ${response.status} ${response.statusText}`;
        const methodUsedForError = usingProxy ? "proxy" : "direct request";

        try {
          const errorData = await response.json();
          if (errorData.error || errorData.message) {
            errorMessage = errorData.error || errorData.message;
          }
        } catch (e) {
          // If we can't parse error response, use the default message
        }

        console.log(`🎬 Final ${methodUsedForError} failed with status ${response.status}`);
        throw new Error(`${errorMessage} (via ${methodUsedForError})`);
      }

      // Handle response from AnimateDiff API
      const contentType = response.headers.get('content-type');
      let result;

      if (contentType && contentType.includes('video/')) {
        // Response is a video file (direct method)
        const videoBlob = await response.blob();
        console.log("🎥 Video blob created:", {
          size: videoBlob.size,
          type: videoBlob.type,
          isValid: videoBlob.size > 0
        });

        const videoUrl = URL.createObjectURL(videoBlob);
        console.log("🎥 Blob URL created:", videoUrl);

        result = {
          success: true,
          video_url: videoUrl,
          content_type: contentType,
          size: videoBlob.size,
          method: "direct"
        };
        console.log("🎬 Video generated successfully (direct):", result);
      } else {
        // Response is JSON (new transfer method)
        result = await response.json();
        console.log("🎬 AnimateDiff API Response:", result);

        // If video was transferred to main system, get it from there
        if (result.success && result.video_id && result.access_url) {
          console.log("🎬 Video transferred to main system, fetching from:", result.access_url);

          try {
            const videoResponse = await fetch(`${API_BASE_URL}${result.access_url}`, {
              method: 'GET',
              headers: {
                'Accept': 'video/mp4',
              },
              mode: 'cors',
              credentials: 'same-origin'
            });

            if (videoResponse.ok) {
              const videoBlob = await videoResponse.blob();
              const videoUrl = URL.createObjectURL(videoBlob);
              result = {
                ...result,
                video_url: videoUrl,
                content_type: videoResponse.headers.get('content-type') || 'video/mp4',
                size: videoBlob.size,
                method: "transferred"
              };
              console.log("🎬 Video fetched from main system successfully:", result);
            } else {
              console.log("⚠️ Failed to fetch video from main system, using fallback");
            }
          } catch (fetchError) {
            console.log("⚠️ Error fetching video from main system:", fetchError);
          }
        }
      }

      // Show success toast with method used and video info
      const methodUsed = usingProxy ? "via backend proxy" : "directly via ngrok";
      const transferMethod = result.method === "transferred" ? " → transferred to main system" : "";
      const videoInfo = result.video_url ? `\n🎥 Video generated (${(result.size / 1024 / 1024).toFixed(2)} MB)` : '';

      toast.success(
        `🎬 AnimateDiff video generation completed ${methodUsed}${transferMethod}!\n📍 Backend Proxy: localhost:8001 → 192.168.0.121:8501\n🌐 Ngrok Fallback: 4e6b01c6e6f2.ngrok-free.app${videoInfo}`,
        {
          duration: 6000,
          style: {
            background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
            color: '#fff',
            border: '1px solid rgba(255, 153, 51, 0.3)',
            borderRadius: '12px',
            fontSize: '14px',
            maxWidth: '500px',
            boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
          },
        }
      );

      // If video was generated, store it for display
      if (result.video_url) {
        console.log("🎥 Generated video URL:", result.video_url);
        console.log("🎥 Video can be displayed using this URL in a <video> element");

        // Store the video data and show in sidebar
        showVideo({
          url: result.video_url,
          contentType: result.content_type,
          size: result.size,
          subject: subject,
          topic: topic,
          timestamp: new Date().toISOString()
        });
      }

      return result;
    } catch (error) {
      console.error("🎬 Error sending to video generation API:", error);

      // Determine error type for better user feedback
      let errorMessage = error.message;
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        errorMessage = "Network connection failed. Please check that your backend (localhost:8001) is running and can reach the AnimateDiff service.";
      } else if (error.message.includes('CORS')) {
        errorMessage = "Cross-origin request blocked. Using backend proxy to resolve the issue.";
      } else if (error.message.includes('404')) {
        errorMessage = "Video generation API endpoint not found. Please verify the AnimateDiff service is running on 192.168.0.121:8501.";
      } else if (error.message.includes('401') || error.message.includes('403')) {
        errorMessage = "Authentication failed. Please verify the API key 'shashank_ka_vision786' is correct.";
      } else if (error.message.includes('500') || error.message.includes('503')) {
        errorMessage = "AnimateDiff service is temporarily unavailable or unreachable. Please try again later.";
      } else if (error.message.includes('504')) {
        errorMessage = "Video generation request timed out. The process may take longer than expected.";
      }

      // Show error toast (non-blocking)
      toast.error(
        `🎬 Failed to connect to AnimateDiff API: ${errorMessage}\n📍 Backend Proxy: localhost:8001 → 192.168.0.121:8501\n🌐 Ngrok Fallback: 2fcd-103-216-69-71.ngrok-free.app`,
        {
          duration: 8000,
          style: {
            background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
            color: '#fff',
            border: '1px solid rgba(255, 153, 51, 0.3)',
            borderRadius: '12px',
            fontSize: '14px',
            maxWidth: '450px',
            boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
          },
        }
      );
    }
  };



  const handleSubmit = async (e) => {
    e.preventDefault();

    // Prevent submission if already processing
    if (isButtonDisabled()) {
      return;
    }

    // Validate form fields and show toast notification if invalid
    if (!isFormValid()) {
      toast.error(
        "Please fill in both Subject and Topic fields before exploring.",
        {
          icon: "⚠️",
          duration: 3000,
          style: {
            background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
            color: '#fff',
            border: '1px solid rgba(255, 153, 51, 0.3)',
            borderRadius: '12px',
            fontSize: '14px',
            maxWidth: '450px',
            boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
          },
        }
      );
      return;
    }

    // Set submitting state immediately to prevent double-submission
    console.log("🚀 Starting lesson generation process...");
    console.log(`📚 Subject: ${selectedSubject.trim()}`);
    console.log(`📖 Topic: ${topic.trim()}`);
    console.log(`🌐 Include Wikipedia: ${includeWikipedia}`);
    console.log(`🗄️ Use Knowledge Store: ${useKnowledgeStore}`);

    setIsSubmitting(true);
    setShowResults(false);
    setLessonData(null);

    // Show initial toast to inform user about expected wait time
    const toastMessage = includeWikipedia && useKnowledgeStore
      ? "📚 Generating lesson with Wikipedia + Knowledge Store content...\n⏱️ This may take 5-10 minutes (processing authentic sources)"
      : includeWikipedia
      ? "🌐 Generating lesson with Wikipedia content...\n⏱️ This may take up to 2 minutes"
      : useKnowledgeStore
      ? "🗄️ Generating lesson with Knowledge Store content...\n⏱️ This may take 5-10 minutes (processing authentic sources)"
      : "📚 Generating basic lesson content...";

    toast.loading(toastMessage, {
      id: "lesson-generation",
      duration: Infinity,
      style: {
        background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
        color: '#fff',
        border: '1px solid rgba(255, 153, 51, 0.3)',
        borderRadius: '12px',
        fontSize: '14px',
        maxWidth: '450px',
        fontWeight: '500',
        boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
      },
    });

    try {
      // Use trimmed values for the API call
      const trimmedSubject = selectedSubject.trim();
      const trimmedTopic = topic.trim();

      console.log("🎯 Calling direct lesson generation API...");
      console.log("API URL:", `${API_BASE_URL}/generate_lesson`);
      console.log("Parameters:", {
        subject: trimmedSubject,
        topic: trimmedTopic,
        include_wikipedia: includeWikipedia,
        use_knowledge_store: useKnowledgeStore,
      });

      // Test direct fetch first
      console.log("🧪 Testing direct fetch...");
      const testUrl = `${API_BASE_URL}/generate_lesson?subject=${encodeURIComponent(trimmedSubject)}&topic=${encodeURIComponent(trimmedTopic)}&include_wikipedia=${includeWikipedia}&use_knowledge_store=${useKnowledgeStore}`;
      console.log("Test URL:", testUrl);

      try {
        const directResponse = await fetch(testUrl, {
          signal: AbortSignal.timeout(600000) // 10 minute timeout for Knowledge Store
        });
        console.log("Direct fetch response:", directResponse);
        const directData = await directResponse.json();
        console.log("Direct fetch data:", directData);

        if (directResponse.ok) {
          console.log("✅ Direct fetch successful, using direct data");
          console.log("📊 Lesson data fields:", Object.keys(directData));
          console.log("📝 Text field:", directData.text ? "Present" : "Missing");
          console.log("📝 Explanation field:", directData.explanation ? "Present" : "Missing");
          setLessonData(directData);
          setShowResults(true);
          setIsSubmitting(false);

          // Trigger TTS for lesson content
          if (serviceHealthy && (directData.explanation || directData.text)) {
            const contentToSpeak = directData.explanation || directData.text;
            console.log("🔊 Jupiter TTS: Triggering auto-play for lesson content");
            handleJupiterResponse(contentToSpeak);
          }

          // Send lesson content to vision API (non-blocking)
          if (directData.text) {
            sendToVisionAPI(trimmedSubject, trimmedTopic, { explanation: directData.text });
          }

          // Dismiss the loading toast and show success
          toast.dismiss("lesson-generation");
          toast.success(
            `🎉 Lesson generated successfully!\n📚 ${trimmedSubject}: ${trimmedTopic}\n✨ Your personalized lesson is ready to explore!`,
            {
              duration: 6000,
              style: {
                background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
                color: '#fff',
                border: '1px solid rgba(255, 153, 51, 0.3)',
                borderRadius: '12px',
                fontSize: '14px',
                maxWidth: '450px',
                fontWeight: '500',
                boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
              },
            }
          );
          return; // Exit early on success
        }
      } catch (directError) {
        console.error("Direct fetch failed:", directError);
      }

      // Fallback to RTK Query if direct fetch fails
      console.log("🔄 Falling back to RTK Query...");
      const response = await generateLesson({
        subject: trimmedSubject,
        topic: trimmedTopic,
        include_wikipedia: includeWikipedia,
        use_knowledge_store: useKnowledgeStore,
      });

      console.log("API Response:", response);
      console.log("Response error:", response.error);
      console.log("Response data:", response.data);

      if (response.error) {
        console.error("API Error details:", response.error);
        throw new Error(response.error?.data?.message || response.error?.message || "Failed to generate lesson");
      }

      if (!response.data) {
        console.error("No data in response:", response);
        throw new Error("No lesson data received from API");
      }

      // Successfully received lesson data
      const lessonData = response.data;
      console.log("Lesson generation completed:", lessonData);

      setLessonData(lessonData);
      setShowResults(true);
      setIsSubmitting(false);

      // Trigger TTS for lesson content
      if (serviceHealthy && (lessonData.explanation || lessonData.text)) {
        const contentToSpeak = lessonData.explanation || lessonData.text;
        console.log("🔊 Jupiter TTS: Triggering auto-play for lesson content");
        handleJupiterResponse(contentToSpeak);
      }

      // Send lesson content to vision API (non-blocking)
      if (lessonData.text) {
        sendToVisionAPI(trimmedSubject, trimmedTopic, { explanation: lessonData.text });
      }

      // Dismiss the loading toast and show success
      toast.dismiss("lesson-generation");
      toast.success(
        `🎉 Lesson generated successfully!\n📚 ${trimmedSubject}: ${trimmedTopic}\n✨ Your personalized lesson is ready to explore!`,
        {
          duration: 6000,
          style: {
            background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
            color: '#fff',
            border: '1px solid rgba(255, 153, 51, 0.3)',
            borderRadius: '12px',
            fontSize: '14px',
            maxWidth: '450px',
            fontWeight: '500',
            boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
          },
        }
      );
    } catch (err) {
      console.error("Error in lesson generation process:", err);
      setLastError(err.message);

      // Dismiss the loading toast first
      toast.dismiss("lesson-generation");
      toast.dismiss("lesson-retry-notification");

      // Enhanced error handling with fallback strategies
      const errorMessage = err.message || 'Unknown error';
      const isNetworkError = errorMessage.includes('network') || errorMessage.includes('fetch') || err.status === 0;
      const isTimeoutError = err.status === "TIMEOUT_ERROR" || errorMessage.includes("timed out");
      const isServerError = err.status >= 500;
      const isContentSourceError = errorMessage.includes("No content sources available") || errorMessage.includes("Ollama not working") || errorMessage.includes("no Wikipedia content found");

      // Handle content source errors (Ollama/Wikipedia issues)
      if (isContentSourceError) {
        toast.error(
          `🤖 AI content generation temporarily unavailable\n📚 This can happen when:\n• AI models are updating\n• Wikipedia content is limited for this topic\n• Network connectivity issues\n\n💡 Try:\n• Refreshing the page\n• Trying a different topic\n• Checking back in a few minutes`,
          {
            duration: 12000,
            style: {
              background: 'rgba(245, 158, 11, 0.9)',
              color: '#fff',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '12px',
              fontSize: '14px',
              maxWidth: '500px',
              fontWeight: '500',
            },
          }
        );

        // Show retry button for content source errors
        setFallbackContentAvailable(true);
      }

      // Check for offline condition
      else if (isNetworkError && !navigator.onLine) {
        setIsOfflineMode(true);

        // Try to load cached content
        const cachedContent = getCachedLessonContent(selectedSubject.trim(), topic.trim());
        if (cachedContent) {
          setLessonData(cachedContent);
          setShowResults(true);
          setFallbackContentAvailable(true);

          // Trigger TTS for cached content
          if (serviceHealthy && (cachedContent.explanation || cachedContent.text)) {
            const contentToSpeak = cachedContent.explanation || cachedContent.text;
            console.log("🔊 Jupiter TTS: Triggering auto-play for cached content");
            handleJupiterResponse(contentToSpeak);
          }

          toast.success(
            `📱 You're offline, but we found cached content!\n📚 Showing previously generated lesson\n🔄 Content will sync when online`,
            {
              duration: 8000,
              style: {
                background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
                color: '#fff',
                border: '1px solid rgba(255, 153, 51, 0.3)',
                borderRadius: '12px',
                fontSize: '14px',
                maxWidth: '450px',
                boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
              },
            }
          );
          return;
        } else {
          toast.error(
            `📱 You appear to be offline and no cached content is available.\n🔄 Please check your connection and try again.\n💾 Future lessons will be cached for offline access.`,
            {
              duration: 10000,
              style: {
                background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
                color: '#fff',
                border: '1px solid rgba(255, 153, 51, 0.3)',
                borderRadius: '12px',
                fontSize: '14px',
                maxWidth: '450px',
                boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
              },
            }
          );
        }
      }

      // Handle timeout with retry option
      else if (isTimeoutError) {
        const currentRetryCount = retryCount + 1;
        setRetryCount(currentRetryCount);

        if (currentRetryCount < maxRetries) {
          toast.error(
            `⏱️ Lesson generation timed out (Attempt ${currentRetryCount}/${maxRetries})\n🔄 Would you like to try again?\n⚡ Click the button below to retry with optimized settings`,
            {
              duration: 10000,
              style: {
                background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
                color: '#fff',
                border: '1px solid rgba(255, 153, 51, 0.3)',
                borderRadius: '12px',
                fontSize: '14px',
                maxWidth: '450px',
                boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
              },
            }
          );

          // Show retry button
          setFallbackContentAvailable(true);
        } else {
          toast.error(
            `⏱️ Lesson generation failed after ${maxRetries} attempts\n🤖 The AI service may be experiencing high load\n🔄 Please try again in a few minutes`,
            {
              duration: 8000,
              style: {
                background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
                color: '#fff',
                border: '1px solid rgba(255, 153, 51, 0.3)',
                borderRadius: '12px',
                fontSize: '14px',
                maxWidth: '450px',
                boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
              },
            }
          );
        }
      }

      // Handle server errors
      else if (isServerError) {
        toast.error(
          `🔧 AI service temporarily unavailable (${err.status})\n⏰ This usually resolves within a few minutes\n🔄 Please try again shortly`,
          {
            duration: 8000,
            style: {
              background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
              color: '#fff',
              border: '1px solid rgba(255, 153, 51, 0.3)',
              borderRadius: '12px',
              fontSize: '14px',
              maxWidth: '450px',
              boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
            },
          }
        );
      }

      // Generic error with helpful suggestions
      else {
        toast.error(
          `⚠️ Lesson generation failed\n🔍 Error: ${errorMessage}\n💡 Try: Check connection, refresh page, or try a simpler topic`,
          {
            duration: 8000,
            style: {
              background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95), rgba(25, 25, 35, 0.95))',
              color: '#fff',
              border: '1px solid rgba(255, 153, 51, 0.3)',
              borderRadius: '12px',
              fontSize: '14px',
              maxWidth: '450px',
              boxShadow: '0 0 20px rgba(255, 153, 51, 0.1)',
            },
          }
        );
      }

      // Show results to display error message and retry options
      setShowResults(true);
    } finally {
      // Always reset submitting state in finally block
      setIsSubmitting(false);
    }
  };

  return (
    <GlassContainer>
      <div className="max-w-7xl mx-auto px-6 py-10">
        {/* Show input form only when no results are displayed */}
        {!showResults && (
          <>
            <div className="text-center mb-8">
              <h2
                className="text-5xl md:text-6xl font-extrabold mb-4 drop-shadow-xl transition-all duration-300 bg-gradient-to-r from-white to-amber-200 bg-clip-text text-transparent"
                style={{
                  fontFamily: "Nunito, sans-serif",
                }}
              >
                Subject Explorer
              </h2>

              {/* Orchestration Status */}
              <div className="flex items-center justify-center space-x-4 mb-6">
                {integrationStatus?.integration_status?.overall_valid && (
                  <div className="flex items-center space-x-2 bg-green-500/20 px-3 py-1 rounded-full border border-green-500/40">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-green-300 text-sm font-medium">AI Enhanced</span>
                  </div>
                )}
              </div>
            </div>

            <p
              className="text-xl md:text-2xl font-medium mb-10 text-center text-white/90"
              style={{
                fontFamily: "Inter, Poppins, sans-serif",
              }}
            >
              Select a subject and enter a topic to begin your learning journey
            </p>

            {/* User Progress Dashboard */}
            {showProgressDashboard && userId && userId !== "guest-user" && (
              <div className="mb-10">
                <UserProgressDashboard
                  userProgress={formatUserProgressData(userProgress)}
                  userAnalytics={userAnalytics}
                  onTriggerIntervention={handleTriggerIntervention}
                  isLoadingProgress={isLoadingProgress}
                  isLoadingAnalytics={isLoadingAnalytics}
                  isTriggeringIntervention={isTriggeringIntervention}
                />
              </div>
            )}
            <form
              onSubmit={handleSubmit}
              noValidate
              className="space-y-8 max-w-3xl mx-auto bg-white/10 p-8 rounded-xl backdrop-blur-md border border-white/20 shadow-lg"
            >
              <div className="relative group">
                <label className={`block mb-3 font-medium text-lg transition-opacity duration-300 ${
                  isButtonDisabled() ? 'text-white/50' : 'text-white/90'
                }`}>
                  Subject:
                </label>
                <div className="relative">
                  <GlassInput
                    type="text"
                    placeholder="Type any subject (e.g. Mathematics, Physics, History)"
                    value={selectedSubject}
                    onChange={(e) => setSelectedSubject(e.target.value)}
                    icon={Book}
                    autoComplete="off"
                    className={`text-lg py-3 ${isButtonDisabled() ? 'opacity-60 cursor-not-allowed' : ''}`}
                    disabled={isButtonDisabled()}
                  />
                </div>
              </div>

              <div>
                <label className={`block mb-3 font-medium text-lg transition-opacity duration-300 ${
                  isButtonDisabled() ? 'text-white/50' : 'text-white/90'
                }`}>
                  Topic:
                </label>
                <GlassInput
                  type="text"
                  placeholder="Enter a topic to explore"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  icon={BookOpen}
                  className={`text-lg py-3 ${isButtonDisabled() ? 'opacity-60 cursor-not-allowed' : ''}`}
                  disabled={isButtonDisabled()}
                />
              </div>

              {/* Toggle Switches */}
              <div className={`space-y-6 transition-opacity duration-300 ${
                isButtonDisabled() ? 'opacity-60' : ''
              }`}>
                {/* First Row - Include Wikipedia and Use Knowledge Store */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Include Wikipedia Toggle */}
                  <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/20">
                    <div>
                      <label className={`font-medium text-lg block transition-opacity duration-300 ${
                        isButtonDisabled() ? 'text-white/50' : 'text-white/90'
                      }`}>
                        Include Wikipedia
                      </label>
                      <p className={`text-sm mt-1 transition-opacity duration-300 ${
                        isButtonDisabled() ? 'text-white/40' : 'text-white/60'
                      }`}>
                        Use Wikipedia data for enhanced content
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => setIncludeWikipedia(!includeWikipedia)}
                      disabled={isButtonDisabled()}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-offset-2 focus:ring-offset-gray-900 ${
                        includeWikipedia ? "bg-amber-500" : "bg-gray-600"
                      } ${
                        isButtonDisabled()
                          ? "opacity-50 cursor-not-allowed"
                          : "cursor-pointer"
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ease-in-out ${
                          includeWikipedia ? "translate-x-6" : "translate-x-1"
                        }`}
                      />
                    </button>
                  </div>

                  {/* Use Knowledge Store Toggle */}
                  <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/20">
                    <div>
                      <label className={`font-medium text-lg block transition-opacity duration-300 ${
                        isButtonDisabled() ? 'text-white/50' : 'text-white/90'
                      }`}>
                        Use Knowledge Store
                      </label>
                      <p className={`text-sm mt-1 transition-opacity duration-300 ${
                        isButtonDisabled() ? 'text-white/40' : 'text-white/60'
                      }`}>
                        Access curated knowledge database
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => setUseKnowledgeStore(!useKnowledgeStore)}
                      disabled={isButtonDisabled()}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-offset-2 focus:ring-offset-gray-900 ${
                        useKnowledgeStore ? "bg-amber-500" : "bg-gray-600"
                      } ${
                        isButtonDisabled()
                          ? "opacity-50 cursor-not-allowed"
                          : "cursor-pointer"
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ease-in-out ${
                          useKnowledgeStore ? "translate-x-6" : "translate-x-1"
                        }`}
                      />
                    </button>
                  </div>
                </div>

                {/* Second Row - View Progress Button and Enhanced Mode Toggle */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* View Progress Button */}
                  {userId && userId !== "guest-user" && (
                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/20">
                      <div>
                        <label className={`font-medium text-lg block transition-opacity duration-300 ${
                          isButtonDisabled() ? 'text-white/50' : 'text-white/90'
                        }`}>
                          View Progress
                        </label>
                        <p className={`text-sm mt-1 transition-opacity duration-300 ${
                          isButtonDisabled() ? 'text-white/40' : 'text-white/60'
                        }`}>
                          Show learning progress dashboard
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => setShowProgressDashboard(!showProgressDashboard)}
                        disabled={isButtonDisabled()}
                        className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-offset-2 focus:ring-offset-gray-900 ${
                          showProgressDashboard
                            ? 'bg-amber-500/20 hover:bg-amber-500/30 border-amber-500/40 text-amber-300 hover:text-amber-200'
                            : 'bg-gray-500/20 hover:bg-amber-500/20 border-gray-500/40 hover:border-amber-500/40 text-gray-300 hover:text-amber-300'
                        } ${
                          isButtonDisabled()
                            ? "opacity-50 cursor-not-allowed"
                            : "cursor-pointer"
                        }`}
                      >
                        <BarChart3 className="w-4 h-4" />
                        <span className="text-sm font-medium">
                          {showProgressDashboard ? 'Hide' : 'Show'}
                        </span>
                      </button>
                    </div>
                  )}

                  {/* Enhanced Mode Toggle */}
                  <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/20">
                    <div>
                      <label className={`font-medium text-lg block transition-opacity duration-300 ${
                        isButtonDisabled() ? 'text-white/50' : 'text-white/90'
                      }`}>
                        Enhanced Mode
                      </label>
                      <p className={`text-sm mt-1 transition-opacity duration-300 ${
                        isButtonDisabled() ? 'text-white/40' : 'text-white/60'
                      }`}>
                        Use AI orchestration for personalized learning
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => setUseOrchestration(!useOrchestration)}
                      disabled={isButtonDisabled()}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-offset-2 focus:ring-offset-gray-900 ${
                        useOrchestration ? "bg-amber-500" : "bg-gray-600"
                      } ${
                        isButtonDisabled()
                          ? "opacity-50 cursor-not-allowed"
                          : "cursor-pointer"
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ease-in-out ${
                          useOrchestration ? "translate-x-6" : "translate-x-1"
                        }`}
                      />
                    </button>
                  </div>
                </div>
              </div>

              <div className="flex justify-center mt-10 relative">
                <div
                  className="relative"
                  onMouseEnter={() => setIsButtonHovered(true)}
                  onMouseLeave={() => setIsButtonHovered(false)}
                >
                  <GlassButton
                    type="submit"
                    icon={
                      isButtonDisabled()
                        ? null // No icon when loading, we'll show spinner instead
                        : isButtonHovered && !isFormValid() && !isButtonDisabled()
                        ? AlertCircle
                        : Search
                    }
                    variant="primary"
                    className={`px-10 py-4 text-lg font-medium transition-all duration-300 flex items-center justify-center ${
                      getButtonVisualState() === "invalid" ? "opacity-75" : ""
                    } ${
                      getButtonVisualState() === "disabled"
                        ? "cursor-not-allowed"
                        : ""
                    }`}
                    disabled={isButtonDisabled()}
                    aria-label={
                      isButtonDisabled()
                        ? "Generating lesson content, please wait approximately 2 minutes..."
                        : !isFormValid()
                        ? "Please fill in both Subject and Topic fields to explore"
                        : "Explore the selected topic"
                    }
                  >
                    {isButtonDisabled() ? "Generating Lesson..." : "Explore Topic"}
                  </GlassButton>
                </div>
              </div>
            </form>
          </>
        )}

        {/* Results Section */}
        {showResults && (
          <div className="bg-white/20 rounded-xl p-8 backdrop-blur-md border border-white/30 shadow-xl">

            {isLoadingData || isSubmitting ? (
              <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
                <div className="bg-white/10 rounded-xl p-8 backdrop-blur-md border border-white/30 shadow-xl w-full max-w-md">
                  {/* Orange Loader - Perfectly Centered */}
                  <div className="flex justify-center items-center mb-6">
                    <div className="animate-spin rounded-full h-20 w-20 border-4 border-orange-500/30 border-t-orange-500 border-r-orange-500"></div>
                  </div>

                  <div className="text-center">
                    <p className="text-white/90 text-xl font-medium mb-3">
                      📚 Generating lesson content for "{selectedSubject.trim()}: {topic.trim()}"
                    </p>
                    <p className="text-amber-300 text-lg mb-4">
                      ⏱️ This will take approximately 2 minutes. Please be patient.
                    </p>
                    <div className="bg-gradient-to-r from-orange-900/20 to-amber-900/20 rounded-lg p-4 border border-orange-500/30">
                      <p className="text-orange-200 text-sm">
                        🤖 AI is analyzing your topic and creating personalized content
                        <br />
                        ✨ Including explanations, activities, and questions
                        <br />
                        📖 Integrating knowledge from multiple sources
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ) : isErrorData ? (
              <div className="text-red-400 text-center my-8 p-6 bg-white/5 rounded-xl border border-red-500/20">
                <p className="font-semibold mb-4 text-xl">
                  Sorry, the service is currently unavailable. Please try again
                  later.
                </p>
                <div className="mt-8">
                  <GlassButton
                    onClick={handleNewSearch}
                    variant="secondary"
                    className="px-6 py-3 text-lg"
                  >
                    Try Again
                  </GlassButton>
                </div>
              </div>
            ) : subjectData ? (
              <div className="space-y-8">

                {/* Structured lesson content */}
                <div className="space-y-10">
                  {/* Title */}
                  {subjectData?.title && (
                    <div className="text-center">
                      <div className="mb-4">
                        <h2 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-amber-200">
                          {subjectData.title}
                        </h2>
                      </div>
                      <div className="mt-3 flex justify-center">
                        <div className="bg-gradient-to-r from-amber-500/30 to-amber-600/30 text-amber-200 px-4 py-1.5 rounded-full text-base font-medium border border-amber-500/30 shadow-lg">
                          {selectedSubject.trim()}: {topic.trim()}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Shloka and Translation */}
                  {subjectData?.shloka && (
                    <div className="bg-amber-900/20 p-8 rounded-xl border border-amber-500/40 shadow-lg mx-auto max-w-4xl">
                      <div className="mb-5">
                        <p className="italic text-amber-200 font-medium text-center text-xl leading-relaxed">
                          {subjectData.shloka}
                        </p>
                      </div>
                      {subjectData?.translation && (
                        <div className="bg-white/10 p-4 rounded-lg">
                          <p className="text-white/90 text-base text-center">
                            <span className="text-amber-300 font-medium">
                              Translation:
                            </span>{" "}
                            {subjectData.translation}
                          </p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Mode and Enhancement Indicators */}
                  {(() => {
                    const formattedLesson = formatEnhancedLessonData(subjectData);
                    const isEnhanced = formattedLesson?.isEnhanced;
                    const contentLength = subjectData?.content?.length || 0;
                    const isLongContent = contentLength > 500;
                    const contentType = subjectData?.content_type || 'standard';

                    return (
                      <div className="space-y-3">
                        {/* Mode Indicator */}
                        <div className={`p-3 rounded-xl border shadow-lg ${
                          isEnhanced
                            ? 'bg-gradient-to-r from-green-900/30 to-emerald-900/30 border-green-500/40'
                            : 'bg-gradient-to-r from-blue-900/30 to-indigo-900/30 border-blue-500/40'
                        }`}>
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                              <div className={`w-3 h-3 rounded-full animate-pulse ${
                                isEnhanced ? 'bg-green-500' : 'bg-blue-500'
                              }`}></div>
                              <span className={`font-medium ${
                                isEnhanced ? 'text-green-300' : 'text-blue-300'
                              }`}>
                                {isEnhanced ? '🚀 Enhanced Mode' : '📚 Basic Mode'}
                              </span>
                            </div>
                            <div className="flex items-center space-x-3 text-sm">
                              {/* Content Length Indicator */}
                              <span className={`px-2 py-1 rounded-full ${
                                isLongContent
                                  ? 'bg-purple-500/20 text-purple-200'
                                  : 'bg-gray-500/20 text-gray-200'
                              }`}>
                                📝 {isLongContent ? 'Comprehensive' : 'Concise'} ({contentLength} chars)
                              </span>

                              {/* Wikipedia Indicator */}
                              <span className={`px-2 py-1 rounded-full ${
                                includeWikipedia
                                  ? 'bg-orange-500/20 text-orange-200'
                                  : 'bg-gray-500/20 text-gray-200'
                              }`}>
                                {includeWikipedia ? '🌐 Wikipedia' : '🧠 Pure AI'}
                              </span>

                              {/* Content Type */}
                              {contentType !== 'standard' && (
                                <span className="bg-indigo-500/20 text-indigo-200 px-2 py-1 rounded-full">
                                  {contentType === 'concise' ? '⚡ Concise' : '📖 Basic'}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Enhanced Features (only for orchestration) */}
                        {isEnhanced && (
                          <div className="bg-gradient-to-r from-amber-900/20 to-yellow-900/20 p-3 rounded-xl border border-amber-500/30">
                            <div className="flex items-center space-x-4 text-sm text-amber-200">
                              {formattedLesson.ragEnhanced && (
                                <span className="bg-green-500/20 px-2 py-1 rounded-full">📚 RAG Enhanced</span>
                              )}
                              {formattedLesson.triggersDetected > 0 && (
                                <span className="bg-amber-500/20 px-2 py-1 rounded-full">
                                  ⚡ {formattedLesson.triggersDetected} Triggers
                                </span>
                              )}
                              {formattedLesson.sourceDocumentsCount > 0 && (
                                <span className="bg-blue-500/20 px-2 py-1 rounded-full">
                                  📖 {formattedLesson.sourceDocumentsCount} Sources
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })()}

                  {/* Explanation/Content */}
                  {(subjectData?.explanation || subjectData?.text) && (
                    <div className="bg-white/10 p-6 rounded-xl shadow-lg border-l-4 border-amber-500">
                      <div className="mb-4">
                        <h3 className="text-2xl font-semibold text-white flex items-center">
                          <span className="bg-gradient-to-br from-amber-500 to-amber-600 w-10 h-10 rounded-full flex items-center justify-center mr-3 text-white font-bold shadow-md">
                            1
                          </span>
                          Lesson Content
                        </h3>
                      </div>
                      <div className="text-white/90 leading-relaxed text-lg whitespace-pre-wrap">
                        {subjectData.explanation || subjectData.text}
                      </div>
                    </div>
                  )}

                  {/* Activity */}
                  {subjectData?.activity && (
                    <div className="bg-indigo-900/30 p-6 rounded-xl shadow-lg border-l-4 border-indigo-500">
                      <div className="mb-4">
                        <h3 className="text-2xl font-semibold text-white flex items-center">
                          <span className="bg-gradient-to-br from-indigo-500 to-indigo-600 w-10 h-10 rounded-full flex items-center justify-center mr-3 text-white font-bold shadow-md">
                            2
                          </span>
                          Activity
                        </h3>
                      </div>
                      <p className="text-white/90 leading-relaxed text-lg">
                        {subjectData.activity}
                      </p>
                    </div>
                  )}

                  {/* Question */}
                  {subjectData?.question && (
                    <div className="bg-amber-900/30 p-6 rounded-xl shadow-lg border-l-4 border-amber-500">
                      <div className="mb-4">
                        <h3 className="text-2xl font-semibold text-white flex items-center">
                          <span className="bg-gradient-to-br from-amber-500 to-amber-600 w-10 h-10 rounded-full flex items-center justify-center mr-3 text-white font-bold shadow-md">
                            3
                          </span>
                          Question to Consider
                        </h3>
                      </div>
                      <p className="text-white/90 leading-relaxed text-lg">
                        {subjectData.question}
                      </p>
                    </div>
                  )}

                  {/* Fallback for legacy content format */}
                  {(subjectData?.lesson || subjectData?.content) &&
                    !subjectData?.title && (
                      <div
                        className="bg-white/10 p-6 rounded-xl shadow-lg border border-white/30"
                        dangerouslySetInnerHTML={{
                          __html: subjectData?.lesson || subjectData?.content,
                        }}
                      />
                    )}

                  {/* Emergency fallback - display raw data if nothing else shows */}
                  {!subjectData?.title && !subjectData?.lesson && !subjectData?.content && (
                    <div className="bg-white/10 p-6 rounded-xl shadow-lg border border-white/30">
                      <h3 className="text-xl font-semibold text-white mb-4">Generated Lesson Content:</h3>
                      <pre className="text-white/90 whitespace-pre-wrap text-sm overflow-auto max-h-96">
                        {JSON.stringify(subjectData, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>

                {/* Action buttons */}
                <div className="flex justify-center items-center gap-6 mt-12">
                  {/* New Search Button */}
                  <GlassButton
                    onClick={handleNewSearch}
                    variant="secondary"
                    className="px-6 py-3 text-lg font-medium shadow-lg hover:shadow-xl transition-all"
                  >
                    New Search
                  </GlassButton>
                </div>
              </div>
            ) : (
              <div className="text-center p-8 bg-white/5 rounded-xl border border-white/20">
                <p className="text-white/80 mb-6 text-lg">
                  No data available. Please try a different topic.
                </p>
                <GlassButton
                  onClick={handleNewSearch}
                  variant="secondary"
                  className="px-6 py-3 text-lg font-medium shadow-lg"
                >
                  Try Again
                </GlassButton>
              </div>
            )}
          </div>
        )}

        {/* Original Video Player Section - Only show when not in sidebar mode */}
        {generatedVideo && !showVideoInSidebar && (
          <div className="mt-8 bg-white/20 rounded-xl p-8 backdrop-blur-md border border-white/30 shadow-xl">
            <div className="text-center mb-6">
              <h3 className="text-3xl font-bold text-white mb-2">
                🎬 Generated Video
              </h3>
              <p className="text-amber-300 text-lg">
                📚 {generatedVideo.subject}: {generatedVideo.topic}
              </p>
              <p className="text-white/70 text-sm mt-2">
                🎥 Size: {(generatedVideo.size / 1024 / 1024).toFixed(2)} MB •
                📅 Generated: {new Date(generatedVideo.timestamp).toLocaleString()}
              </p>
            </div>

            <div className="flex justify-center">
              <div className="bg-black/50 rounded-xl p-4 backdrop-blur-sm border border-orange-500/30 shadow-2xl">
                <video
                  src={generatedVideo.url}
                  controls
                  autoPlay={false}
                  loop
                  muted
                  className="max-w-full max-h-96 rounded-lg shadow-lg"
                  style={{ maxWidth: '600px', width: '100%' }}
                  onLoadStart={() => console.log("🎥 Video load started")}
                  onLoadedData={() => console.log("🎥 Video data loaded")}
                  onCanPlay={() => console.log("🎥 Video can play")}
                  onError={(e) => {
                    console.error("🎥 Video error:", e);
                    console.error("🎥 Video error details:", e.target.error);
                    console.error("🎥 Video src:", generatedVideo.url);
                  }}
                >
                  <p className="text-white">
                    Your browser doesn't support video playback.
                    <a
                      href={generatedVideo.url}
                      download="generated_video.mp4"
                      className="text-orange-400 hover:text-orange-300 underline ml-1"
                    >
                      Download the video instead
                    </a>
                  </p>
                </video>

                {/* Debug info */}
                <div className="mt-4 p-3 bg-gray-800/50 rounded-lg text-xs text-white/70">
                  <p><strong>Debug Info:</strong></p>
                  <p>URL: {generatedVideo.url}</p>
                  <p>Content Type: {generatedVideo.contentType}</p>
                  <p>Size: {generatedVideo.size} bytes</p>
                  <p>URL Valid: {generatedVideo.url ? 'Yes' : 'No'}</p>
                  <p>URL Type: {generatedVideo.url?.startsWith('blob:') ? 'Blob URL' : 'Regular URL'}</p>
                </div>
              </div>
            </div>

            <div className="flex justify-center gap-4 mt-6">
              <a
                href={generatedVideo.url}
                download={`${generatedVideo.subject}_${generatedVideo.topic}_video.mp4`}
                className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-orange-600 to-amber-600 text-white font-medium rounded-lg hover:from-orange-700 hover:to-amber-700 transition-all duration-300 shadow-lg hover:shadow-xl"
              >
                📥 Download Video
              </a>

              <button
                onClick={() => {
                  console.log("🎥 Testing video URL:", generatedVideo.url);
                  const video = document.querySelector('video');
                  if (video) {
                    console.log("🎥 Video element found:", video);
                    console.log("🎥 Video src:", video.src);
                    console.log("🎥 Video readyState:", video.readyState);
                    console.log("🎥 Video networkState:", video.networkState);
                    video.load(); // Force reload
                  }
                }}
                className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-medium rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-300 shadow-lg hover:shadow-xl"
              >
                🔄 Test Video
              </button>

              <button
                onClick={hideVideo}
                className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white font-medium rounded-lg hover:from-gray-700 hover:to-gray-800 transition-all duration-300 shadow-lg hover:shadow-xl"
              >
                ✕ Close Video
              </button>
            </div>
          </div>
        )}
      </div>
    </GlassContainer>
  );
}
