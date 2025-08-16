import React, { useState, useEffect } from "react";
import GlassContainer from "../components/GlassContainer";
import CenteredLoader from "../components/CenteredLoader";
import GlassInput from "../components/GlassInput";
import GlassButton from "../components/GlassButton";
import {
  useLazyGetLessonDataQuery,
  useCreateLessonMutation,
} from "../api/subjectsApiSlice";
import { Book, BookOpen, Search, AlertCircle } from "lucide-react";
import { useSelector } from "react-redux";
import { selectUserId } from "../store/authSlice";
import { toast } from "react-hot-toast";
import { TTS_SERVICE_URL } from "../config";
import TTSButton from "../components/TTSButton";
import { useGenerateTTSMutation } from "../api/ttsApiSlice";
import { ttsUtils } from "../api/ttsApiSlice";
import { Volume2 } from "lucide-react";

export default function Subjects() {
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

  // TTS hook for complete lesson
  const [generateTTS, { isLoading: isGeneratingTTS }] = useGenerateTTSMutation();

  // Component state
  const [selectedSubject, setSelectedSubject] = useState("");
  const [topic, setTopic] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [isButtonHovered, setIsButtonHovered] = useState(false);
  const [lessonData, setLessonData] = useState(null);
  const [includeWikipedia, setIncludeWikipedia] = useState(true);
  const [useKnowledgeStore, setUseKnowledgeStore] = useState(true);

  // Edge case handling states
  const [retryCount, setRetryCount] = useState(0);
  const [extendedWaitMode, setExtendedWaitMode] = useState(false);
  const [fallbackContentAvailable, setFallbackContentAvailable] = useState(false);
  const [isOfflineMode, setIsOfflineMode] = useState(false);
  const [lastError, setLastError] = useState(null);
  const maxRetries = 3;
  const extendedWaitThreshold = 300; // 5 minutes

  const userId = useSelector(selectUserId) || "guest-user";

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
              background: 'rgba(245, 158, 11, 0.9)',
              color: '#fff',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '12px',
              fontSize: '14px',
              maxWidth: '450px',
            },
          }
        );

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

  // Handle complete lesson TTS
  const handleCompleteLessonTTS = async () => {
    if (!subjectData) return;

    try {
      toast.loading("🎵 Generating complete lesson audio...", {
        id: "complete-lesson-tts",
        duration: 5000,
      });

      const result = await ttsUtils.generateCompleteLessonTTS(generateTTS, subjectData);

      toast.dismiss("complete-lesson-tts");
      toast.success("🎵 Complete lesson audio ready! Playing...", {
        duration: 3000,
      });

      // Play the complete lesson audio
      await ttsUtils.playAudio(result.full_audio_url);

    } catch (error) {
      console.error("Complete lesson TTS error:", error);
      toast.dismiss("complete-lesson-tts");
      toast.error("❌ Failed to generate complete lesson audio", {
        duration: 3000,
      });
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
        }
      );
      return;
    }

    // Set submitting state immediately to prevent double-submission
    console.log("🚀 Starting lesson generation process...");
    console.log(`📚 Subject: ${selectedSubject.trim()}`);
    console.log(`📖 Topic: ${topic.trim()}`);
    console.log(`⏱️ Expected duration: ~2 minutes`);

    setIsSubmitting(true);
    setShowResults(false);
    setLessonData(null);

    // Show initial toast to inform user about expected wait time
    toast.loading(
      "📚 Generating lesson content. This will take approximately 2 minutes. Please be patient.",
      {
        id: "lesson-generation", // Use ID to update this toast later
        duration: 8000, // Show for 8 seconds to ensure user sees it
        style: {
          background: 'rgba(59, 130, 246, 0.9)',
          color: '#fff',
          border: '1px solid rgba(59, 130, 246, 0.3)',
          borderRadius: '12px',
          fontSize: '14px',
          maxWidth: '450px',
          fontWeight: '500',
        },
      }
    );

    try {
      // Use trimmed values for the API call
      const trimmedSubject = selectedSubject.trim();
      const trimmedTopic = topic.trim();

      // Step 1: Create lesson with POST request
      console.log("Step 1: Creating lesson...");
      console.log("POST URL:", `${TTS_SERVICE_URL}/lessons`);
      console.log("POST Body:", {
        subject: trimmedSubject,
        topic: trimmedTopic,
        user_id: userId,
        include_wikipedia: includeWikipedia,
        force_regenerate: true,
      });

      const createResponse = await createLesson({
        subject: trimmedSubject,
        topic: trimmedTopic,
        user_id: userId,
        include_wikipedia: includeWikipedia,
        force_regenerate: true,
      });

      console.log("POST Response:", createResponse);

      if (createResponse.error) {
        throw (
          createResponse.error ||
          new Error("Failed to initiate lesson creation")
        );
      }

      // Step 2: Get task_id from the response and poll for completion
      const taskId = createResponse.data?.task_id;
      if (!taskId) {
        throw new Error("No task ID received from lesson creation");
      }

      console.log("Step 2: Polling for lesson completion using task ID:", taskId);
      const maxAttempts = 60; // 60 attempts with 2 second intervals = 2 minutes timeout
      let attempts = 0;
      let lessonFound = false;

      while (attempts < maxAttempts && !lessonFound) {
        attempts++;
        console.log(`Polling attempt ${attempts}/${maxAttempts} for task ${taskId}...`);

        // Wait 2 seconds between attempts (lesson generation takes time)
        if (attempts > 1) {
          await new Promise((resolve) => setTimeout(resolve, 2000));
        }

        try {
          // Check task status using the status endpoint
          const statusResponse = await fetch(`${TTS_SERVICE_URL}/lessons/status/${taskId}`);
          const statusData = await statusResponse.json();

          console.log(`Status Response (attempt ${attempts}):`, statusData);

          if (statusData.status === 'completed' && statusData.lesson_data) {
            // Successfully retrieved lesson data
            console.log("Lesson generation completed:", statusData.lesson_data);
            setLessonData(statusData.lesson_data);
            setShowResults(true);
            setIsSubmitting(false); // Explicitly set isSubmitting to false here
            lessonFound = true;

            // Cache the lesson content
            cacheLessonContent(trimmedSubject, trimmedTopic, statusData.lesson_data);

            // Dismiss the loading toast and show success
            toast.dismiss("lesson-generation");
            toast.success(
              `🎉 Lesson generated successfully!\n📚 ${trimmedSubject}: ${trimmedTopic}\n✨ Your personalized lesson is ready to explore!`,
              {
                duration: 6000,
                style: {
                  background: 'rgba(34, 197, 94, 0.9)',
                  color: '#fff',
                  border: '1px solid rgba(34, 197, 94, 0.3)',
                  borderRadius: '12px',
                  fontSize: '14px',
                  maxWidth: '450px',
                  fontWeight: '500',
                },
              }
            );
          } else if (statusData.status === 'failed') {
            throw new Error(statusData.error_message || "Lesson generation failed");
          } else if (statusData.status === 'in_progress') {
            console.log("Lesson generation in progress...");
            // Update toast with progress info every 10 attempts (20 seconds)
            if (attempts % 10 === 0) {
              const elapsedMinutes = Math.floor(attempts * 2 / 60);
              const elapsedSeconds = attempts * 2 % 60;
              const progressPercentage = Math.min(Math.round((attempts / maxAttempts) * 100), 95);

              toast.loading(
                `📚 Generating lesson content...\n⏱️ ${elapsedMinutes}:${String(elapsedSeconds).padStart(2, '0')} elapsed • ${progressPercentage}% complete\n🤖 AI is analyzing and creating your personalized lesson`,
                {
                  id: "lesson-generation",
                  style: {
                    background: 'rgba(59, 130, 246, 0.9)',
                    color: '#fff',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    borderRadius: '12px',
                    fontSize: '14px',
                    maxWidth: '450px',
                    fontWeight: '500',
                  },
                }
              );
            }
            // Continue polling
          } else if (statusData.status === 'pending') {
            console.log("Lesson generation pending...");
            // Continue polling
          }
        } catch (pollError) {
          console.log(`Polling attempt ${attempts} failed:`, pollError);
          // Continue polling unless it's the last attempt
          if (attempts === maxAttempts) {
            throw pollError;
          }
        }
      }

      if (!lessonFound) {
        // Show timeout warning
        toast.dismiss("lesson-generation");
        throw new Error("Lesson generation timed out after 2 minutes");
      }
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

          toast.success(
            `📱 You're offline, but we found cached content!\n📚 Showing previously generated lesson\n🔄 Content will sync when online`,
            {
              duration: 8000,
              style: {
                background: 'rgba(34, 197, 94, 0.9)',
                color: '#fff',
                border: '1px solid rgba(34, 197, 94, 0.3)',
                borderRadius: '12px',
                fontSize: '14px',
                maxWidth: '450px',
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
                background: 'rgba(239, 68, 68, 0.9)',
                color: '#fff',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '12px',
                fontSize: '14px',
                maxWidth: '450px',
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
                background: 'rgba(239, 68, 68, 0.9)',
                color: '#fff',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '12px',
                fontSize: '14px',
                maxWidth: '450px',
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
                background: 'rgba(239, 68, 68, 0.9)',
                color: '#fff',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '12px',
                fontSize: '14px',
                maxWidth: '450px',
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
              background: 'rgba(239, 68, 68, 0.9)',
              color: '#fff',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '12px',
              fontSize: '14px',
              maxWidth: '450px',
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
              background: 'rgba(239, 68, 68, 0.9)',
              color: '#fff',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '12px',
              fontSize: '14px',
              maxWidth: '450px',
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
            <h2
              className="text-5xl md:text-6xl font-extrabold mb-8 drop-shadow-xl transition-all duration-300 bg-gradient-to-r from-white to-amber-200 bg-clip-text text-transparent text-center"
              style={{
                fontFamily: "Nunito, sans-serif",
              }}
            >
              Subject Explorer
            </h2>
            <p
              className="text-xl md:text-2xl font-medium mb-10 text-center text-white/90"
              style={{
                fontFamily: "Nunito, sans-serif",
              }}
            >
              Select a subject and enter a topic to begin your learning journey
            </p>
            <form
              onSubmit={handleSubmit}
              noValidate
              className="space-y-8 max-w-3xl mx-auto bg-white/10 p-8 rounded-xl backdrop-blur-md border border-white/20 shadow-lg"
            >
              <div className="relative group">
                <label className="text-white/90 block mb-3 font-medium text-lg">
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
                    className="text-lg py-3"
                    disabled={isButtonDisabled()}
                  />
                </div>
              </div>

              <div>
                <label className="text-white/90 block mb-3 font-medium text-lg">
                  Topic:
                </label>
                <GlassInput
                  type="text"
                  placeholder="Enter a topic to explore"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  icon={BookOpen}
                  className="text-lg py-3"
                  disabled={isButtonDisabled()}
                />
              </div>

              {/* Toggle Switches */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Include Wikipedia Toggle */}
                <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/20">
                  <div>
                    <label className="text-white/90 font-medium text-lg block">
                      Include Wikipedia
                    </label>
                    <p className="text-white/60 text-sm mt-1">
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
                    <label className="text-white/90 font-medium text-lg block">
                      Use Knowledge Store
                    </label>
                    <p className="text-white/60 text-sm mt-1">
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
                    {isButtonDisabled() ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                        Generating Lesson...
                      </>
                    ) : (
                      "Explore Topic"
                    )}
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
              <div className="flex flex-col items-center justify-center my-12">
                <CenteredLoader />
                <div className="mt-6 text-center">
                  <p className="text-white/90 text-xl font-medium mb-3">
                    📚 Generating lesson content for "{selectedSubject.trim()}: {topic.trim()}"
                  </p>
                  <p className="text-amber-300 text-lg mb-4">
                    ⏱️ This will take approximately 2 minutes. Please be patient.
                  </p>
                  <div className="bg-white/10 rounded-lg p-4 max-w-md mx-auto">
                    <p className="text-white/70 text-sm">
                      🤖 AI is analyzing your topic and creating personalized content
                      <br />
                      ✨ Including explanations, activities, and questions
                      <br />
                      📖 Integrating knowledge from multiple sources
                    </p>
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
                      <div className="flex items-center justify-center gap-4 mb-4">
                        <h2 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-amber-200">
                          {subjectData.title}
                        </h2>
                        <TTSButton
                          text={subjectData.title}
                          section="title"
                          variant="accent"
                          size="lg"
                        />
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
                      <div className="flex items-start justify-between mb-5">
                        <p className="italic text-amber-200 font-medium text-center text-xl leading-relaxed flex-1">
                          {subjectData.shloka}
                        </p>
                        <TTSButton
                          text={subjectData.shloka}
                          section="shloka"
                          variant="accent"
                          className="ml-4 flex-shrink-0"
                        />
                      </div>
                      {subjectData?.translation && (
                        <div className="bg-white/10 p-4 rounded-lg">
                          <div className="flex items-start justify-between">
                            <p className="text-white/90 text-base text-center flex-1">
                              <span className="text-amber-300 font-medium">
                                Translation:
                              </span>{" "}
                              {subjectData.translation}
                            </p>
                            <TTSButton
                              text={`Translation: ${subjectData.translation}`}
                              section="translation"
                              variant="secondary"
                              size="sm"
                              className="ml-4 flex-shrink-0"
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Explanation */}
                  {subjectData?.explanation && (
                    <div className="bg-white/10 p-6 rounded-xl shadow-lg border-l-4 border-amber-500">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-2xl font-semibold text-white flex items-center">
                          <span className="bg-gradient-to-br from-amber-500 to-amber-600 w-10 h-10 rounded-full flex items-center justify-center mr-3 text-white font-bold shadow-md">
                            1
                          </span>
                          Explanation
                        </h3>
                        <TTSButton
                          text={subjectData.explanation}
                          section="explanation"
                          variant="primary"
                        />
                      </div>
                      <p className="text-white/90 leading-relaxed text-lg">
                        {subjectData.explanation}
                      </p>
                    </div>
                  )}

                  {/* Activity */}
                  {subjectData?.activity && (
                    <div className="bg-indigo-900/30 p-6 rounded-xl shadow-lg border-l-4 border-indigo-500">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-2xl font-semibold text-white flex items-center">
                          <span className="bg-gradient-to-br from-indigo-500 to-indigo-600 w-10 h-10 rounded-full flex items-center justify-center mr-3 text-white font-bold shadow-md">
                            2
                          </span>
                          Activity
                        </h3>
                        <TTSButton
                          text={subjectData.activity}
                          section="activity"
                          variant="primary"
                        />
                      </div>
                      <p className="text-white/90 leading-relaxed text-lg">
                        {subjectData.activity}
                      </p>
                    </div>
                  )}

                  {/* Question */}
                  {subjectData?.question && (
                    <div className="bg-amber-900/30 p-6 rounded-xl shadow-lg border-l-4 border-amber-500">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-2xl font-semibold text-white flex items-center">
                          <span className="bg-gradient-to-br from-amber-500 to-amber-600 w-10 h-10 rounded-full flex items-center justify-center mr-3 text-white font-bold shadow-md">
                            3
                          </span>
                          Question to Consider
                        </h3>
                        <TTSButton
                          text={subjectData.question}
                          section="question"
                          variant="accent"
                        />
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
                  {/* Complete Lesson TTS Button */}
                  <GlassButton
                    onClick={handleCompleteLessonTTS}
                    disabled={isGeneratingTTS}
                    icon={Volume2}
                    variant="primary"
                    className="px-6 py-3 text-lg font-medium shadow-lg hover:shadow-xl transition-all"
                  >
                    {isGeneratingTTS ? "Generating Audio..." : "🎵 Play Complete Lesson"}
                  </GlassButton>

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
      </div>
    </GlassContainer>
  );
}


