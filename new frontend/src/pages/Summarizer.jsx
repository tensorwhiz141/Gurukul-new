import React, { useState, useRef, useEffect } from "react";
import GlassContainer from "../components/GlassContainer";
import { FiUpload, FiFileText } from "react-icons/fi";
import { useNavigate } from "react-router-dom";
import { toast } from "react-hot-toast";
import { useTranslation } from "react-i18next";
import { supabase } from "../supabaseClient";
import chatLogsService from "../services/chatLogsService";
import {
  useUploadPdfForSummaryMutation,
  useUploadImageForSummaryMutation,
  useLazyGetPdfSummaryQuery,
  useLazyGetImageSummaryQuery,
} from "../api/summaryApiSlice";

export default function Summarizer() {
  const { t } = useTranslation();
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedModel, setSelectedModel] = useState("grok"); // Default to grok model
  const [userId, setUserId] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  // RTK Query hooks
  const [uploadPdfForSummary] = useUploadPdfForSummaryMutation();
  const [uploadImageForSummary] = useUploadImageForSummaryMutation();
  const [getPdfSummary] = useLazyGetPdfSummaryQuery();
  const [getImageSummary] = useLazyGetImageSummaryQuery();

  // Load the selected model from localStorage if available
  useEffect(() => {
    const savedModel = localStorage.getItem("selectedAIModel");
    if (savedModel) {
      setSelectedModel(savedModel);
    }
  }, []);

  // Get user ID on component mount
  useEffect(() => {
    const getUserId = async () => {
      try {
        // Get the current session
        const { data: sessionData } = await supabase.auth.getSession();

        // If we have a session with a user, use that ID
        if (sessionData?.session?.user?.id) {
          setUserId(sessionData.session.user.id);
          return;
        }

        // Fallback to getUser if session doesn't have what we need
        const { data: userData } = await supabase.auth.getUser();

        if (userData?.user?.id) {
          setUserId(userData.user.id);
        } else {
          // Use guest-user for anonymous users
          setUserId("guest-user");
        }
      } catch (error) {
        console.error("Error getting user ID:", error);
        setUserId("guest-user");
      }
    };

    getUserId();
  }, []);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      const validTypes = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "image/jpeg",
        "image/png",
      ];

      if (!validTypes.includes(selectedFile.type)) {
        const errorMsg = "Please upload a PDF, DOC, DOCX, JPG, or PNG file";
        setError(errorMsg);
        toast.error(errorMsg, {
          icon: "❌",
          position: "bottom-right",
        });
        return;
      }

      if (selectedFile.size > 10 * 1024 * 1024) {
        const errorMsg = "File size should be less than 10MB";
        setError(errorMsg);
        toast.error(errorMsg, {
          icon: "⚠️",
          position: "bottom-right",
        });
        return;
      }

      setFile(selectedFile);
      setError("");

      // Show success toast for file selection
      toast.success(`File "${selectedFile.name}" selected`, {
        icon: "📄",
        position: "bottom-right",
      });
    }
  };

  const getSummary = async (isImage) => {
    try {
      if (isImage) {
        const response = await getImageSummary().unwrap();
        return response;
      } else {
        const response = await getPdfSummary().unwrap();
        return response;
      }
    } catch (error) {
      throw new Error("Failed to fetch summary");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      const errorMsg = "Please select a file first";
      setError(errorMsg);
      toast.error(errorMsg, {
        icon: "📁",
        position: "bottom-right",
      });
      return;
    }

    try {
      setLoading(true);
      setError("");

      const isImage = file.type.startsWith("image/");

      // Save the selected model to localStorage for use in other components
      localStorage.setItem("selectedAIModel", selectedModel);

      // Show upload progress toast
      toast.loading(
        `Uploading file to ${
          selectedModel.charAt(0).toUpperCase() + selectedModel.slice(1)
        }...`,
        {
          id: "upload-progress",
          position: "bottom-right",
        }
      );

      // Upload file using RTK Query
      if (isImage) {
        await uploadImageForSummary({
          file,
          llm: selectedModel,
        }).unwrap();
      } else {
        await uploadPdfForSummary({
          file,
          llm: selectedModel,
        }).unwrap();
      }

      // Dismiss upload progress toast
      toast.dismiss("upload-progress");

      // Show analyzing toast
      toast.loading("Analyzing document...", {
        id: "analyzing-progress",
        position: "bottom-right",
      });

      // Get summary from the appropriate endpoint based on file type
      const summaryResponse = await getSummary(isImage);

      // Dismiss analyzing toast
      toast.dismiss("analyzing-progress");

      // Save data to localStorage
      localStorage.setItem("summaryData", JSON.stringify(summaryResponse));
      localStorage.setItem(
        "fileData",
        JSON.stringify({
          type: file.type,
          name: file.name,
        })
      );

      // Show success toast
      toast.success("Document analyzed successfully!", {
        icon: "🎉",
        position: "bottom-right",
        duration: 3000,
      });

      // Log document summary to Supabase
      try {
        const effectiveUserId = userId || "guest-user";
        await chatLogsService.logDocumentSummary({
          userId: effectiveUserId,
          fileName: file.name,
          fileType: file.type,
          fileSize: file.size,
          model: selectedModel,
          hasAudio: summaryResponse.audio_file ? true : false,
        });
        console.log("Document summary logged successfully");
      } catch (logError) {
        console.error("Error logging document summary:", logError);
        // Continue even if logging fails
      }

      // Navigate to summary page
      navigate("/learn/summary");
    } catch (err) {
      const errorMsg = err.message || "Failed to process file";
      setError(errorMsg);
      toast.error(errorMsg, {
        position: "bottom-right",
        duration: 5000,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <GlassContainer>
      <div className="w-[95%] mx-auto px-4">
        {/* Header Section */}
        <div className="text-center mb-12">
          <h2
            className="text-5xl md:text-6xl font-extrabold mb-6 drop-shadow-lg transition-all duration-300 hover:bg-gradient-to-r hover:from-white hover:to-[#FF9933] hover:bg-clip-text hover:text-transparent"
            style={{
              color: "#FFFFFF",
              fontFamily: "Nunito, sans-serif",
            }}
          >
            Smart Document Analysis
          </h2>
          <p className="text-white/80 text-xl mb-3">
            Upload your documents for instant AI-powered summaries
          </p>
          <p className="text-white/60 text-sm">
            Supported formats: PDF, DOC, DOCX, JPG, PNG • Max size: 10MB
          </p>

          {/* AI Model Selector */}
          <div className="mt-4 flex items-center justify-center">
            <div className="relative group">
              <div className="absolute -top-8 left-0 bg-black/80 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
                {t("Select AI Model for Analysis")}
              </div>
              <label className="text-white/80 mr-3">{t("AI Model")}:</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="appearance-none bg-gradient-to-r from-[#FF9933]/30 to-[#FF9933]/10 text-white border-2 border-[#FF9933]/30 rounded-lg px-3 py-2 text-sm font-medium outline-none cursor-pointer"
                style={{
                  backdropFilter: "blur(10px)",
                  boxShadow: "0 2px 10px rgba(255, 153, 51, 0.2)",
                  textShadow: "0 1px 2px rgba(0, 0, 0, 0.3)",
                  minWidth: "110px",
                }}
                disabled={loading}
              >
                <option value="grok" className="bg-[#1E1E28] text-white">
                  Grok
                </option>
                <option value="llama" className="bg-[#1E1E28] text-white">
                  Llama
                </option>
                <option value="chatgpt" className="bg-[#1E1E28] text-white">
                  ChatGPT
                </option>
                <option value="uniguru" className="bg-[#1E1E28] text-white">
                  UniGuru
                </option>
              </select>
              {/* Custom dropdown arrow */}
              <div className="absolute right-2 top-1/2 transform -translate-y-1/2 pointer-events-none">
                <svg
                  width="12"
                  height="6"
                  viewBox="0 0 12 6"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M1 1L6 5L11 1"
                    stroke="white"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-center space-y-6 mt-6">
          {/* Upload Box */}
          <div
            className="w-72 h-72 border-2 border-dashed border-white/30 rounded-2xl flex flex-col items-center justify-center p-8 bg-white/5 hover:bg-white/10 transition-all duration-300 cursor-pointer group relative overflow-hidden"
            onClick={() => fileInputRef.current.click()}
          >
            <input
              type="file"
              onChange={handleFileChange}
              ref={fileInputRef}
              className="hidden"
              accept=".pdf,.doc,.docx,.jpg,.jpeg,.png"
            />

            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>

            {!file ? (
              <>
                <FiFileText className="text-white/70 w-20 h-20 mb-6 group-hover:text-[#FF9933] transition-colors duration-300" />
                <div className="text-center relative z-10">
                  <p className="text-white text-lg font-medium mb-2">
                    Drop your file here
                  </p>
                  <p className="text-white/60">or click to browse</p>
                </div>
              </>
            ) : (
              <div className="text-center relative z-10">
                <FiFileText className="text-[#FF9933] w-16 h-16 mx-auto mb-4" />
                <p className="text-white/90 font-medium mb-2 break-all">
                  {file.name}
                </p>
                <p className="text-white/60">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            )}
          </div>

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className={`w-72 px-8 py-4 rounded-xl transition-all duration-300 flex items-center justify-center space-x-3 ${
              !file || loading
                ? "bg-gray-500/50 cursor-not-allowed"
                : "bg-[#FF9933]/20 hover:bg-[#FF9933]/30 hover:scale-105 active:scale-95"
            }`}
          >
            {loading ? (
              <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
            ) : (
              <FiUpload className="text-white w-5 h-5" />
            )}
            <span className="text-white text-lg">
              {loading ? "Processing..." : "Analyze Document"}
            </span>
          </button>

          {/* Error Display */}
          {error && (
            <div className="text-red-400 text-center bg-red-500/10 p-4 rounded-xl border border-red-500/20 backdrop-blur-sm">
              {error}
            </div>
          )}
        </div>
      </div>
    </GlassContainer>
  );
}
