import React, { useEffect, useState, useRef } from "react";
import {
  FiDownload,
  FiX,
  FiPlay,
  FiPause,
  FiVolume2,
  FiVolumeX,
} from "react-icons/fi";
import { useNavigate } from "react-router-dom";
import GlassContainer from "../components/GlassContainer";
import { API_BASE_URL } from "../config";

// Custom styled audio player component
const CustomAudioPlayer = ({ audioSrc }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const audioRef = useRef(null);
  const progressBarRef = useRef(null);

  useEffect(() => {
    // Get audio duration when metadata is loaded
    const audio = audioRef.current;
    if (!audio) return;

    const setAudioData = () => {
      setDuration(audio.duration);
    };

    const updateTime = () => {
      setCurrentTime(audio.currentTime);
    };

    // Add event listeners
    audio.addEventListener("loadedmetadata", setAudioData);
    audio.addEventListener("timeupdate", updateTime);

    // Cleanup
    return () => {
      audio.removeEventListener("loadedmetadata", setAudioData);
      audio.removeEventListener("timeupdate", updateTime);
    };
  }, [audioRef]);

  // Format time in MM:SS
  const formatTime = (time) => {
    if (isNaN(time)) return "00:00";
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes.toString().padStart(2, "0")}:${seconds
      .toString()
      .padStart(2, "0")}`;
  };

  // Toggle play/pause
  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  // Toggle mute
  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  // Handle volume change
  const handleVolumeChange = (e) => {
    const newVolume = parseFloat(e.target.value);
    const audio = audioRef.current;
    if (!audio) return;

    audio.volume = newVolume;
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  };

  // Handle progress bar click
  const handleProgressBarClick = (e) => {
    const audio = audioRef.current;
    const progressBar = progressBarRef.current;
    if (!audio || !progressBar) return;

    const rect = progressBar.getBoundingClientRect();
    const clickPosition = (e.clientX - rect.left) / rect.width;
    const newTime = clickPosition * duration;

    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  // Handle audio end
  const handleAudioEnd = () => {
    setIsPlaying(false);
    setCurrentTime(0);
    audioRef.current.currentTime = 0;
  };

  return (
    <div className="bg-black/20 rounded-lg p-3">
      {/* Hidden native audio element */}
      <audio
        ref={audioRef}
        src={audioSrc}
        onEnded={handleAudioEnd}
        preload="metadata"
      />

      {/* Custom audio player UI */}
      <div className="flex items-center gap-3">
        {/* Play/Pause button */}
        <button
          onClick={togglePlay}
          className="w-10 h-10 flex items-center justify-center bg-[#FF9933]/30 hover:bg-[#FF9933]/50 rounded-full transition-all duration-200"
        >
          {isPlaying ? (
            <FiPause className="text-white w-5 h-5" />
          ) : (
            <FiPlay className="text-white w-5 h-5 ml-0.5" />
          )}
        </button>

        {/* Time display */}
        <div className="text-white/80 text-xs w-16">
          {formatTime(currentTime)}
        </div>

        {/* Progress bar */}
        <div
          ref={progressBarRef}
          className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden cursor-pointer"
          onClick={handleProgressBarClick}
        >
          <div
            className="h-full bg-gradient-to-r from-[#FF9933] to-[#FFD700] transition-all duration-100"
            style={{ width: `${(currentTime / duration) * 100}%` }}
          ></div>
        </div>

        {/* Duration */}
        <div className="text-white/80 text-xs w-16 text-right">
          {formatTime(duration)}
        </div>

        {/* Volume control */}
        <div className="flex items-center gap-2">
          <button
            onClick={toggleMute}
            className="text-white/80 hover:text-white transition-colors duration-200"
          >
            {isMuted ? (
              <FiVolumeX className="w-5 h-5" />
            ) : (
              <FiVolume2 className="w-5 h-5" />
            )}
          </button>

          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={isMuted ? 0 : volume}
            onChange={handleVolumeChange}
            className="w-16 accent-[#FF9933] cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
};

export default function SummaryView() {
  const navigate = useNavigate();
  const [summaryData, setSummaryData] = React.useState(null);
  const [file, setFile] = React.useState(null);
  const [error, setError] = React.useState(null);

  // Load data from localStorage on mount
  useEffect(() => {
    try {
      const savedSummary = localStorage.getItem("summaryData");
      const savedFile = localStorage.getItem("fileData");
      if (savedSummary && savedFile) {
        setSummaryData(JSON.parse(savedSummary));
        setFile(JSON.parse(savedFile));
      } else {
        navigate("/learn");
      }
    } catch (err) {
      setError("Failed to load summary data");
      console.error("Error loading summary:", err);
    }
  }, [navigate]);

  const handleReset = () => {
    try {
      localStorage.removeItem("summaryData");
      localStorage.removeItem("fileData");
      navigate("/learn");
    } catch (err) {
      console.error("Error resetting data:", err);
    }
  };

  const formatSummary = (text) => {
    if (!text) return [];

    try {
      // First check if the text contains numbered points (1., 2., etc.)
      const paragraphs = text.split("\n\n");

      // Check if we have a list of numbered points
      const hasNumberedPoints = paragraphs.some((p) =>
        p.trim().match(/^\d+\.\s/)
      );

      if (hasNumberedPoints) {
        // Process as a list of sections with numbered points
        const sections = [];
        let currentSection = { title: "Summary", points: [] };

        paragraphs.forEach((paragraph) => {
          // If it's a numbered point
          if (paragraph.trim().match(/^\d+\.\s/)) {
            currentSection.points.push(paragraph.trim());
          } else if (paragraph.trim()) {
            // If we already have points and find a new paragraph, it's a new section
            if (currentSection.points.length > 0) {
              sections.push(currentSection);
              currentSection = { title: paragraph.trim(), points: [] };
            } else {
              // Otherwise it's the title/intro of the current section
              currentSection.title = paragraph.trim();
            }
          }
        });

        // Add the last section if it has points
        if (currentSection.points.length > 0) {
          sections.push(currentSection);
        }

        return sections;
      } else {
        // Process as regular paragraphs
        // Extract bullet points (lines starting with *)
        const bulletPoints = text
          .split("\n")
          .filter(
            (line) => line.trim().startsWith("*") || line.trim().startsWith("-")
          )
          .map((line) =>
            line
              .trim()
              .substring(line.trim().startsWith("*") ? 1 : 1)
              .trim()
          );

        if (bulletPoints.length > 0) {
          // If we found bullet points, return them as a single section
          return [
            {
              title: "Key Points",
              points: bulletPoints,
            },
          ];
        } else {
          // Otherwise split by paragraphs and create sections
          const paragraphs = text.split("\n\n").filter((p) => p.trim());

          if (paragraphs.length <= 1) {
            // If there's only one paragraph, return it as a single section
            return [
              {
                title: "Summary",
                points: [paragraphs[0]],
              },
            ];
          } else {
            // First paragraph is intro, rest are points
            return [
              {
                title: paragraphs[0],
                points: paragraphs.slice(1),
              },
            ];
          }
        }
      }
    } catch (error) {
      console.error("Error formatting summary:", error);
      return [{ title: "Summary", points: [text] }];
    }
  };

  if (error) {
    return (
      <GlassContainer>
        <div className="text-center py-8">
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={() => navigate("/learn")}
            className="px-4 py-2 bg-[#FF9933]/20 hover:bg-[#FF9933]/30 rounded-lg transition-all duration-300 text-white"
          >
            Return to Summarizer
          </button>
        </div>
      </GlassContainer>
    );
  }

  if (!summaryData || !file) {
    return (
      <GlassContainer>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[#FF9933]"></div>
        </div>
      </GlassContainer>
    );
  }

  const phases = formatSummary(summaryData?.answer);

  return (
    <GlassContainer className="p-0">
      <div className="w-[95%] mx-auto mt-8 mb-8">
        <div className="bg-black/30 backdrop-blur-md p-8 rounded-xl shadow-xl space-y-6 relative">
          {/* Close Button */}
          <button
            onClick={handleReset}
            className="absolute -right-3 -top-3 p-2.5 bg-red-600/80 hover:bg-red-500/90 rounded-full transition-all duration-300 group z-20 border border-red-400/30"
            title="Close summary"
          >
            <FiX className="w-5 h-5 text-white group-hover:text-white/90 transition-colors duration-300" />
          </button>

          {/* Title and Model Info */}
          <div className="border-b border-white/10 pb-4">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl text-white font-bold">
                {summaryData.title || "Document Summary"}
              </h2>

              {/* AI Model Badge */}
              {summaryData.llm && (
                <div
                  className="inline-block px-3 py-1 rounded text-sm font-medium"
                  style={{
                    background:
                      summaryData.llm === "grok"
                        ? "linear-gradient(135deg, rgba(255, 153, 51, 0.3), rgba(255, 153, 51, 0.1))"
                        : summaryData.llm === "llama"
                        ? "linear-gradient(135deg, rgba(0, 128, 255, 0.3), rgba(0, 128, 255, 0.1))"
                        : summaryData.llm === "chatgpt"
                        ? "linear-gradient(135deg, rgba(16, 163, 127, 0.3), rgba(16, 163, 127, 0.1))"
                        : "linear-gradient(135deg, rgba(128, 0, 255, 0.3), rgba(128, 0, 255, 0.1))",
                    border:
                      summaryData.llm === "grok"
                        ? "1px solid rgba(255, 153, 51, 0.3)"
                        : summaryData.llm === "llama"
                        ? "1px solid rgba(0, 128, 255, 0.3)"
                        : summaryData.llm === "chatgpt"
                        ? "1px solid rgba(16, 163, 127, 0.3)"
                        : "1px solid rgba(128, 0, 255, 0.3)",
                    boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
                  }}
                >
                  Analyzed by{" "}
                  {summaryData.llm.charAt(0).toUpperCase() +
                    summaryData.llm.slice(1)}
                </div>
              )}
            </div>
          </div>

          {/* Content Sections */}
          <div className="space-y-8">
            {phases.map((section, index) => (
              <div
                key={index}
                className="bg-black/20 p-6 rounded-xl border border-white/5"
              >
                {/* Section Header */}
                <div className="flex items-center mb-4">
                  <span className="bg-[#FF9933]/20 px-3 py-1 rounded-md text-sm font-bold text-white/90">
                    Section {index + 1}
                  </span>
                  <h3 className="text-lg text-white/90 font-semibold ml-3">
                    {section.title}
                  </h3>
                </div>

                {/* Section Content */}
                <div className="space-y-4 ml-4">
                  {section.points.map((point, i) => (
                    <div
                      key={i}
                      className="flex items-start text-white/80 group"
                    >
                      {point.match(/^\d+\.\s/) ? (
                        // For numbered points, extract the number
                        <>
                          <span className="text-[#FF9933] mr-2 font-bold min-w-[24px]">
                            {point.match(/^\d+\./)[0]}
                          </span>
                          <span>{point.replace(/^\d+\.\s/, "")}</span>
                        </>
                      ) : (
                        // For regular points
                        <>
                          <span className="text-[#FF9933] mr-2 mt-1.5 transform group-hover:scale-125 transition-transform duration-200">
                            •
                          </span>
                          <span>{point}</span>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {/* Audio Section */}
            {summaryData.audio_file && (
              <div className="mt-8 pt-6 border-t border-white/10">
                <div className="bg-black/20 rounded-lg backdrop-blur-sm border border-white/10">
                  <div className="flex items-center px-4 py-3">
                    <div className="flex items-center flex-1 min-w-0">
                      <span className="w-8 h-8 flex items-center justify-center bg-[#FF9933]/20 rounded-full mr-3">
                        <span className="text-lg">🎧</span>
                      </span>
                      <div className="flex-1 min-w-0">
                        <h4 className="text-white font-medium text-sm">
                          Audio Summary
                        </h4>
                        <p className="text-white/60 text-xs truncate">
                          {summaryData.audio_file.split("/").pop()}
                        </p>
                      </div>
                    </div>

                    <a
                      href={`${API_BASE_URL}/api/audio/${summaryData.audio_file
                        .split("/")
                        .pop()}`}
                      download
                      className="ml-4 p-2 bg-[#FF9933]/10 hover:bg-[#FF9933]/20 rounded-lg transition-all duration-300 group"
                      title="Download audio"
                    >
                      <FiDownload className="w-4 h-4 text-white/70 group-hover:text-white group-hover:scale-110 transition-all duration-300" />
                    </a>
                  </div>

                  <div className="px-4 pb-3">
                    <CustomAudioPlayer
                      audioSrc={`${API_BASE_URL}/api/stream/${summaryData.audio_file
                        .split("/")
                        .pop()}`}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </GlassContainer>
  );
}
