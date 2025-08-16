import React from "react";
import { ChevronLeft } from "lucide-react";

export default function SideBarAltUse({
  generatedVideo,
  onClose,
  isVisible = true
}) {



  if (!isVisible || !generatedVideo) {
    return null;
  }

  return (
    <aside
      className="transition-all duration-300 ease-in-out h-auto rounded-2xl"
      style={{
        marginTop: "1.3rem",
        marginBottom: "0",
        background: "rgba(30, 30, 40, 0.25)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        border: "0px solid rgba(255, 255, 255, 0.18)",
        borderRight: "1px solid rgba(255, 255, 255, 0.18)",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.1), 0 1px 8px rgba(255, 215, 0, 0.05) inset",
        display: "flex",
        flexDirection: "column",
        justifyContent: "flex-start",
        width: "100%",
        maxWidth: "100%",
        height: "auto",
        padding: "0",
      }}
    >
      {/* Header with toggle button - matching normal sidebar */}
      <div className="py-5 px-5 flex items-center justify-between">
        <span
          className="text-lg font-semibold text-white"
          style={{ fontFamily: "Nunito, sans-serif" }}
        >
          🎬 Video
        </span>
        <button
          onClick={onClose}
          className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center ml-2 hover:bg-white/20 transition-all duration-300"
        >
          <ChevronLeft size={16} className="text-white" />
        </button>
      </div>

      {/* Video content area */}
      <div className="flex-1 px-5 pb-5">
        <div className="flex flex-col h-full">
          <div className="mb-3">
            <p className="text-amber-300 text-sm font-medium">
              📚 {generatedVideo.subject}
            </p>
            <p className="text-white/70 text-xs">
              {generatedVideo.topic}
            </p>
          </div>

          <div className="flex-1">
            <video
              src={generatedVideo.url}
              controls
              className="w-full rounded-lg"
              style={{
                minHeight: '150px',
                maxHeight: '250px',
                objectFit: 'cover',
                backgroundColor: 'rgba(0,0,0,0.3)'
              }}
            />
          </div>

          <div className="mt-3 text-center">
            <p className="text-white/50 text-xs">
              🎥 {(generatedVideo.size / 1024 / 1024).toFixed(1)} MB
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}
