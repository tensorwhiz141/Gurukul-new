import React from "react";

/**
 * GlassContainer - Reusable glassmorphic container for consistent styling
 * Props: children, className, style, maxWidth, noFixedHeight
 */
export default function GlassContainer({
  children,
  className = "",
  style = {},
  maxWidth = "100rem", // Increased from 1152px
  noFixedHeight = false,
  ...props
}) {
  const containerStyle = {
    background: "rgba(30, 30, 40, 0.25)",
    backdropFilter: "blur(20px)",
    WebkitBackdropFilter: "blur(20px)",
    border: "1px solid rgba(255, 255, 255, 0.225)",
    color: "#FFF",
    marginTop: "10px",
    marginBottom: "10px",
    maxWidth: maxWidth,
    ...style,
  };
  if (!noFixedHeight) {
    containerStyle.height = "calc(100vh - 120px)";
    containerStyle.maxHeight = "calc(100vh - 120px)";
    containerStyle.overflow = "auto"; // Changed from "hidden" to "auto"
  }
  return (
    <div className="w-full flex-1 flex items-start justify-center">
      <div
        className={`w-full mx-auto rounded-3xl p-8 ${className}`}
        style={containerStyle}
        {...props}
      >
        <div
          className={`w-full${
            noFixedHeight ? "" : " h-full overflow-y-auto pr-2 custom-scrollbar"
          }`}
          style={noFixedHeight ? { overflow: "visible" } : {}}
        >
          {children}
        </div>
      </div>
    </div>
  );
}
