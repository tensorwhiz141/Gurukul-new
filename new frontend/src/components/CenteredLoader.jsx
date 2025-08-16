import React from "react";
import { RiseLoader } from "react-spinners";

/**
 * CenteredLoader - A centered loader component for use within GlassContainer
 *
 * @param {Object} props - Component props
 * @param {string} props.color - Color of the loader (default: "#FF9933")
 * @param {string} props.size - Size of the loader (default: 15)
 * @param {string} props.margin - Margin between loader dots (default: 2)
 * @param {string} props.text - Optional text to display below the loader
 * @param {string} props.className - Additional CSS classes
 * @returns {JSX.Element} - Centered loader component
 */
export default function CenteredLoader({
  color = "#FF9933",
  size = 15,
  margin = 2,
  text,
  className = "",
}) {
  return (
    <div
      className={`absolute inset-0 flex flex-col items-center justify-center ${className}`}
      style={{ minHeight: "200px" }}
    >
      <div className="flex justify-center items-center">
        <RiseLoader color={color} size={size} margin={margin} />
      </div>
      {text && <p className="mt-4 text-white/80 text-center">{text}</p>}
    </div>
  );
}
