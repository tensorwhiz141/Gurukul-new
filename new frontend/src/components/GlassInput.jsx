import React from "react";

/**
 * Glassmorphic input, styled after https://codepen.io/dasshounak/pen/QWKKYdj
 * Props: type, placeholder, name, value, onChange, autoComplete, icon, ...rest
 */
export default function GlassInput({ className = "", icon: Icon, ...props }) {
  return (
    <div className="relative w-full mb-4">
      {Icon && (
        <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-white/70">
          <Icon size={18} />
        </div>
      )}
      <input
        className={`w-full h-12 px-4 ${
          Icon ? "pl-10" : "pl-4"
        } rounded-xl text-white font-medium placeholder:text-gray-300/70 outline-none ${className}`}
        style={{
          background: "rgba(255, 255, 255, 0.15)",
          backdropFilter: "blur(10px)",
          WebkitBackdropFilter: "blur(10px)",
          border: "1px solid rgba(255, 255, 255, 0.2)",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
          transition: "all 0.3s ease",
        }}
        {...props}
      />
    </div>
  );
}
