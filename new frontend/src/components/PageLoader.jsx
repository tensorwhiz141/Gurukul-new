import React from "react";

/**
 * PageLoader - Full page loader with blurred background and centered spinner.
 * Usage: Show while R3F scene or UI is loading.
 */
export default function PageLoader() {
  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        zIndex: 9999,
        background: "rgba(255,255,255,0.3)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        className="loader-spinner"
        style={{
          width: 64,
          height: 64,
          border: "6px solid rgba(162, 89, 255, 0.2)",
          borderTop: "6px solid #a259ff",
          borderRadius: "50%",
          animation: "spin 1s linear infinite",
          boxShadow: "0 4px 32px 0 rgba(162, 89, 255, 0.09)",
        }}
      />
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
