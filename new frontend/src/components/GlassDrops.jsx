import React from "react";

/**
 * Floating glassmorphic drops for background (from CodePen inspiration)
 * Usage: Place <GlassDrops /> inside a relatively positioned container
 */
export default function GlassDrops() {
  return (
    <div className="drops pointer-events-none select-none">
      <div
        className="drop drop-1"
        style={{
          position: "absolute",
          height: 80,
          width: 80,
          top: -20,
          left: -40,
          zIndex: -1,
          background: "rgba(255,255,255,0.3)",
          backdropFilter: "blur(10px)",
          borderRadius: 10,
          borderLeft: "1px solid rgba(255,255,255,0.3)",
          borderTop: "1px solid rgba(255,255,255,0.3)",
          boxShadow: "10px 10px 60px -8px rgba(0,0,0,0.2)",
        }}
      ></div>
      <div
        className="drop drop-2"
        style={{
          position: "absolute",
          height: 80,
          width: 80,
          bottom: -30,
          right: -10,
          background: "rgba(255,255,255,0.3)",
          backdropFilter: "blur(10px)",
          borderRadius: 10,
          borderLeft: "1px solid rgba(255,255,255,0.3)",
          borderTop: "1px solid rgba(255,255,255,0.3)",
          boxShadow: "10px 10px 60px -8px rgba(0,0,0,0.2)",
        }}
      ></div>
      <div
        className="drop drop-3"
        style={{
          position: "absolute",
          height: 100,
          width: 100,
          bottom: 120,
          right: -50,
          zIndex: -1,
          background: "rgba(255,255,255,0.3)",
          backdropFilter: "blur(10px)",
          borderRadius: 10,
          borderLeft: "1px solid rgba(255,255,255,0.3)",
          borderTop: "1px solid rgba(255,255,255,0.3)",
          boxShadow: "10px 10px 60px -8px rgba(0,0,0,0.2)",
        }}
      ></div>
      <div
        className="drop drop-4"
        style={{
          position: "absolute",
          height: 120,
          width: 120,
          top: -60,
          right: -60,
          background: "rgba(255,255,255,0.3)",
          backdropFilter: "blur(10px)",
          borderRadius: 10,
          borderLeft: "1px solid rgba(255,255,255,0.3)",
          borderTop: "1px solid rgba(255,255,255,0.3)",
          boxShadow: "10px 10px 60px -8px rgba(0,0,0,0.2)",
        }}
      ></div>
      <div
        className="drop drop-5"
        style={{
          position: "absolute",
          height: 60,
          width: 60,
          bottom: 170,
          left: 90,
          zIndex: -1,
          background: "rgba(255,255,255,0.3)",
          backdropFilter: "blur(10px)",
          borderRadius: 10,
          borderLeft: "1px solid rgba(255,255,255,0.3)",
          borderTop: "1px solid rgba(255,255,255,0.3)",
          boxShadow: "10px 10px 60px -8px rgba(0,0,0,0.2)",
        }}
      ></div>
    </div>
  );
}
