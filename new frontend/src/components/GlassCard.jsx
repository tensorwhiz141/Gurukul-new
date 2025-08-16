import React, { useEffect, useRef } from "react";
import gsap from "gsap";

/**
 * GlassCard - Animated flip card component with GSAP
 * Props:
 * - frontContent: React element for the front of the card
 * - backContent: React element for the back of the card
 * - className: Additional classes
 * - onClick: Click handler
 */
export default function GlassCard({
  frontContent,
  backContent,
  className = "",
  onClick,
}) {
  const cardRef = useRef(null);
  const frontRef = useRef(null);
  const backRef = useRef(null);

  useEffect(() => {
    const card = cardRef.current;
    const front = frontRef.current;
    const back = backRef.current;

    if (!card || !front || !back) return;

    // Initial state
    gsap.set(back, { rotationY: 180 });

    // Create hover animation
    const timeline = gsap
      .timeline({ paused: true })
      .to(front, {
        rotationY: 180,
        duration: 0.6,
        ease: "power2.inOut",
      })
      .to(
        back,
        {
          rotationY: 360,
          duration: 0.6,
          ease: "power2.inOut",
        },
        0
      )
      .to(
        card,
        {
          z: 50,
          duration: 0.3,
          ease: "power1.inOut",
        },
        0
      );

    // Entrance animation
    gsap.fromTo(
      card,
      {
        opacity: 0,
        y: 20,
        scale: 0.95,
      },
      {
        opacity: 1,
        y: 0,
        scale: 1,
        duration: 0.5,
        ease: "power2.out",
      }
    );

    // Event handlers
    const onMouseEnter = () => timeline.play();
    const onMouseLeave = () => timeline.reverse();

    card.addEventListener("mouseenter", onMouseEnter);
    card.addEventListener("mouseleave", onMouseLeave);

    return () => {
      card.removeEventListener("mouseenter", onMouseEnter);
      card.removeEventListener("mouseleave", onMouseLeave);
      timeline.kill();
    };
  }, []);

  return (
    <div
      ref={cardRef}
      className={`flip-card cursor-pointer ${className}`}
      onClick={onClick}
      style={{
        perspective: "1000px",
        transformStyle: "preserve-3d",
      }}
    >
      <div
        ref={frontRef}
        className="flip-card-front absolute w-full h-full backface-hidden"
        style={{
          backfaceVisibility: "hidden",
          transformStyle: "preserve-3d",
        }}
      >
        {frontContent}
      </div>
      <div
        ref={backRef}
        className="flip-card-back absolute w-full h-full backface-hidden"
        style={{
          backfaceVisibility: "hidden",
          transformStyle: "preserve-3d",
        }}
      >
        {backContent}
      </div>
    </div>
  );
}
