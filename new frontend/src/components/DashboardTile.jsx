import React, { useEffect, useRef } from "react";
import gsap from "gsap";

/**
 * DashboardTile - Animated glass tile component for dashboard sections
 */
export default function DashboardTile({
  icon,
  title,
  children,
  footer,
  className = "",
}) {
  const tileRef = useRef(null);
  const contentRef = useRef(null);
  const iconRef = useRef(null);

  useEffect(() => {
    const tile = tileRef.current;
    const content = contentRef.current;
    const iconContainer = iconRef.current;

    if (!tile || !content || !iconContainer) return;

    // Entrance animation
    gsap.fromTo(
      tile,
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

    // Content fade in
    gsap.fromTo(
      content,
      {
        opacity: 0,
        y: 10,
      },
      {
        opacity: 1,
        y: 0,
        duration: 0.4,
        delay: 0.2,
        ease: "power2.out",
      }
    );

    // Icon animation
    gsap.fromTo(
      iconContainer,
      {
        scale: 0,
        rotate: -45,
      },
      {
        scale: 1,
        rotate: 0,
        duration: 0.6,
        ease: "back.out(1.7)",
      }
    );

    // Hover animation
    const hoverAnimation = gsap
      .timeline({ paused: true })
      .to(tile, {
        scale: 1.02,
        boxShadow:
          "0 8px 32px rgba(0, 0, 0, 0.15), 0 2px 8px rgba(255, 215, 0, 0.2) inset",
        duration: 0.3,
        ease: "power2.out",
      })
      .to(
        iconContainer,
        {
          scale: 1.1,
          rotate: 5,
          duration: 0.3,
          ease: "power2.out",
        },
        0
      );

    // Event handlers
    const onMouseEnter = () => hoverAnimation.play();
    const onMouseLeave = () => hoverAnimation.reverse();

    tile.addEventListener("mouseenter", onMouseEnter);
    tile.addEventListener("mouseleave", onMouseLeave);

    return () => {
      tile.removeEventListener("mouseenter", onMouseEnter);
      tile.removeEventListener("mouseleave", onMouseLeave);
      hoverAnimation.kill();
    };
  }, []);

  return (
    <div
      ref={tileRef}
      className={`bg-white/5 rounded-xl shadow-inner flex flex-col justify-between h-full min-h-80 p-6 ${className}`}
    >
      <div className="flex items-center mb-3">
        <span ref={iconRef} className="wide-list-icon">
          {icon}
        </span>
        <span className="font-bold ml-2 text-lg md:text-xl">{title}</span>
      </div>
      <div ref={contentRef} className="flex-1 flex flex-col justify-center">
        {children}
      </div>
      {footer && <div className="mt-4">{footer}</div>}
    </div>
  );
}
