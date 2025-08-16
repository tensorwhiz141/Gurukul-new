import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

// Register ScrollTrigger plugin
gsap.registerPlugin(ScrollTrigger);

/**
 * Custom hook for GSAP animations
 * @param {Function} animationCallback - Function that sets up the animation
 * @param {Array} dependencies - Dependencies array for useEffect
 * @returns {Object} - Ref to be attached to the animated element
 */
export const useGSAP = (animationCallback, dependencies = []) => {
  const elementRef = useRef(null);

  useEffect(() => {
    // Skip if element ref is not set
    if (!elementRef.current) return;

    // Create context to isolate animations
    const ctx = gsap.context(() => {
      // Call the animation callback with the element ref
      animationCallback(elementRef.current, gsap);
    });

    // Cleanup function
    return () => {
      ctx.revert(); // Reverts all animations created in this context
    };
  }, dependencies);

  return elementRef;
};

/**
 * Custom hook for hover animations
 * @param {Object} options - Animation options
 * @returns {Object} - Ref and event handlers
 */
export const useHoverAnimation = (options = {}) => {
  const elementRef = useRef(null);
  const animation = useRef(null);
  const initialState = useRef({});
  const isHovering = useRef(false);

  useEffect(() => {
    if (!elementRef.current) return;

    const element = elementRef.current;
    const defaults = {
      scale: 1.05,
      duration: 0.3,
      ease: "power2.out",
    };

    // Store initial state of the element
    const computedStyle = window.getComputedStyle(element);
    initialState.current = {
      scale: 1,
      color: computedStyle.color,
      background: computedStyle.backgroundColor,
      boxShadow: computedStyle.boxShadow,
      textShadow: computedStyle.textShadow,
    };

    // Create animation config
    const animationConfig = {
      ...defaults,
      ...options,
      paused: true,
      onReverseComplete: () => {
        if (!isHovering.current && animation.current) {
          // Force reset to initial state if needed
          gsap.set(element, initialState.current);
        }
      },
    };

    // Create the animation
    animation.current = gsap.to(element, animationConfig);

    // Ensure initial state is set
    gsap.set(element, initialState.current);

    return () => {
      if (animation.current) {
        animation.current.kill();
      }
    };
  }, [options]);

  const handleMouseEnter = () => {
    if (animation.current) {
      isHovering.current = true;
      animation.current.play();
    }
  };

  const handleMouseLeave = () => {
    if (animation.current) {
      isHovering.current = false;

      // Ensure animation completes the reverse
      animation.current.reverse();

      // Force immediate reset if animation is stuck
      if (animation.current.progress() === 0) {
        gsap.set(elementRef.current, initialState.current);
      }
    }
  };

  return { ref: elementRef, handleMouseEnter, handleMouseLeave };
};

/**
 * Custom hook for scroll animations
 * @param {Object} options - Animation options
 * @returns {Object} - Ref to be attached to the animated element
 */
export const useScrollAnimation = (options = {}) => {
  const elementRef = useRef(null);

  useEffect(() => {
    if (!elementRef.current) return;

    const element = elementRef.current;
    const defaults = {
      opacity: 0,
      y: 50,
      duration: 0.8,
      ease: "power2.out",
      start: "top 80%",
      end: "bottom 20%",
      toggleActions: "play none none reverse",
      ...options,
    };

    const animation = gsap.from(element, {
      opacity: defaults.opacity,
      y: defaults.y,
      duration: defaults.duration,
      ease: defaults.ease,
      scrollTrigger: {
        trigger: element,
        start: defaults.start,
        end: defaults.end,
        toggleActions: defaults.toggleActions,
        ...options.scrollTrigger,
      },
      ...options,
    });

    return () => {
      if (animation.scrollTrigger) {
        animation.scrollTrigger.kill();
      }
      animation.kill();
    };
  }, [options]);

  return elementRef;
};

export default useGSAP;
