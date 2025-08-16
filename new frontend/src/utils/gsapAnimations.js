import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

// Register ScrollTrigger plugin
gsap.registerPlugin(ScrollTrigger);

/**
 * Hover animation for elements
 * @param {HTMLElement} element - The element to animate
 * @param {Object} options - Animation options
 * @returns {Object} - Animation timeline
 */
export const hoverAnimation = (element, options = {}) => {
  const defaults = {
    scale: 1.05,
    duration: 0.3,
    ease: 'power2.out',
    ...options
  };

  // Create a timeline for hover animations
  const tl = gsap.timeline({ paused: true });
  
  tl.to(element, {
    scale: defaults.scale,
    duration: defaults.duration,
    ease: defaults.ease,
    ...options
  });

  return tl;
};

/**
 * Click animation for elements
 * @param {HTMLElement} element - The element to animate
 * @param {Object} options - Animation options
 */
export const clickAnimation = (element, options = {}) => {
  const defaults = {
    scale: 0.95,
    duration: 0.15,
    ease: 'power2.inOut',
    ...options
  };

  gsap.to(element, {
    scale: defaults.scale,
    duration: defaults.duration,
    ease: defaults.ease,
    ...options,
    onComplete: () => {
      gsap.to(element, {
        scale: 1,
        duration: defaults.duration,
        ease: defaults.ease
      });
    }
  });
};

/**
 * Fade in animation
 * @param {HTMLElement} element - The element to animate
 * @param {Object} options - Animation options
 * @returns {Object} - GSAP tween
 */
export const fadeIn = (element, options = {}) => {
  const defaults = {
    opacity: 0,
    y: 20,
    duration: 0.6,
    ease: 'power2.out',
    ...options
  };

  return gsap.from(element, {
    opacity: defaults.opacity,
    y: defaults.y,
    duration: defaults.duration,
    ease: defaults.ease,
    ...options
  });
};

/**
 * Stagger animation for multiple elements
 * @param {String|Array} elements - CSS selector or array of elements
 * @param {Object} options - Animation options
 * @returns {Object} - GSAP tween
 */
export const staggerAnimation = (elements, options = {}) => {
  const defaults = {
    opacity: 0,
    y: 20,
    duration: 0.6,
    stagger: 0.1,
    ease: 'power2.out',
    ...options
  };

  return gsap.from(elements, {
    opacity: defaults.opacity,
    y: defaults.y,
    duration: defaults.duration,
    stagger: defaults.stagger,
    ease: defaults.ease,
    ...options
  });
};

/**
 * Scroll trigger animation
 * @param {String|HTMLElement} trigger - The trigger element
 * @param {String|HTMLElement} target - The target element to animate
 * @param {Object} options - Animation options
 * @returns {Object} - ScrollTrigger instance
 */
export const scrollAnimation = (trigger, target, options = {}) => {
  const defaults = {
    opacity: 0,
    y: 50,
    duration: 0.8,
    ease: 'power2.out',
    start: 'top 80%',
    end: 'bottom 20%',
    toggleActions: 'play none none reverse',
    ...options
  };

  return gsap.from(target, {
    opacity: defaults.opacity,
    y: defaults.y,
    duration: defaults.duration,
    ease: defaults.ease,
    scrollTrigger: {
      trigger: trigger,
      start: defaults.start,
      end: defaults.end,
      toggleActions: defaults.toggleActions,
      ...options.scrollTrigger
    },
    ...options
  });
};

/**
 * Text reveal animation
 * @param {String|HTMLElement} element - The text element
 * @param {Object} options - Animation options
 * @returns {Object} - GSAP tween
 */
export const textReveal = (element, options = {}) => {
  const defaults = {
    duration: 0.8,
    ease: 'power2.out',
    ...options
  };

  // Split text into words or characters if needed
  const text = element.textContent;
  element.innerHTML = '';
  
  const chars = text.split('');
  chars.forEach(char => {
    const span = document.createElement('span');
    span.textContent = char === ' ' ? '\u00A0' : char;
    span.style.display = 'inline-block';
    span.style.opacity = 0;
    element.appendChild(span);
  });

  return gsap.to(element.children, {
    opacity: 1,
    y: 0,
    stagger: 0.03,
    duration: defaults.duration,
    ease: defaults.ease,
    ...options
  });
};
