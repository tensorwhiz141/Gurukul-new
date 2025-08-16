import gsap from 'gsap';

/**
 * Simple fade in animation
 * @param {HTMLElement} element - The element to animate
 * @param {number} duration - Animation duration in seconds
 * @param {number} delay - Animation delay in seconds
 * @param {number} y - Initial y offset
 * @returns {Object} - GSAP tween
 */
export const fadeIn = (element, duration = 0.5, delay = 0, y = 20) => {
  return gsap.fromTo(
    element,
    { opacity: 0, y },
    { 
      opacity: 1, 
      y: 0, 
      duration, 
      delay, 
      ease: "power2.out" 
    }
  );
};

/**
 * Fade in staggered elements
 * @param {Array|String} elements - Elements to animate
 * @param {number} duration - Animation duration in seconds
 * @param {number} stagger - Stagger amount in seconds
 * @param {number} delay - Animation delay in seconds
 * @param {number} y - Initial y offset
 * @returns {Object} - GSAP tween
 */
export const fadeInStaggered = (elements, duration = 0.5, stagger = 0.1, delay = 0, y = 20) => {
  return gsap.fromTo(
    elements,
    { opacity: 0, y },
    { 
      opacity: 1, 
      y: 0, 
      duration, 
      stagger, 
      delay, 
      ease: "power2.out" 
    }
  );
};

/**
 * Hover animation for buttons and interactive elements
 * @param {HTMLElement} element - The element to animate
 * @param {Object} enterProps - Properties to animate on hover in
 * @param {Object} leaveProps - Properties to animate on hover out
 */
export const attachHoverAnimation = (element, enterProps = {}, leaveProps = {}) => {
  if (!element) return;
  
  const defaultEnterProps = {
    scale: 1.05,
    duration: 0.3,
    ease: "power2.out"
  };
  
  const defaultLeaveProps = {
    scale: 1,
    duration: 0.3,
    ease: "power2.out"
  };
  
  const combinedEnterProps = { ...defaultEnterProps, ...enterProps };
  const combinedLeaveProps = { ...defaultLeaveProps, ...leaveProps };
  
  element.addEventListener('mouseenter', () => {
    gsap.to(element, combinedEnterProps);
  });
  
  element.addEventListener('mouseleave', () => {
    gsap.to(element, combinedLeaveProps);
  });
};

/**
 * Click animation for buttons
 * @param {HTMLElement} element - The element to animate
 * @param {number} scale - Scale amount
 * @param {number} duration - Animation duration in seconds
 */
export const createClickAnimation = (element, scale = 0.95, duration = 0.15) => {
  if (!element) return;
  
  element.addEventListener('mousedown', () => {
    gsap.to(element, {
      scale,
      duration,
      ease: "power2.inOut"
    });
  });
  
  element.addEventListener('mouseup', () => {
    gsap.to(element, {
      scale: 1,
      duration,
      ease: "power2.inOut"
    });
  });
};

/**
 * Text reveal animation (character by character)
 * @param {HTMLElement} element - The text element
 * @param {number} duration - Animation duration in seconds
 * @param {number} stagger - Stagger amount in seconds
 */
export const textReveal = (element, duration = 1, stagger = 0.03) => {
  if (!element) return;
  
  // Store original text
  const text = element.textContent;
  
  // Clear the element
  element.textContent = '';
  
  // Create spans for each character
  const chars = text.split('');
  chars.forEach(char => {
    const span = document.createElement('span');
    span.textContent = char === ' ' ? '\u00A0' : char;
    span.style.display = 'inline-block';
    span.style.opacity = 0;
    element.appendChild(span);
  });
  
  // Animate each character
  return gsap.to(element.children, {
    opacity: 1,
    y: 0,
    stagger,
    duration: duration / 2,
    ease: "power2.out"
  });
};
