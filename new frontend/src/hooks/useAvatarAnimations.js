import { useEffect, useRef, useState, useCallback } from 'react';
import { useAnimations } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Custom hook for managing avatar animations with interactive triggers
 * Supports both built-in GLTF animations and procedural animations
 * @param {Object} gltf - The loaded GLTF object containing animations
 * @param {Object} meshRef - Reference to the 3D mesh
 * @param {Object} options - Configuration options
 * @returns {Object} Animation controls and state
 */
export function useAvatarAnimations(gltf, meshRef, options = {}) {
  const {
    enableInteractions = true,
    idleAnimationName = null, // Will auto-detect if null
    transitionDuration = 0.5,
    debug = false,
    enableProceduralAnimations = true // Enable manual animations when no GLTF animations exist
  } = options;

  // Early return with safe defaults if no gltf or meshRef
  if (!gltf || !meshRef) {
    if (debug) console.log('useAvatarAnimations: Missing gltf or meshRef, returning safe defaults');
    return {
      currentAnimation: null,
      availableAnimations: [],
      isHovered: false,
      isClicked: false,
      playAnimation: () => false,
      playIdleAnimation: () => false,
      getRandomAnimationByType: () => null,
      handleMouseEnter: () => {},
      handleMouseLeave: () => {},
      handleClick: () => {},
      actions: {},
      names: [],
      clips: []
    };
  }

  // Animation system from drei
  const { actions, names, clips } = useAnimations(gltf?.animations || [], meshRef);
  
  // Animation state
  const [currentAnimation, setCurrentAnimation] = useState(null);
  const [availableAnimations, setAvailableAnimations] = useState([]);
  const [isHovered, setIsHovered] = useState(false);
  const [isClicked, setIsClicked] = useState(false);
  const [hasGLTFAnimations, setHasGLTFAnimations] = useState(false);

  // Animation management refs
  const currentActionRef = useRef(null);
  const previousActionRef = useRef(null);
  const idleActionRef = useRef(null);
  const animationTimeoutRef = useRef(null);

  // Procedural animation refs
  const proceduralAnimationRef = useRef(null);
  const baseRotationRef = useRef([0, 0, 0]);
  const basePositionRef = useRef([0, 0, 0]);
  const animationStartTimeRef = useRef(0);

  // Categorize animation based on name patterns
  const categorizeAnimation = useCallback((name) => {
    const lowerName = name.toLowerCase();

    if (lowerName.includes('idle') || lowerName.includes('breathing') || lowerName.includes('rest')) {
      return 'idle';
    }
    if (lowerName.includes('wave') || lowerName.includes('hello') || lowerName.includes('hi')) {
      return 'greeting';
    }
    if (lowerName.includes('dance') || lowerName.includes('dancing')) {
      return 'dance';
    }
    if (lowerName.includes('jump') || lowerName.includes('hop')) {
      return 'jump';
    }
    if (lowerName.includes('walk') || lowerName.includes('run')) {
      return 'locomotion';
    }
    if (lowerName.includes('look') || lowerName.includes('turn') || lowerName.includes('head')) {
      return 'attention';
    }
    if (lowerName.includes('clap') || lowerName.includes('applaud')) {
      return 'celebration';
    }

    return 'gesture';
  }, []);

  // Discover and categorize available animations
  useEffect(() => {
    const hasAnimations = clips && clips.length > 0;
    setHasGLTFAnimations(hasAnimations);

    if (!hasAnimations) {
      if (debug) console.log('No GLTF animations found, using procedural animations');

      // Create procedural animation list
      if (enableProceduralAnimations) {
        const proceduralAnimations = [
          { name: 'idle', duration: 2, tracks: 1, type: 'idle' },
          { name: 'hover', duration: 1, tracks: 1, type: 'attention' },
          { name: 'click', duration: 0.5, tracks: 1, type: 'gesture' },
          { name: 'wave', duration: 1.5, tracks: 1, type: 'greeting' },
          { name: 'bounce', duration: 1, tracks: 1, type: 'celebration' }
        ];
        setAvailableAnimations(proceduralAnimations);
        if (debug) console.log('Using procedural animations:', proceduralAnimations);
      }
      return;
    }

    const animations = clips.map(clip => ({
      name: clip.name,
      duration: clip.duration,
      tracks: clip.tracks.length,
      type: categorizeAnimation(clip.name)
    }));

    setAvailableAnimations(animations);

    if (debug) {
      console.log('Available GLTF animations:', animations);
      console.log('Animation actions:', Object.keys(actions));
    }

    // Auto-detect idle animation if not specified
    if (!idleAnimationName && animations.length > 0) {
      const idleAnim = animations.find(anim =>
        anim.type === 'idle' ||
        anim.name.toLowerCase().includes('idle') ||
        anim.name.toLowerCase().includes('breathing') ||
        anim.name.toLowerCase().includes('rest')
      ) || animations[0]; // Fallback to first animation

      if (actions[idleAnim.name]) {
        idleActionRef.current = actions[idleAnim.name];
        if (debug) console.log('Auto-detected idle animation:', idleAnim.name);
      }
    } else if (idleAnimationName && actions[idleAnimationName]) {
      idleActionRef.current = actions[idleAnimationName];
    }
  }, [clips, actions, idleAnimationName, debug, enableProceduralAnimations, categorizeAnimation]);

  // Procedural animation functions
  const playProceduralAnimation = useCallback((animationName, duration = 1000) => {
    if (!meshRef.current || !enableProceduralAnimations) return false;

    // Store initial state
    if (meshRef.current.rotation) {
      baseRotationRef.current = [
        meshRef.current.rotation.x,
        meshRef.current.rotation.y,
        meshRef.current.rotation.z
      ];
    }
    if (meshRef.current.position) {
      basePositionRef.current = [
        meshRef.current.position.x,
        meshRef.current.position.y,
        meshRef.current.position.z
      ];
    }

    animationStartTimeRef.current = Date.now();

    const animate = () => {
      if (!meshRef.current) return;

      const elapsed = Date.now() - animationStartTimeRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const easeProgress = 0.5 - 0.5 * Math.cos(progress * Math.PI); // Ease in-out

      switch (animationName) {
        case 'idle':
          // Gentle breathing motion
          const breathe = Math.sin(elapsed * 0.003) * 0.02;
          meshRef.current.position.y = basePositionRef.current[1] + breathe;
          break;

        case 'hover':
          // Slight upward movement and gentle rotation
          const hoverY = Math.sin(elapsed * 0.005) * 0.1;
          const hoverRotY = Math.sin(elapsed * 0.004) * 0.1;
          meshRef.current.position.y = basePositionRef.current[1] + hoverY;
          meshRef.current.rotation.y = baseRotationRef.current[1] + hoverRotY;
          break;

        case 'click':
          // Quick scale pulse
          const pulse = 1 + Math.sin(progress * Math.PI * 2) * 0.1;
          meshRef.current.scale.setScalar(pulse);
          break;

        case 'wave':
          // Rotation wave motion
          const waveRotZ = Math.sin(progress * Math.PI * 4) * 0.3;
          meshRef.current.rotation.z = baseRotationRef.current[2] + waveRotZ;
          break;

        case 'bounce':
          // Bouncing motion
          const bounceY = Math.abs(Math.sin(progress * Math.PI * 3)) * 0.3;
          meshRef.current.position.y = basePositionRef.current[1] + bounceY;
          break;
      }

      if (progress < 1) {
        proceduralAnimationRef.current = requestAnimationFrame(animate);
      } else {
        // Reset to base state
        if (animationName !== 'idle') {
          meshRef.current.position.set(...basePositionRef.current);
          meshRef.current.rotation.set(...baseRotationRef.current);
          meshRef.current.scale.setScalar(1);
        }
        proceduralAnimationRef.current = null;
        setCurrentAnimation(null);
      }
    };

    // Cancel any existing procedural animation
    if (proceduralAnimationRef.current) {
      cancelAnimationFrame(proceduralAnimationRef.current);
    }

    setCurrentAnimation(animationName);
    proceduralAnimationRef.current = requestAnimationFrame(animate);

    if (debug) console.log(`Playing procedural animation: ${animationName}`);
    return true;
  }, [meshRef, enableProceduralAnimations, debug]);

  // Play animation with smooth transition
  const playAnimation = useCallback((animationName, options = {}) => {
    // If we have GLTF animations, try to use them first
    if (hasGLTFAnimations && actions[animationName]) {
      const {
        loop = THREE.LoopRepeat,
        crossfade = true,
        duration = transitionDuration,
        onComplete = null
      } = options;

      const newAction = actions[animationName];

      // Store previous action for crossfading
      if (currentActionRef.current && crossfade) {
        previousActionRef.current = currentActionRef.current;
      }

      // Configure new action
      newAction.reset();
      newAction.setLoop(loop);
      newAction.clampWhenFinished = true;

      // Crossfade or direct play
      if (previousActionRef.current && crossfade) {
        newAction.crossFadeFrom(previousActionRef.current, duration, true);
      } else {
        newAction.play();
      }

      currentActionRef.current = newAction;
      setCurrentAnimation(animationName);

      // Handle animation completion
      if (onComplete || loop === THREE.LoopOnce) {
        const mixer = newAction.getMixer();
        const onFinished = () => {
          if (onComplete) onComplete();
          if (loop === THREE.LoopOnce) {
            // Return to idle after one-shot animations
            setTimeout(() => playIdleAnimation(), 100);
          }
          mixer.removeEventListener('finished', onFinished);
        };
        mixer.addEventListener('finished', onFinished);
      }

      if (debug) console.log(`Playing GLTF animation: ${animationName}`);
      return true;
    }

    // Fallback to procedural animations
    if (enableProceduralAnimations) {
      const animDuration = options.duration || (animationName === 'idle' ? 2000 : 1000);
      return playProceduralAnimation(animationName, animDuration);
    }

    if (debug) console.warn(`Animation "${animationName}" not found and procedural animations disabled`);
    return false;
  }, [hasGLTFAnimations, actions, transitionDuration, debug, enableProceduralAnimations, playProceduralAnimation]);

  // Play idle animation
  const playIdleAnimation = useCallback(() => {
    if (hasGLTFAnimations && idleActionRef.current) {
      const idleName = availableAnimations.find(anim =>
        actions[anim.name] === idleActionRef.current
      )?.name;

      if (idleName) {
        playAnimation(idleName, { loop: THREE.LoopRepeat });
        return;
      }
    }

    // Fallback to procedural idle animation
    if (enableProceduralAnimations) {
      playAnimation('idle', { duration: 2000 });
    }
  }, [hasGLTFAnimations, availableAnimations, actions, playAnimation, enableProceduralAnimations]);

  // Get random animation by type
  const getRandomAnimationByType = useCallback((type) => {
    const animsOfType = availableAnimations.filter(anim => anim.type === type);
    if (animsOfType.length === 0) return null;
    
    const randomIndex = Math.floor(Math.random() * animsOfType.length);
    return animsOfType[randomIndex].name;
  }, [availableAnimations]);

  // Mouse interaction handlers
  const handleMouseEnter = useCallback((event) => {
    if (!enableInteractions) return;
    
    setIsHovered(true);
    
    // Clear any pending idle return
    if (animationTimeoutRef.current) {
      clearTimeout(animationTimeoutRef.current);
    }

    // Try to play attention/look animation
    const attentionAnim = getRandomAnimationByType('attention');
    if (attentionAnim) {
      playAnimation(attentionAnim, {
        loop: hasGLTFAnimations ? THREE.LoopRepeat : undefined,
        crossfade: hasGLTFAnimations,
        duration: hasGLTFAnimations ? undefined : 1000
      });
    } else if (enableProceduralAnimations) {
      // Fallback to procedural hover animation
      playAnimation('hover', { duration: 1000 });
    }

    if (debug) console.log('Mouse entered avatar');
  }, [enableInteractions, getRandomAnimationByType, playAnimation, debug]);

  const handleMouseLeave = useCallback((event) => {
    if (!enableInteractions) return;
    
    setIsHovered(false);
    
    // Return to idle after a short delay
    animationTimeoutRef.current = setTimeout(() => {
      playIdleAnimation();
    }, 1000);

    if (debug) console.log('Mouse left avatar');
  }, [enableInteractions, playIdleAnimation, debug]);

  const handleClick = useCallback((event) => {
    if (!enableInteractions) return;
    
    setIsClicked(true);
    
    // Clear any pending idle return
    if (animationTimeoutRef.current) {
      clearTimeout(animationTimeoutRef.current);
    }

    // Try different types of click animations
    const clickAnimTypes = ['greeting', 'celebration', 'dance', 'gesture'];
    let clickAnim = null;
    
    for (const type of clickAnimTypes) {
      clickAnim = getRandomAnimationByType(type);
      if (clickAnim) break;
    }

    if (clickAnim) {
      playAnimation(clickAnim, {
        loop: hasGLTFAnimations ? THREE.LoopOnce : undefined,
        crossfade: hasGLTFAnimations,
        duration: hasGLTFAnimations ? undefined : 500,
        onComplete: () => {
          setIsClicked(false);
          // Return to idle after click animation
          setTimeout(() => playIdleAnimation(), 500);
        }
      });
    } else if (enableProceduralAnimations) {
      // Fallback to procedural click animation
      playAnimation('click', { duration: 500 });
      setTimeout(() => {
        setIsClicked(false);
        playIdleAnimation();
      }, 600);
    } else {
      setIsClicked(false);
    }

    if (debug) console.log('Avatar clicked');
  }, [enableInteractions, getRandomAnimationByType, playAnimation, playIdleAnimation, debug]);

  // Start idle animation on mount
  useEffect(() => {
    if (idleActionRef.current && !currentAnimation) {
      playIdleAnimation();
    }
  }, [idleActionRef.current, currentAnimation, playIdleAnimation]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationTimeoutRef.current) {
        clearTimeout(animationTimeoutRef.current);
      }
      if (proceduralAnimationRef.current) {
        cancelAnimationFrame(proceduralAnimationRef.current);
      }
    };
  }, []);

  return {
    // Animation state
    currentAnimation,
    availableAnimations,
    isHovered,
    isClicked,
    
    // Animation controls
    playAnimation,
    playIdleAnimation,
    getRandomAnimationByType,
    
    // Mouse handlers
    handleMouseEnter,
    handleMouseLeave,
    handleClick,
    
    // Raw animation data
    actions,
    names,
    clips
  };
}

export default useAvatarAnimations;
