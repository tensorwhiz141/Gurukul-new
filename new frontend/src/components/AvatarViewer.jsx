import React, {
  Suspense,
  useRef,
  useState,
  useEffect,
  Component,
  memo,
  useMemo,
} from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  OrbitControls,
  useGLTF,
  Environment,
  ContactShadows,
  PresentationControls,
  Float,
  Text,
  Html,
} from "@react-three/drei";
import useAvatarAnimations from "../hooks/useAvatarAnimations.js";
import { useModelLoadingPerformance } from "../hooks/usePerformanceMonitor";

// Error Boundary for Avatar Viewer
class AvatarErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Avatar Viewer Error:", error, errorInfo);
    if (this.props.onError) {
      this.props.onError(error);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-full bg-black/20 text-white">
          <div className="text-center">
            <p className="mb-2">Failed to load avatar</p>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="px-4 py-2 bg-blue-500 rounded hover:bg-blue-600"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Optimized Avatar Model Loader with performance monitoring and memory management
const AvatarModel = memo(function AvatarModel({
  url,
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  scale = 1,
  autoRotate = false,
  enableAnimations = true,
  enableInteractions = true,
  onError = null,
  onLoad = null,
  ...props
}) {
  const [loadTimeout, setLoadTimeout] = useState(false);

  // Performance monitoring for model loading
  const { startLoading, endLoading } = useModelLoadingPerformance(url);

  // Special handling for default avatars (jupiter.glb)
  const isDefaultAvatar = useMemo(() =>
    url.includes("jupiter.glb"), [url]);

  // Memoize GLTF loading to prevent unnecessary re-loads
  const gltf = useGLTF(url);
  const { scene, error } = gltf;
  const meshRef = useRef();

  // Use refs for base transformations
  const baseRotation = useRef(rotation);
  const basePosition = useRef(position);
  const baseScale = useRef(scale);

  // Log default avatar loading specifically
  useEffect(() => {
    if (isDefaultAvatar) {
      console.log("🎭 Default avatar loading started:", url);
      console.log("🎭 GLTF object:", gltf);
      console.log("🎭 Scene:", scene);
      console.log("🎭 Error:", error);
    }
  }, [isDefaultAvatar, url, gltf, scene, error]);

  // Initialize avatar animations - only if we have a valid scene
  const {
    availableAnimations,
    handleMouseEnter,
    handleMouseLeave,
    handleClick,
  } = useAvatarAnimations(scene ? gltf : null, meshRef, {
    enableInteractions: enableInteractions && enableAnimations && !!scene,
    debug: url.includes("jupiter.glb"), // Enable debug for jupiter avatar
    enableProceduralAnimations: true,
  });

  // Set a timeout for loading - longer timeout for large files like jupiter.glb
  useEffect(() => {
    const timeoutDuration = url.includes("jupiter.glb") ? 15000 : 5000; // 15 seconds for jupiter.glb
    const timer = setTimeout(() => {
      if (!scene && !error) {
        console.warn(
          `GLB loading timeout for: ${url} (waited ${timeoutDuration}ms)`
        );
        setLoadTimeout(true);
      }
    }, timeoutDuration);

    return () => clearTimeout(timer);
  }, [url, scene, error]);

  // Log animation info for debugging
  useEffect(() => {
    if (availableAnimations.length > 0) {
      console.log(
        `Avatar ${url} has ${availableAnimations.length} animations:`,
        availableAnimations.map((a) => a.name)
      );
    }
  }, [availableAnimations, url]);

  // Update base values when props change - immediate updates for smooth controls
  useEffect(() => {
    baseRotation.current = rotation;
    // Force immediate update if mesh is available
    if (meshRef.current) {
      meshRef.current.rotation.x = rotation[0];
      meshRef.current.rotation.y = rotation[1];
      meshRef.current.rotation.z = rotation[2];
    }
  }, [rotation]);

  useEffect(() => {
    basePosition.current = position;
    // Force immediate update if mesh is available
    if (meshRef.current) {
      meshRef.current.position.set(position[0], position[1], position[2]);
    }
  }, [position]);

  useEffect(() => {
    baseScale.current = scale;
    // Force immediate update if mesh is available
    if (meshRef.current) {
      meshRef.current.scale.set(scale, scale, scale);
    }
  }, [scale]);

  // Performance monitoring and loading status
  useEffect(() => {
    if (scene && !error) {
      // Model loaded successfully
      endLoading(true);
      if (onLoad) onLoad();

      if (isDefaultAvatar) {
        console.log("🎭 Default avatar loaded successfully:", url);
      }
    } else if (error) {
      // Model failed to load
      endLoading(false);
      if (onError) onError(error);

      console.error("AvatarModel error details:", error);
      console.error("Failed URL:", url);
      console.error("Error type:", error.constructor.name);
      console.error("Error message:", error.message);
    } else {
      // Model is loading
      startLoading();

      if (isDefaultAvatar) {
        console.log("🎭 Default avatar loading started:", url);
      }
    }
  }, [
    scene,
    error,
    isDefaultAvatar,
    url,
    startLoading,
    endLoading,
    onLoad,
    onError,
  ]);

  useFrame((state) => {
    if (meshRef.current) {
      // Update position
      meshRef.current.position.set(
        basePosition.current[0],
        basePosition.current[1],
        basePosition.current[2]
      );

      // Update scale
      meshRef.current.scale.set(
        baseScale.current,
        baseScale.current,
        baseScale.current
      );

      // Update rotation
      if (autoRotate) {
        // Apply gentle auto-rotation on top of base rotation for avatars
        meshRef.current.rotation.x = baseRotation.current[0];
        meshRef.current.rotation.y =
          baseRotation.current[1] +
          Math.sin(state.clock.elapsedTime * 0.3) * 0.05;
        meshRef.current.rotation.z = baseRotation.current[2];
      } else {
        // Apply only the manual rotation
        meshRef.current.rotation.x = baseRotation.current[0];
        meshRef.current.rotation.y = baseRotation.current[1];
        meshRef.current.rotation.z = baseRotation.current[2];
      }
    }
  });

  // If there's an error loading the model, return null to trigger fallback
  if (error || loadTimeout) {
    console.error("Failed to load avatar model:", url, error || "Timeout");
    console.error("Error details:", {
      error,
      loadTimeout,
      url,
      scene: !!scene,
      gltf,
    });
    return null;
  }

  // If scene is not loaded yet, return null (loading state)
  if (!scene) {
    return null;
  }

  return (
    <primitive
      ref={meshRef}
      object={scene}
      onPointerEnter={
        enableAnimations && enableInteractions ? handleMouseEnter : undefined
      }
      onPointerLeave={
        enableAnimations && enableInteractions ? handleMouseLeave : undefined
      }
      onClick={enableAnimations && enableInteractions ? handleClick : undefined}
      {...props}
    />
  );
});

// Avatar fallback component
const AvatarFallback = memo(function AvatarFallback({ message = "Avatar" }) {
  return (
    <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.3}>
      {/* Simple avatar representation */}
      <group position={[0, -0.5, 0]}>
        {/* Head */}
        <mesh position={[0, 0.8, 0]}>
          <sphereGeometry args={[0.2, 16, 16]} />
          <meshStandardMaterial
            color="#FFB366"
            emissive="#FFB366"
            emissiveIntensity={0.1}
          />
        </mesh>

        {/* Eyes */}
        <mesh position={[-0.08, 0.85, 0.15]}>
          <sphereGeometry args={[0.03, 8, 8]} />
          <meshStandardMaterial color="#2C3E50" />
        </mesh>
        <mesh position={[0.08, 0.85, 0.15]}>
          <sphereGeometry args={[0.03, 8, 8]} />
          <meshStandardMaterial color="#2C3E50" />
        </mesh>

        {/* Nose */}
        <mesh position={[0, 0.8, 0.18]}>
          <sphereGeometry args={[0.02, 8, 8]} />
          <meshStandardMaterial color="#E8A87C" />
        </mesh>

        {/* Body */}
        <mesh position={[0, 0.3, 0]}>
          <cylinderGeometry args={[0.12, 0.18, 0.5, 8]} />
          <meshStandardMaterial
            color="#4A90E2"
            emissive="#4A90E2"
            emissiveIntensity={0.05}
          />
        </mesh>

        {/* Arms */}
        <mesh position={[-0.25, 0.4, 0]} rotation={[0, 0, 0.3]}>
          <cylinderGeometry args={[0.04, 0.04, 0.35, 8]} />
          <meshStandardMaterial
            color="#FFB366"
            emissive="#FFB366"
            emissiveIntensity={0.05}
          />
        </mesh>
        <mesh position={[0.25, 0.4, 0]} rotation={[0, 0, -0.3]}>
          <cylinderGeometry args={[0.04, 0.04, 0.35, 8]} />
          <meshStandardMaterial
            color="#FFB366"
            emissive="#FFB366"
            emissiveIntensity={0.05}
          />
        </mesh>

        {/* Hands */}
        <mesh position={[-0.35, 0.15, 0]}>
          <sphereGeometry args={[0.06, 8, 8]} />
          <meshStandardMaterial
            color="#FFB366"
            emissive="#FFB366"
            emissiveIntensity={0.05}
          />
        </mesh>
        <mesh position={[0.35, 0.15, 0]}>
          <sphereGeometry args={[0.06, 8, 8]} />
          <meshStandardMaterial
            color="#FFB366"
            emissive="#FFB366"
            emissiveIntensity={0.05}
          />
        </mesh>

        {/* Legs */}
        <mesh position={[-0.1, -0.15, 0]}>
          <cylinderGeometry args={[0.05, 0.05, 0.45, 8]} />
          <meshStandardMaterial
            color="#2C3E50"
            emissive="#2C3E50"
            emissiveIntensity={0.05}
          />
        </mesh>
        <mesh position={[0.1, -0.15, 0]}>
          <cylinderGeometry args={[0.05, 0.05, 0.45, 8]} />
          <meshStandardMaterial
            color="#2C3E50"
            emissive="#2C3E50"
            emissiveIntensity={0.05}
          />
        </mesh>

        {/* Feet */}
        <mesh position={[-0.1, -0.45, 0.08]}>
          <boxGeometry args={[0.08, 0.06, 0.16]} />
          <meshStandardMaterial color="#1A1A1A" />
        </mesh>
        <mesh position={[0.1, -0.45, 0.08]}>
          <boxGeometry args={[0.08, 0.06, 0.16]} />
          <meshStandardMaterial color="#1A1A1A" />
        </mesh>
      </group>

      {/* Label */}
      <Text
        fontSize={0.15}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        position={[0, 0.8, 0]}
        outlineWidth={0.02}
        outlineColor="#000000"
      >
        {message}
      </Text>
    </Float>
  );
});

// Loading component for avatars
const AvatarLoading = memo(function AvatarLoading({ isLargeFile = false }) {
  return (
    <Html center>
      <div className="flex flex-col items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-400"></div>
        <span className="ml-2 text-white">
          {isLargeFile ? "Loading Guru..." : "Loading Guru..."}
        </span>
        {isLargeFile && (
          <span className="text-xs text-white/70 mt-1">
            This may take a moment
          </span>
        )}
      </div>
    </Html>
  );
});

/**
 * Dedicated Avatar Viewer Component
 * Optimized specifically for avatar display and interaction
 */
const AvatarViewer = memo(function AvatarViewer({
  avatarPath,
  fallbackPath = null, // New prop for fallback avatar path
  className = "",
  enableControls = true,
  autoRotate = false,
  autoRotateSpeed = null, // Custom rotation speed
  showEnvironment = true,
  environmentPreset = "sunset",
  fallbackMessage = "Avatar",
  position = [0, 0, 0],
  rotation = [0, Math.PI, 0], // Default to face forward (180 degrees Y)
  scale = 1,
  cameraPosition = [0, 0, 4],
  enableInteraction = true,
  enableAnimations = true, // New prop for animations
  isSpeaking = false, // New prop for speaking state
  onError = null,
  onLoad = null, // New prop for load success callback
  onLoadStart = null, // New prop for load start callback
  style = "realistic", // realistic, cartoon, stylized
}) {
  const [avatarError, setAvatarError] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [currentPath, setCurrentPath] = useState(avatarPath);
  const [usingFallback, setUsingFallback] = useState(false);
  const maxRetries = 2;

  // Reduced debug logging to prevent performance issues
  useEffect(() => {
    // Only log on initial load or path changes
    if (avatarPath !== currentPath) {
      console.log("AvatarViewer loading:", avatarPath);
    }

    // Log procedural avatar handling
    if (avatarPath?.startsWith("procedural://")) {
      console.log("🎭 Rendering procedural avatar:", avatarPath);
    }
  }, [avatarPath, currentPath]);

  // Reset error state and retry count when avatar path changes
  useEffect(() => {
    setCurrentPath(avatarPath);
    setUsingFallback(false);
    setAvatarError(false);
    setRetryCount(0);

    // Trigger load start callback
    if (onLoadStart) {
      onLoadStart();
    }
  }, [avatarPath, onLoadStart]);

  // Avatar-specific lighting setup
  const getLighting = () => {
    switch (style) {
      case "cartoon":
        return {
          ambient: 0.6,
          directional: 0.8,
          point: 0.3,
        };
      case "stylized":
        return {
          ambient: 0.5,
          directional: 1.0,
          point: 0.4,
        };
      default: // realistic
        return {
          ambient: 0.4,
          directional: 1.2,
          point: 0.5,
        };
    }
  };

  const lighting = getLighting();

  return (
    <div className={`w-full h-full ${className} ${isSpeaking ? 'avatar-speaking' : ''}`}>
      <AvatarErrorBoundary
        onError={(error) => {
          console.error("Avatar Error:", error);
          console.error("Error occurred for path:", avatarPath);
          console.error("Retry count:", retryCount);

          if (retryCount < maxRetries) {
            console.log(
              `Retrying avatar load (attempt ${retryCount + 1}/${maxRetries})`
            );
            setRetryCount((prev) => prev + 1);
            // Reset error state to trigger retry
            setTimeout(() => {
              setAvatarError(false);
            }, 1000);
          } else if (fallbackPath && !usingFallback) {
            console.log(
              "Max retries reached, trying fallback path:",
              fallbackPath
            );
            setCurrentPath(fallbackPath);
            setUsingFallback(true);
            setRetryCount(0);
            setAvatarError(false);
          } else {
            console.error(
              "Max retries reached and no fallback available, showing error"
            );
            setAvatarError(true);
          }

          if (onError) onError(error);
        }}
      >
        <Canvas
          camera={{ position: cameraPosition, fov: 50 }}
          style={{ background: "transparent" }}
        >
          {/* Avatar-optimized lighting */}
          <ambientLight intensity={lighting.ambient} />
          <directionalLight
            position={[5, 10, 5]}
            intensity={lighting.directional}
            castShadow
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
          />
          <pointLight position={[-5, 5, 5]} intensity={lighting.point} />
          <pointLight position={[5, -5, -5]} intensity={lighting.point * 0.5} />

          {/* Environment for avatars */}
          {showEnvironment && (
            <Environment preset={environmentPreset} background={false} />
          )}

          <Suspense
            fallback={
              <AvatarLoading
                isLargeFile={currentPath?.includes("jupiter.glb")}
              />
            }
          >
            {currentPath && !avatarError && currentPath.trim() !== "" ? (
              enableControls ? (
                <PresentationControls
                  enabled={enableInteraction}
                  global={false}
                  cursor={true}
                  snap={false}
                  speed={1.2}
                  zoom={1.2}
                  rotation={[0, 0, 0]}
                  polar={[-Math.PI / 4, Math.PI / 4]}
                  azimuth={[-Math.PI, Math.PI]}
                >
                  <AvatarModel
                    url={currentPath}
                    position={position}
                    rotation={rotation}
                    scale={scale}
                    autoRotate={autoRotate}
                    enableAnimations={enableAnimations}
                    enableInteractions={enableInteraction}
                    onLoad={onLoad}
                    onError={(error) => {
                      console.error(
                        "Avatar loading error (with controls):",
                        error
                      );
                      console.error("Avatar path:", currentPath);
                      setAvatarError(true);
                      if (onError) onError(error);
                    }}
                  />
                </PresentationControls>
              ) : (
                <AvatarModel
                  url={currentPath}
                  position={position}
                  rotation={rotation}
                  scale={scale}
                  autoRotate={autoRotate}
                  enableAnimations={enableAnimations}
                  enableInteractions={enableInteraction}
                  onLoad={onLoad}
                  onError={(error) => {
                    console.error(
                      "Avatar loading error (without controls):",
                      error
                    );
                    console.error("Avatar path:", currentPath);
                    setAvatarError(true);
                    if (onError) onError(error);
                  }}
                />
              )
            ) : (
              <AvatarFallback message={fallbackMessage} />
            )}
          </Suspense>

          {/* Avatar-specific controls */}
          {enableControls && enableInteraction && (
            <OrbitControls
              enablePan={false}
              enableZoom={true}
              enableRotate={true}
              autoRotate={avatarPath.includes('jupiter.glb') ? autoRotate : (autoRotate && !enableInteraction)}
              autoRotateSpeed={autoRotateSpeed || (avatarPath.includes('jupiter.glb') ? 2.0 : 0.3)}
              minDistance={2}
              maxDistance={8}
              minPolarAngle={Math.PI / 6}
              maxPolarAngle={Math.PI - Math.PI / 6}
              target={[0, 0, 0]}
            />
          )}

          {/* Subtle shadows for avatars - exclude jupiter.glb */}
          {!avatarPath.includes('jupiter.glb') && (
            <ContactShadows
              position={[0, -1.2, 0]}
              opacity={0.3}
              scale={8}
              blur={2}
              far={3}
            />
          )}
        </Canvas>
      </AvatarErrorBoundary>
    </div>
  );
});

// Preload common avatar models
useGLTF.preload("/avatar/jupiter.glb");

export default AvatarViewer;
