import React, { Suspense, useRef, useState, Component } from "react";
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

// Simple Error Boundary Component
class SimpleErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('3D Model Error:', error, errorInfo);
    if (this.props.onError) {
      this.props.onError(error);
    }
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || this.props.FallbackComponent
        ? React.createElement(this.props.FallbackComponent, {
            error: this.state.error,
            resetErrorBoundary: () => this.setState({ hasError: false, error: null })
          })
        : <div className="flex items-center justify-center h-full bg-black/20 text-white">
            <div className="text-center">
              <p className="mb-2">Failed to load Guru</p>
              <button
                onClick={() => this.setState({ hasError: false, error: null })}
                className="px-4 py-2 bg-orange-500 rounded hover:bg-orange-600"
              >
                Retry
              </button>
            </div>
          </div>;
    }

    return this.props.children;
  }
}

// Loading component for 3D models
function ModelLoader({ url, position = [0, 0, 0], rotation = [0, 0, 0], scale = 1, autoRotate = false, ...props }) {
  const { scene } = useGLTF(url);
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current && autoRotate) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  return (
    <primitive
      ref={meshRef}
      object={scene}
      position={position}
      rotation={rotation}
      scale={scale}
      {...props}
    />
  );
}

// Fallback component for when models fail to load
function ModelFallback({ message = "3D Model" }) {
  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
      <Text
        fontSize={0.5}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        position={[0, 0, 0]}
      >
        {message}
      </Text>
      <mesh position={[0, -0.5, 0]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#3B82F6" />
      </mesh>
    </Float>
  );
}

// Error fallback component
function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <div className="flex items-center justify-center h-full bg-black/20 text-white">
      <div className="text-center">
        <p className="mb-2">Failed to load Guru</p>
        <button
          onClick={resetErrorBoundary}
          className="px-4 py-2 bg-orange-500 rounded hover:bg-orange-600"
        >
          Retry
        </button>
      </div>
    </div>
  );
}

// Loading fallback
function LoadingFallback() {
  return (
    <Html center>
      <div className="flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-400"></div>
        <span className="ml-2 text-white">Loading Guru...</span>
      </div>
    </Html>
  );
}

/**
 * Enhanced ModelViewer with React Three Fiber
 * Supports loading 3D models with proper error handling and fallbacks
 */
export default function SimpleModelViewer({
  modelPath,
  className = "",
  enableControls = true,
  autoRotate = false,
  showEnvironment = true,
  environmentPreset = "sunset",
  fallbackMessage = "3D Model",
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  scale = 1
}) {
  const [modelError, setModelError] = useState(false);

  return (
    <div className={`w-full h-full ${className}`}>
      <SimpleErrorBoundary
        FallbackComponent={ErrorFallback}
        onError={(error) => {
          console.error("3D Model Error:", error);
          setModelError(true);
        }}
      >
        <Canvas
          camera={{ position: [0, 0, 5], fov: 45 }}
          style={{ background: "transparent" }}
        >
          {showEnvironment && (
            <Environment preset={environmentPreset} background={false} />
          )}

          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <pointLight position={[-10, -10, -10]} intensity={0.5} />

          <Suspense fallback={<LoadingFallback />}>
            {modelPath && !modelError ? (
              <PresentationControls
                enabled={enableControls}
                global={false}
                cursor={true}
                snap={false}
                speed={1}
                zoom={1}
                rotation={[0, 0, 0]}
                polar={[-Math.PI / 3, Math.PI / 3]}
                azimuth={[-Math.PI / 1.4, Math.PI / 1.4]}
              >
                <ModelLoader
                  url={modelPath}
                  position={position}
                  rotation={rotation}
                  scale={scale}
                  autoRotate={autoRotate}
                />
              </PresentationControls>
            ) : (
              <ModelFallback message={fallbackMessage} />
            )}
          </Suspense>

          {enableControls && (
            <OrbitControls
              enablePan={false}
              enableZoom={true}
              enableRotate={true}
              autoRotate={autoRotate}
              autoRotateSpeed={0.5}
              minDistance={2}
              maxDistance={10}
              minPolarAngle={Math.PI / 6}
              maxPolarAngle={Math.PI - Math.PI / 6}
            />
          )}

          <ContactShadows
            position={[0, -1.4, 0]}
            opacity={0.4}
            scale={10}
            blur={2.5}
            far={4.5}
          />
        </Canvas>
      </SimpleErrorBoundary>
    </div>
  );
}
