import React, {
  memo,
  useMemo,
  useCallback,
  useState,
  useEffect,
  useRef,
  Suspense,
} from "react";
import { Heart } from "lucide-react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Html, useGLTF } from "@react-three/drei";
import { AvatarGridSkeleton, ModelLoadingSkeleton } from "./LoadingSkeletons";
import { useAvatarAnimations } from "../hooks/useAvatarAnimations";

// Simple 3D Model Component for Pin Mode with real-time position updates
function SimplePinModel({
  url,
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  scale = 1,
  onLoad,
  onError,
  enableAnimations = false,
}) {
  const gltf = useGLTF(url, true);
  const { scene, error } = gltf;
  const meshRef = useRef();

  // Enable animations for jupiter.glb
  const shouldEnableAnimations = enableAnimations && url.includes('jupiter.glb');

  // Initialize avatar animations for jupiter.glb
  const {
    availableAnimations,
    handleMouseEnter,
    handleMouseLeave,
    handleClick,
  } = useAvatarAnimations(shouldEnableAnimations ? gltf : null, meshRef, {
    enableInteractions: shouldEnableAnimations,
    debug: url.includes("jupiter.glb"),
    enableProceduralAnimations: true,
  });

  useEffect(() => {
    if (scene && onLoad) {

      onLoad();
    }
  }, [scene, onLoad, url]);

  useEffect(() => {
    if (error && onError) {
      console.error(`SimplePinModel load error:`, error);
      onError(error);
    }
  }, [error, onError]);

  // Immediate position updates for smooth controls
  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.position.set(position[0], position[1], position[2]);
    }
  }, [position]);

  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.rotation.set(rotation[0], rotation[1], rotation[2]);
    }
  }, [rotation]);

  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.scale.set(scale, scale, scale);
    }
  }, [scale]);

  // Auto-rotation for jupiter.glb
  useFrame((state, delta) => {
    if (shouldEnableAnimations && meshRef.current) {
      meshRef.current.rotation.y += delta * 2.0; // Rotate at 2.0 radians per second (4x faster)
    }
  });

  if (error) {
    return null;
  }

  if (!scene) {
    return null;
  }

  return (
    <primitive
      ref={meshRef}
      object={scene}
      position={position}
      rotation={rotation}
      scale={[scale, scale, scale]}
      onPointerEnter={shouldEnableAnimations ? handleMouseEnter : undefined}
      onPointerLeave={shouldEnableAnimations ? handleMouseLeave : undefined}
      onClick={shouldEnableAnimations ? handleClick : undefined}
    />
  );
}

/**
 * Dedicated Pin Mode Component
 * Handles avatar display in pin mode layout with preview functionality
 */
const PinModeComponent = memo(
  ({
    favorites,
    selectedAvatar,
    pinPosition,
    pinRotation,
    pinScale,
    onSelectFavorite,
    onDeleteFavorite,
    isLoading = false,
    enableVirtualization = false,
    maxVisibleItems = 20,
  }) => {

    // Auto-select default avatar if none selected (for Pin Mode)
    useEffect(() => {
      if (!selectedAvatar && favorites.length > 0 && onSelectFavorite) {
        const defaultAvatar =
          favorites.find((f) => f.isDefault) || favorites[0];
        console.log("🎭 Pin mode auto-selecting avatar:", defaultAvatar.name);
        onSelectFavorite(defaultAvatar);
      }
    }, [selectedAvatar, favorites, onSelectFavorite]);

    console.log("🎭 Pin mode state:", {
      selectedAvatar: selectedAvatar?.name || "None",
      favoritesCount: favorites.length,
      hasOnSelectFavorite: !!onSelectFavorite,
    });
    const [visibleRange, setVisibleRange] = useState({
      start: 0,
      end: maxVisibleItems,
    });
    const [isInitialLoad, setIsInitialLoad] = useState(true);

    // Handle initial loading state
    useEffect(() => {
      if (favorites.length > 0) {
        const timer = setTimeout(() => setIsInitialLoad(false), 100);
        return () => clearTimeout(timer);
      }
    }, [favorites.length]);

    // Grid configuration for Pin Mode (smaller cards)
    const gridConfig = {
      columns: "grid-cols-3 md:grid-cols-4",
      gap: "gap-3",
      size: "small",
    };

    // Memoize visible favorites for virtualization
    const visibleFavorites = useMemo(() => {
      if (!enableVirtualization || favorites.length <= maxVisibleItems) {
        return favorites;
      }
      return favorites.slice(visibleRange.start, visibleRange.end);
    }, [favorites, enableVirtualization, maxVisibleItems, visibleRange]);

    // Optimized select handler
    const handleSelectFavorite = useCallback(
      (favorite) => {
        onSelectFavorite(favorite);
      },
      [onSelectFavorite]
    );

    // Optimized delete handler
    const handleDeleteFavorite = useCallback(
      (favoriteId) => {
        onDeleteFavorite(favoriteId);
      },
      [onDeleteFavorite]
    );

    // Handle scroll for virtualization
    const handleScroll = useCallback(
      (e) => {
        if (!enableVirtualization) return;

        const container = e.target;
        const scrollTop = container.scrollTop;
        const itemHeight = 200; // Match GridViewComponent height
        const containerHeight = container.clientHeight;

        const start = Math.floor(scrollTop / itemHeight);
        const end = Math.min(
          start + Math.ceil(containerHeight / itemHeight) + 2,
          favorites.length
        );

        setVisibleRange({ start, end });
      },
      [enableVirtualization, favorites.length]
    );

    // Show loading skeleton during initial load
    if (isLoading || isInitialLoad) {
      return (
        <div className="flex-1 overflow-y-auto min-h-0">
          <AvatarGridSkeleton count={12} columns={4} />
        </div>
      );
    }

    // Show empty state
    if (favorites.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-white/50">
          <Heart className="w-16 h-16 mb-3 opacity-50" />
          <p className="text-lg mb-2">No favorites yet</p>
          <p className="text-sm">
            Generate an avatar and save it as a favorite!
          </p>
        </div>
      );
    }

    return (
      <div className="flex-1 overflow-y-auto min-h-0" onScroll={handleScroll}>
        <div className={`${gridConfig.columns} ${gridConfig.gap} grid`}>
          {visibleFavorites.map((favorite) => (
            <PinAvatarCard
              key={`pin-${favorite.id}`}
              favorite={favorite}
              isSelected={selectedAvatar?.id === favorite.id}
              pinPosition={pinPosition}
              pinRotation={pinRotation}
              pinScale={pinScale}
              onSelect={handleSelectFavorite}
              onDelete={handleDeleteFavorite}
            />
          ))}
        </div>

        {/* Virtualization spacer */}
        {enableVirtualization && favorites.length > maxVisibleItems && (
          <div
            style={{
              height: `${(favorites.length - visibleFavorites.length) * 200}px`,
            }}
            className="w-full"
          />
        )}
      </div>
    );
  }
);

/**
 * Pin Avatar Card Component
 * Optimized for Pin Mode display with preview functionality
 */
const PinAvatarCard = memo(
  ({
    favorite,
    isSelected,
    pinPosition,
    pinRotation,
    pinScale,
    onSelect,
    onDelete,
  }) => {
    const [isLoading, setIsLoading] = useState(true);
    const [hasError, setHasError] = useState(false);

    // Determine if this should show as pin preview (default avatar when no avatar selected)
    const shouldShowAsPinPreview = !isSelected && favorite.isDefault;

    // Calculate avatar position for pin mode with real-time updates
    const avatarPosition = useMemo(() => {
      // Avatar-specific base offset
      let baseOffset = 0.3; // Better offset for pin mode cards (jupiter.glb)
      if (favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')) {
        baseOffset = 0.3; // Much higher position for jupiter.glb
      }

      if (isSelected || shouldShowAsPinPreview) {
        // Hardcoded perfect position for jupiter.glb
        if (favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')) {
          return [0, 0.5, -4.0];
        }
        // Use current Redux state for real-time updates
        return [
          Math.max(-1.5, Math.min(1.5, pinPosition?.x || 0)),
          Math.max(-1.5, Math.min(1.0, (pinPosition?.y || 0) + baseOffset)),
          Math.max(-1.5, Math.min(1.5, pinPosition?.z || 0)),
        ];
      } else {
        // Hardcoded perfect position for jupiter.glb
        if (favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')) {
          return [0, 0.5, -4.0];
        }
        // Use saved state for non-selected avatars
        return [
          Math.max(
            -1.5,
            Math.min(
              1.5,
              favorite.pinPosition?.x || favorite.avatarPosition?.x || 0
            )
          ),
          Math.max(
            -1.5,
            Math.min(
              1.0,
              (favorite.pinPosition?.y || favorite.avatarPosition?.y || 0) +
                baseOffset
            )
          ),
          Math.max(
            -1.5,
            Math.min(
              1.5,
              favorite.pinPosition?.z || favorite.avatarPosition?.z || 0
            )
          ),
        ];
      }
    }, [isSelected, shouldShowAsPinPreview, pinPosition, favorite]);

    // Calculate avatar rotation for pin mode
    const avatarRotation = useMemo(() => {
      if (isSelected || shouldShowAsPinPreview) {
        return [
          ((pinRotation?.x || 0) * Math.PI) / 180,
          ((pinRotation?.y || 180) * Math.PI) / 180, // Default to face forward
          ((pinRotation?.z || 0) * Math.PI) / 180,
        ];
      } else {
        return [
          ((favorite.pinRotation?.x || favorite.avatarRotation?.x || 0) *
            Math.PI) /
            180,
          ((favorite.pinRotation?.y || favorite.avatarRotation?.y || 180) *
            Math.PI) /
            180, // Default to face forward
          ((favorite.pinRotation?.z || favorite.avatarRotation?.z || 0) *
            Math.PI) /
            180,
        ];
      }
    }, [isSelected, shouldShowAsPinPreview, pinRotation, favorite]);

    // Calculate avatar scale for pin mode (larger scale for better visibility)
    const avatarScale = useMemo(() => {
      // Avatar-specific default scales
      let defaultScale = 0.8; // Default for jupiter.glb
      if (favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')) {
        defaultScale = 0.8; // Decreased by 0.3 (was 1.1, now 0.8)
      }

      if (isSelected || shouldShowAsPinPreview) {
        return Math.max(2.0, Math.min(5.0, pinScale || defaultScale));
      } else {
        return Math.max(
          2.0,
          Math.min(5.0, favorite.pinScale || favorite.avatarScale || defaultScale)
        );
      }
    }, [isSelected, shouldShowAsPinPreview, pinScale, favorite]);

    // Event handlers
    const handleSelect = useCallback(() => {
      console.log(
        `🎭 Pin mode avatar card clicked: ${favorite.name} (${favorite.id})`
      );
      console.log(`🎭 onSelect function:`, !!onSelect);
      if (onSelect) {
        console.log(`🎭 Calling onSelect for ${favorite.name}`);
        onSelect(favorite);
      } else {
        console.error(`🎭 No onSelect function provided!`);
      }
    }, [onSelect, favorite]);

    const handleDelete = useCallback(
      (e) => {
        e.stopPropagation();
        onDelete(favorite.id);
      },
      [onDelete, favorite.id]
    );

    // Calculate the correct avatar path based on favorite data
    const avatarPath = useMemo(() => {
      console.log(`🎭 Pin mode calculating avatar path for ${favorite.name}:`, {
        isDefault: favorite.isDefault,
        isCustomModel: favorite.isCustomModel,
        previewUrl: favorite.previewUrl,
        fileData: !!favorite.fileData,
      });

      if (favorite.isDefault) {
        // For default avatars, use their specific previewUrl (jupiter.glb)
        return favorite.previewUrl || "/avatar/jupiter.glb";
      } else if (favorite.isCustomModel) {
        return (
          favorite.previewUrl || favorite.fileData || "/avatar/jupiter.glb"
        );
      } else {
        return (
          favorite.previewUrl || favorite.fileData || "/avatar/jupiter.glb"
        );
      }
    }, [
      favorite.isDefault,
      favorite.isCustomModel,
      favorite.previewUrl,
      favorite.fileData,
    ]);

    const handleLoadStart = useCallback(() => {
      console.log(
        `🎭 Pin mode loading started for ${favorite.name} (${favorite.id}) - Path: ${avatarPath}`
      );
      console.log(`🎭 Pin mode AvatarViewer props:`, {
        avatarPath,
        fallbackPath: "/avatar/jupiter.glb",
        position: avatarPosition,
        rotation: avatarRotation,
        scale: avatarScale,
        enableControls: false,
        autoRotate: false,
        showEnvironment: false,
      });
      setIsLoading(true);
      setHasError(false);
    }, [
      favorite.name,
      favorite.id,
      avatarPath,
      avatarPosition,
      avatarRotation,
      avatarScale,
    ]);

    const handleLoadSuccess = useCallback(() => {
      console.log(
        `🎭 Pin mode loading SUCCESS for ${favorite.name} (${favorite.id}) - clearing loading state`
      );
      setIsLoading(false);
      setHasError(false);
    }, [favorite.name, favorite.id]);

    const handleLoadError = useCallback(
      (error) => {
        console.error(
          `🎭 Pin mode avatar load ERROR for ${favorite.name} (${favorite.id}):`,
          error
        );
        console.error(`🎭 Avatar path was:`, avatarPath);
        console.error(`🎭 Error details:`, {
          error: error?.message || error,
          stack: error?.stack,
          name: error?.name,
          type: typeof error,
        });
        setIsLoading(false);
        setHasError(true);
      },
      [favorite.name, favorite.id, avatarPath]
    );

    // Reset loading state when avatar changes
    useEffect(() => {
      console.log(
        `🎭 Pin mode avatar loading: ${favorite.name} (${favorite.id}) - Path: ${avatarPath}`
      );
      console.log(`🎭 Pin mode favorite data:`, favorite);
      console.log(
        `🎭 Pin mode position:`,
        avatarPosition,
        `rotation:`,
        avatarRotation,
        `scale:`,
        avatarScale
      );
      setIsLoading(true);
      setHasError(false);

      // Add a timeout to prevent infinite loading - shorter timeout for better UX
      const loadingTimeout = setTimeout(() => {
        console.warn(
          `🎭 Pin mode loading TIMEOUT for ${favorite.name} - forcing completion`
        );
        setIsLoading(false);
        setHasError(false); // Don't show error, just stop loading
      }, 5000); // 5 second timeout (matching Grid View)

      return () => clearTimeout(loadingTimeout);
    }, [
      favorite.id,
      favorite.previewUrl,
      favorite.name,
      favorite.isDefault,
      favorite.fileData,
      avatarPath,
    ]);

    return (
      <div
        className={`bg-white/5 rounded-lg border overflow-hidden transition-all group cursor-pointer ${
          isSelected
            ? "border-blue-500/50 bg-blue-500/10"
            : shouldShowAsPinPreview
            ? "border-blue-400/30 bg-blue-400/5"
            : "border-white/10"
        }`}
        onClick={handleSelect}
        style={{ height: "320px" }} // Match GridViewComponent height
      >
        {/* Avatar Preview */}
        <div className="h-64 bg-black/20 relative overflow-hidden">
          {/* Loading state */}
          {isLoading && (
            <div className="absolute inset-0 z-20 bg-black/30 backdrop-blur-sm">
              <ModelLoadingSkeleton />
            </div>
          )}

          {/* Error state */}
          {hasError && !isLoading && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center text-white/50 p-2">
              <p className="text-xs text-center mb-1">Failed to load</p>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setIsLoading(true);
                  setHasError(false);
                }}
                className="px-1 py-0.5 bg-red-500/20 text-red-400 text-xs rounded hover:bg-red-500/30 transition-colors"
              >
                Retry
              </button>
            </div>
          )}

          {/* 3D Avatar */}
          {!hasError && (
            <div className="w-full h-full relative">
              {/* Simple Pin Mode 3D Viewer - Built from scratch */}
              <div className="w-full h-full relative">
                <Canvas
                  camera={{ position: [0, 0, 2], fov: 50 }}
                  style={{ background: "transparent" }}
                >
                  <ambientLight intensity={0.4} />
                  <directionalLight position={[5, 10, 5]} intensity={1.2} />
                  <pointLight position={[-5, 5, 5]} intensity={0.5} />

                  <Suspense
                    fallback={
                      <Html center>
                        <div className="flex items-center justify-center">
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-orange-400"></div>
                          <span className="ml-2 text-white text-sm">
                            Loading Guru...
                          </span>
                        </div>
                      </Html>
                    }
                  >
                    <SimplePinModel
                      url={avatarPath}
                      position={avatarPosition}
                      rotation={avatarRotation}
                      scale={avatarScale}
                      enableAnimations={favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')}
                      onLoad={handleLoadSuccess}
                      onError={handleLoadError}
                    />
                  </Suspense>
                </Canvas>
              </div>
            </div>
          )}

          {/* Hover overlay */}
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all flex items-center justify-center opacity-0 group-hover:opacity-100">
            <div className="text-white font-medium text-xs">Select</div>
          </div>

          {/* Pin indicator */}
          {(isSelected || shouldShowAsPinPreview) && (
            <div
              className={`absolute top-1 right-1 w-3 h-3 ${
                isSelected ? "bg-blue-500" : "bg-blue-400/70"
              } rounded-full border border-white/70 ${
                isSelected ? "animate-pulse" : ""
              }`}
              title={isSelected ? "Selected for pinning" : "Pin mode preview"}
            />
          )}

          {/* Delete button for non-default avatars */}
          {!favorite.isDefault && (
            <button
              onClick={handleDelete}
              className="absolute top-1 left-1 p-0.5 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 transition-colors opacity-0 group-hover:opacity-100"
              title="Delete avatar"
            >
              <svg
                className="w-2.5 h-2.5"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9zM4 5a2 2 0 012-2h8a2 2 0 012 2v6a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 102 0v3a1 1 0 11-2 0V9zm4 0a1 1 0 10-2 0v3a1 1 0 102 0V9z"
                  clipRule="evenodd"
                />
              </svg>
            </button>
          )}
        </div>

        {/* Avatar Name */}
        <div className="p-2">
          <h4 className="text-white font-medium text-xs truncate text-center">
            {favorite.name}
          </h4>
          {favorite.isDefault && (
            <p className="text-blue-400 text-xs text-center">Default</p>
          )}
          {favorite.isCustomModel && (
            <p className="text-purple-400 text-xs text-center">Custom</p>
          )}
        </div>
      </div>
    );
  }
);

PinModeComponent.displayName = "PinModeComponent";
PinAvatarCard.displayName = "PinAvatarCard";

export default PinModeComponent;
