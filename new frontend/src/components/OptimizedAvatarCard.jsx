import React, { memo, useState, useCallback, useMemo } from "react";
import { useSelector } from "react-redux";
import { User, Trash2 } from "lucide-react";
import AvatarViewer from "./AvatarViewer";
import { ModelLoadingSkeleton } from "./LoadingSkeletons";
import { selectIsSpeaking } from "../store/avatarSlice";

/**
 * Optimized Avatar Card Component
 * Uses React.memo and optimized rendering patterns for better performance
 */
const OptimizedAvatarCard = memo(({
  favorite,
  isSelected,
  activeSettingsTab = "pin", // Always pin mode now
  pinPosition,
  pinRotation,
  pinScale,
  isPinModeEnabled,
  onSelect,
  onDelete,
  enableControls = true,
  size = "normal" // "normal" | "small"
}) => {
  const [isLoading, setIsLoading] = useState(true); // Start with true to show loading state
  const [hasError, setHasError] = useState(false);

  // Get speaking state for avatar animation
  const isSpeaking = useSelector(selectIsSpeaking);

  // For Pin Mode: if no avatar is selected, show the first favorite (usually default) as a preview
  // This helps during loading/restoration when selectedAvatar might be temporarily null
  const shouldShowAsPinPreview = activeSettingsTab === 'pin' && !isSelected && favorite.id === 'default';

  // Calculate avatar path based on favorite data
  const avatarPath = useMemo(() => {
    if (favorite.isDefault) {
      // For default avatars, use their specific previewUrl (jupiter.glb)
      return favorite.previewUrl || '/avatar/jupiter.glb';
    } else if (favorite.isCustomModel) {
      return favorite.previewUrl || favorite.fileData || '/avatar/jupiter.glb';
    } else {
      return favorite.previewUrl || favorite.fileData || '/avatar/jupiter.glb';
    }
  }, [favorite.isDefault, favorite.isCustomModel, favorite.previewUrl, favorite.fileData]);

  // Memoize position calculations to prevent unnecessary recalculations
  const avatarPosition = useMemo(() => {
    if (activeSettingsTab === 'grid') {
      return isSelected ? [
        Math.max(-3, Math.min(3, gridPosition?.x || 0)),
        Math.max(-3, Math.min(2, (gridPosition?.y || 0) - 1)),
        Math.max(-3, Math.min(3, gridPosition?.z || 0))
      ] : [
        Math.max(-3, Math.min(3, (favorite.gridPosition?.x || favorite.avatarPosition?.x || 0))),
        Math.max(-3, Math.min(2, ((favorite.gridPosition?.y || favorite.avatarPosition?.y || 0) - 1))),
        Math.max(-3, Math.min(3, (favorite.gridPosition?.z || favorite.avatarPosition?.z || 0)))
      ];
    } else {
      // Pin mode - use a smaller offset for better visibility in cards
      let baseOffset = size === "small" ? -0.3 : -0.5; // Reduced offset for better visibility

      // Avatar-specific position adjustments
      if (favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')) {
        baseOffset = size === "small" ? 0.6 : 0.8; // Much higher position for jupiter.glb
      }

      // Use current pin settings if selected, or show as preview if this is the default avatar and no avatar is selected
      if (isSelected || shouldShowAsPinPreview) {
        // Hardcoded perfect position for jupiter.glb
        if (favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')) {
          return [0, 0.5, -4.0];
        }
        return [
          Math.max(-2.5, Math.min(2.5, pinPosition?.x || 0)),
          Math.max(-2.5, Math.min(1.5, (pinPosition?.y || 0) + baseOffset)),
          Math.max(-2.5, Math.min(2.5, pinPosition?.z || 0))
        ];
      } else {
        // Hardcoded perfect position for jupiter.glb
        if (favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')) {
          return [0, 0.5, -4.0];
        }
        return [
          Math.max(-2.5, Math.min(2.5, (favorite.pinPosition?.x || favorite.avatarPosition?.x || 0))),
          Math.max(-2.5, Math.min(1.5, ((favorite.pinPosition?.y || favorite.avatarPosition?.y || 0) + baseOffset))),
          Math.max(-2.5, Math.min(2.5, (favorite.pinPosition?.z || favorite.avatarPosition?.z || 0)))
        ];
      }
    }
  }, [
    activeSettingsTab, isSelected, gridPosition, pinPosition, favorite, size, shouldShowAsPinPreview
  ]);

  const avatarRotation = useMemo(() => {
    if (activeSettingsTab === 'grid') {
      return isSelected ? [
        ((gridRotation?.x || 0) * Math.PI) / 180,
        ((gridRotation?.y || 0) * Math.PI) / 180,
        ((gridRotation?.z || 0) * Math.PI) / 180
      ] : [
        ((favorite.gridRotation?.x || favorite.avatarRotation?.x || 0) * Math.PI) / 180,
        ((favorite.gridRotation?.y || favorite.avatarRotation?.y || 0) * Math.PI) / 180,
        ((favorite.gridRotation?.z || favorite.avatarRotation?.z || 0) * Math.PI) / 180
      ];
    } else {
      // Pin mode - use current pin settings if selected or showing as preview
      if (isSelected || shouldShowAsPinPreview) {
        return [
          ((pinRotation?.x || 0) * Math.PI) / 180,
          ((pinRotation?.y || 0) * Math.PI) / 180,
          ((pinRotation?.z || 0) * Math.PI) / 180
        ];
      } else {
        return [
          ((favorite.pinRotation?.x || favorite.avatarRotation?.x || 0) * Math.PI) / 180,
          ((favorite.pinRotation?.y || favorite.avatarRotation?.y || 0) * Math.PI) / 180,
          ((favorite.pinRotation?.z || favorite.avatarRotation?.z || 0) * Math.PI) / 180
        ];
      }
    }
  }, [
    activeSettingsTab, isSelected, gridRotation, pinRotation, favorite, shouldShowAsPinPreview
  ]);

  const avatarScale = useMemo(() => {
    // Avatar-specific default scales
    let defaultScale = 0.6; // Default for jupiter.glb
    if (favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')) {
      defaultScale = 0.6; // Decreased by 0.3 (was 0.9, now 0.6)
    }

    // Pin mode - use current pin settings if selected or showing as preview
    // Use higher minimum scale for better visibility in pin mode
    if (isSelected || shouldShowAsPinPreview) {
      return Math.max(1.5, Math.min(4.0, pinScale || defaultScale));
    } else {
      return Math.max(1.5, Math.min(4.0, favorite.pinScale || favorite.avatarScale || defaultScale));
    }
  }, [
    isSelected, pinScale, favorite, shouldShowAsPinPreview
  ]);

  // Memoize camera position based on size
  const cameraPosition = useMemo(() => {
    return size === "small" ? [0, 0, 2.5] : [0, 0, 3];
  }, [size]);

  // Optimized event handlers
  const handleSelect = useCallback(() => {
    onSelect(favorite);
  }, [onSelect, favorite]);

  // Reduced debug logging to prevent performance issues
  React.useEffect(() => {
    // Only log errors or initial load
    if (hasError) {
      console.log(`Avatar Card Error - ${favorite.name}:`, {
        id: favorite.id,
        isDefault: favorite.isDefault,
        previewUrl: favorite.previewUrl,
        hasError
      });
    }
  }, [favorite.name, favorite.id, favorite.isDefault, favorite.previewUrl, hasError]);

  // Reset loading state when avatar changes
  React.useEffect(() => {
    setIsLoading(true);
    setHasError(false);
  }, [favorite.id, favorite.previewUrl]);

  // Force loading state to clear after timeout (fallback)
  React.useEffect(() => {
    if (isLoading) {
      const timeout = setTimeout(() => {
        console.warn(`⚠️ Loading timeout for ${favorite.name}, forcing load success`);
        setIsLoading(false);
      }, 10000); // 10 second timeout for better loading experience
      return () => clearTimeout(timeout);
    }
  }, [isLoading, favorite.name]);

  const handleDelete = useCallback((e) => {
    e.stopPropagation();
    onDelete(favorite.id);
  }, [onDelete, favorite.id]);

  const handleLoadStart = useCallback(() => {
    console.log(`🔄 Starting to load avatar: ${favorite.name}`);
    setIsLoading(true);
    setHasError(false);
  }, [favorite.name]);

  const handleLoadSuccess = useCallback(() => {
    console.log(`✅ Avatar loaded successfully: ${favorite.name} (${favorite.isDefault ? 'Default' : 'Custom'})`);
    // Small delay to ensure smooth transition from loading state
    setTimeout(() => {
      setIsLoading(false);
      setHasError(false);
    }, 300);
  }, [favorite.name, favorite.isDefault]);

  const handleLoadError = useCallback((error) => {
    console.error(`Avatar card error for ${favorite.name}:`, error);
    console.error('Avatar path:', favorite.previewUrl);
    console.error('Error details:', error);
    setIsLoading(false);
    setHasError(true);
  }, [favorite.name, favorite.previewUrl]);

  // Determine if this card should show controls
  // In Pin Mode, we want to show the avatar but with limited controls (no orbit controls)
  // In Grid View, we want full controls when selected
  const shouldShowControls = enableControls && isSelected && activeSettingsTab === 'grid';

  // Auto-rotate logic:
  // - Always auto-rotate when not selected
  // - In Pin Mode, auto-rotate even when selected (for preview effect)
  // - In Grid View, don't auto-rotate when selected (user can manually control)
  // - Always auto-rotate jupiter.glb regardless of selection state
  const shouldAutoRotate = (favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')) ||
                           !isSelected ||
                           activeSettingsTab === 'pin';

  // Enable interaction only in Grid View when selected
  const shouldEnableInteraction = enableControls && isSelected && activeSettingsTab === 'grid';

  // Size-specific classes
  const sizeClasses = size === "small" ? {
    card: "hover:border-orange-500/30",
    icon: "w-8 h-8",
    deleteButton: "top-1 right-1 p-0.5",
    deleteIcon: "w-2.5 h-2.5",
    indicator: "top-1 left-1 w-3 h-3",
    name: "p-2",
    nameText: "text-xs",
    nameSubtext: "text-[10px] mt-0.5"
  } : {
    card: "hover:border-orange-500/30",
    icon: "w-12 h-12",
    deleteButton: "top-2 right-2 p-1",
    deleteIcon: "w-3 h-3",
    indicator: "top-2 left-2 w-3 h-3",
    name: "p-3",
    nameText: "text-sm",
    nameSubtext: "text-xs mt-1"
  };

  return (
    <div
      className={`bg-white/5 rounded-lg border overflow-hidden transition-all group cursor-pointer ${
        isSelected
          ? 'border-orange-500/50 bg-orange-500/10'
          : shouldShowAsPinPreview
          ? 'border-orange-400/30 bg-orange-400/5'
          : 'border-white/10'
      } ${sizeClasses.card}`}
      onClick={handleSelect}
    >
      {/* Avatar Preview */}
      <div className="aspect-square bg-black/20 relative overflow-hidden">
        {/* Loading state */}
        {isLoading && (
          <div className="absolute inset-0 z-10">
            <ModelLoadingSkeleton />
          </div>
        )}

        {/* Avatar content */}
        {hasError ? (
          /* Error fallback */
          <div className="w-full h-full flex flex-col items-center justify-center bg-red-500/10 border border-red-500/20">
            <User className={`${sizeClasses.icon} text-red-400/60 mb-2`} />
            <div className="text-red-400 text-xs text-center px-2">
              Failed to load
            </div>
            <button
              onClick={() => {
                setHasError(false);
                setIsLoading(true);
              }}
              className="mt-1 px-2 py-1 bg-red-500/20 text-red-400 text-xs rounded hover:bg-red-500/30 transition-colors"
            >
              Retry
            </button>
          </div>
        ) : (
          // Always show 3D avatar for any favorite
          <>
            {/* Debug indicator for Pin Mode */}
            {activeSettingsTab === 'pin' && (
              <div className="absolute top-0 left-0 bg-yellow-500/20 text-yellow-300 text-xs p-1 z-20 pointer-events-none">
                Pin Mode: {isSelected ? 'Selected' : 'Not Selected'} | Scale: {avatarScale.toFixed(1)}
              </div>
            )}
            <AvatarViewer
            key={`avatar-${favorite.id}`}
            avatarPath={avatarPath}
            fallbackPath="/avatar/jupiter.glb"
            enableControls={shouldShowControls}
            autoRotate={shouldAutoRotate}
            autoRotateSpeed={favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb') ? 2.0 : 0.3} // 4x faster for jupiter
            showEnvironment={false}
            fallbackMessage={favorite.isDefault ? "Default Avatar" : "3D Avatar"}
            className="w-full h-full"
            position={avatarPosition}
            rotation={avatarRotation}
            scale={avatarScale}
            style="realistic"
            enableInteraction={shouldEnableInteraction}
            enableAnimations={favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb')} // Enable animations for jupiter.glb
            isSpeaking={isSpeaking && isSelected} // Only show speaking animation if this avatar is selected
            cameraPosition={cameraPosition}
            onLoadStart={handleLoadStart}
            onLoad={handleLoadSuccess}
            onError={(error) => {
              console.error(`Avatar load error for ${favorite.name}:`, error);
              console.log('Avatar data:', {
                previewUrl: favorite.previewUrl,
                fileData: favorite.fileData,
                fallbackUrl: favorite.fallbackUrl,
                isDefault: favorite.isDefault,
                favorite: favorite
              });
              console.log('Avatar render props:', {
                activeSettingsTab,
                isSelected,
                shouldShowControls,
                shouldAutoRotate,
                shouldEnableInteraction,
                avatarPosition,
                avatarRotation,
                avatarScale,
                shouldShowAsPinPreview
              });
              handleLoadError(error);
            }}
          />
          </>
        )}

        {/* Hover overlay */}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all flex items-center justify-center opacity-0 group-hover:opacity-100">
          <div className={`text-white font-medium ${size === "small" ? "text-xs" : "text-sm"}`}>
            Select
          </div>
        </div>

        {/* Custom Settings Indicator */}
        {(favorite.avatarPosition || favorite.avatarRotation || favorite.avatarScale !== 1 || favorite.isPinModeEnabled) && (
          <div 
            className={`absolute ${sizeClasses.indicator} bg-green-500 rounded-full border border-white/50 opacity-80`}
            title="Custom settings saved"
          />
        )}

        {/* Pin indicator for selected avatar in pin mode */}
        {(isSelected || shouldShowAsPinPreview) && activeSettingsTab === 'pin' && (
          <div
            className={`absolute top-1 right-1 w-3 h-3 ${isSelected ? 'bg-orange-500' : 'bg-orange-400/70'} rounded-full border border-white/70 ${isSelected ? 'animate-pulse' : ''}`}
            title={isSelected ? "Selected for pinning" : "Pin mode preview"}
          />
        )}

        {/* Delete button */}
        {!favorite.isDefault && (
          <button
            onClick={handleDelete}
            className={`absolute ${sizeClasses.deleteButton} bg-black/50 backdrop-blur-sm rounded-full border border-white/20 hover:bg-red-500/50 transition-all opacity-0 group-hover:opacity-100`}
          >
            <Trash2 className={`${sizeClasses.deleteIcon} text-white`} />
          </button>
        )}
      </div>

      {/* Avatar Name */}
      <div className={sizeClasses.name}>
        <h4 className={`text-white font-medium ${sizeClasses.nameText} truncate text-center`}>
          {favorite.name}
        </h4>
        {favorite.isDefault && (
          <p className={`text-orange-400 ${sizeClasses.nameSubtext} text-center`}>Default</p>
        )}
        {favorite.isCustomModel && (
          <p className={`text-purple-400 ${sizeClasses.nameSubtext} text-center`}>
            {size === "small" ? "Custom" : "Custom Model"}
          </p>
        )}
      </div>
    </div>
  );
});

OptimizedAvatarCard.displayName = 'OptimizedAvatarCard';

export default OptimizedAvatarCard;
