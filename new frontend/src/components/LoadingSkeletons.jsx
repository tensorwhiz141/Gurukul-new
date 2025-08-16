import React from "react";

/**
 * Skeleton loader for avatar grid cards
 */
export const AvatarCardSkeleton = ({ className = "" }) => {
  return (
    <div className={`bg-white/5 rounded-lg border border-white/10 overflow-hidden animate-pulse ${className}`}>
      {/* Avatar Preview Skeleton */}
      <div className="aspect-square bg-white/10 relative">
        <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-12 h-12 bg-white/20 rounded-full"></div>
        </div>
      </div>
      
      {/* Avatar Name Skeleton */}
      <div className="p-3">
        <div className="h-4 bg-white/20 rounded w-3/4 mx-auto mb-1"></div>
        <div className="h-3 bg-white/10 rounded w-1/2 mx-auto"></div>
      </div>
    </div>
  );
};

/**
 * Skeleton loader for avatar grid
 */
export const AvatarGridSkeleton = ({ count = 8, columns = 4 }) => {
  return (
    <div className={`grid grid-cols-2 md:grid-cols-3 lg:grid-cols-${columns} gap-4`}>
      {Array.from({ length: count }, (_, index) => (
        <AvatarCardSkeleton key={index} />
      ))}
    </div>
  );
};

/**
 * Skeleton loader for 3D model loading
 */
export const ModelLoadingSkeleton = ({ className = "" }) => {
  return (
    <div className={`w-full h-full bg-black/20 rounded-lg border border-white/10 relative overflow-hidden ${className}`}>
      {/* Animated gradient background with orange theme */}
      <div className="absolute inset-0 bg-gradient-to-br from-orange-500/10 via-amber-500/10 to-orange-500/10 animate-pulse"></div>

      {/* Loading spinner with orange theme */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-white/20 border-t-orange-400 rounded-full animate-spin"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-8 h-8 bg-orange-400/20 rounded-full animate-pulse"></div>
          </div>
        </div>
      </div>

      {/* Loading text */}
      <div className="absolute bottom-4 left-0 right-0 text-center">
        <div className="text-white/60 text-sm animate-pulse">Loading Guru...</div>
      </div>
    </div>
  );
};

/**
 * Skeleton loader for settings panel
 */
export const SettingsPanelSkeleton = ({ className = "" }) => {
  return (
    <div className={`bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6 ${className}`}>
      {/* Header skeleton */}
      <div className="mb-6">
        <div className="h-6 bg-white/20 rounded w-1/2 mb-2 animate-pulse"></div>
        <div className="h-4 bg-white/10 rounded w-3/4 animate-pulse"></div>
      </div>
      
      {/* Tab skeleton */}
      <div className="flex space-x-2 mb-6">
        {[1, 2].map((_, index) => (
          <div key={index} className="h-10 bg-white/10 rounded w-20 animate-pulse"></div>
        ))}
      </div>
      
      {/* Controls skeleton */}
      <div className="space-y-4">
        {[1, 2, 3, 4].map((_, index) => (
          <div key={index} className="space-y-2">
            <div className="h-4 bg-white/15 rounded w-1/3 animate-pulse"></div>
            <div className="h-10 bg-white/10 rounded animate-pulse"></div>
          </div>
        ))}
      </div>
      
      {/* Button skeleton */}
      <div className="mt-6 space-y-2">
        <div className="h-10 bg-blue-500/20 rounded animate-pulse"></div>
        <div className="h-10 bg-white/10 rounded animate-pulse"></div>
      </div>
    </div>
  );
};

/**
 * Skeleton loader for generation form
 */
export const GenerationFormSkeleton = ({ className = "" }) => {
  return (
    <div className={`bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6 ${className}`}>
      {/* Header skeleton */}
      <div className="h-6 bg-white/20 rounded w-1/3 mb-6 animate-pulse"></div>
      
      {/* Method tabs skeleton */}
      <div className="flex space-x-1 mb-6 bg-white/5 rounded-lg p-1">
        {[1, 2].map((_, index) => (
          <div key={index} className="flex-1 h-10 bg-white/10 rounded animate-pulse"></div>
        ))}
      </div>
      
      {/* Input area skeleton */}
      <div className="h-48 bg-white/10 rounded-lg mb-6 animate-pulse"></div>
      
      {/* Settings row skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {[1, 2].map((_, index) => (
          <div key={index} className="space-y-2">
            <div className="h-4 bg-white/15 rounded w-1/3 animate-pulse"></div>
            <div className="h-11 bg-white/10 rounded animate-pulse"></div>
          </div>
        ))}
      </div>
      
      {/* Generate button skeleton */}
      <div className="flex justify-center">
        <div className="h-12 bg-gradient-to-r from-blue-500/20 to-purple-600/20 rounded-lg w-48 animate-pulse"></div>
      </div>
    </div>
  );
};

/**
 * Skeleton loader for preview panel
 */
export const PreviewPanelSkeleton = ({ className = "" }) => {
  return (
    <div className={`bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6 ${className}`}>
      {/* Header skeleton */}
      <div className="h-5 bg-white/20 rounded w-1/4 mb-4 animate-pulse"></div>
      
      {/* Preview area skeleton */}
      <div className="flex-1 bg-black/20 rounded-lg border border-white/10 mb-4 relative overflow-hidden">
        <ModelLoadingSkeleton />
      </div>
      
      {/* Info text skeleton */}
      <div className="text-center">
        <div className="h-4 bg-white/10 rounded w-3/4 mx-auto animate-pulse"></div>
      </div>
    </div>
  );
};

/**
 * Skeleton loader for custom models tab
 */
export const CustomModelsTabSkeleton = ({ className = "" }) => {
  return (
    <div className={`${className}`}>
      {/* Header skeleton */}
      <div className="h-6 bg-white/20 rounded w-1/3 mb-6 animate-pulse"></div>
      
      {/* Upload area skeleton */}
      <div className="border-2 border-dashed border-white/20 rounded-lg p-8 mb-6">
        <div className="text-center space-y-3">
          <div className="w-12 h-12 bg-white/20 rounded mx-auto animate-pulse"></div>
          <div className="h-4 bg-white/15 rounded w-1/2 mx-auto animate-pulse"></div>
          <div className="h-3 bg-white/10 rounded w-1/3 mx-auto animate-pulse"></div>
        </div>
      </div>
      
      {/* Models grid skeleton */}
      <AvatarGridSkeleton count={6} columns={3} />
    </div>
  );
};

/**
 * Inline loading spinner for small areas
 */
export const InlineSpinner = ({ size = "sm", className = "" }) => {
  const sizeClasses = {
    xs: "w-3 h-3",
    sm: "w-4 h-4", 
    md: "w-6 h-6",
    lg: "w-8 h-8"
  };
  
  return (
    <div className={`${sizeClasses[size]} border-2 border-white/20 border-t-blue-400 rounded-full animate-spin ${className}`}></div>
  );
};

/**
 * Loading overlay for full-screen loading states
 */
export const LoadingOverlay = ({ message = "Loading...", className = "" }) => {
  return (
    <div className={`absolute inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 ${className}`}>
      <div className="text-center space-y-4">
        <div className="w-12 h-12 border-4 border-white/20 border-t-orange-400 rounded-full animate-spin mx-auto"></div>
        <div className="text-white/80 text-sm">{message}</div>
      </div>
    </div>
  );
};
