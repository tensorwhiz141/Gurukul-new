import React, { memo, useMemo, useCallback, useState, useEffect } from "react";
import { Heart } from "lucide-react";
import OptimizedAvatarCard from "./OptimizedAvatarCard";
import { AvatarGridSkeleton } from "./LoadingSkeletons";

/**
 * Optimized Avatar Grid Component
 * Implements virtualization for large lists and optimized rendering
 */
const OptimizedAvatarGrid = memo(({
  favorites,
  selectedAvatar,
  pinPosition,
  pinRotation,
  pinScale,
  isPinModeEnabled,
  onSelectFavorite,
  onDeleteFavorite,
  isLoading = false,
  enableVirtualization = false,
  maxVisibleItems = 20
}) => {
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: maxVisibleItems });
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  // Handle initial loading state
  useEffect(() => {
    if (favorites.length > 0) {
      const timer = setTimeout(() => setIsInitialLoad(false), 100);
      return () => clearTimeout(timer);
    }
  }, [favorites.length]);

  // Grid configuration for pin mode
  const gridConfig = useMemo(() => {
    return {
      columns: "grid-cols-3 md:grid-cols-4",
      gap: "gap-3",
      size: "small"
    };
  }, []);

  // Memoize visible favorites for virtualization
  const visibleFavorites = useMemo(() => {
    if (!enableVirtualization || favorites.length <= maxVisibleItems) {
      return favorites;
    }
    return favorites.slice(visibleRange.start, visibleRange.end);
  }, [favorites, enableVirtualization, maxVisibleItems, visibleRange]);

  // Optimized select handler
  const handleSelectFavorite = useCallback((favorite) => {
    onSelectFavorite(favorite);
  }, [onSelectFavorite]);

  // Optimized delete handler
  const handleDeleteFavorite = useCallback((favoriteId) => {
    onDeleteFavorite(favoriteId);
  }, [onDeleteFavorite]);

  // Handle scroll for virtualization (if needed)
  const handleScroll = useCallback((e) => {
    if (!enableVirtualization) return;
    
    const container = e.target;
    const scrollTop = container.scrollTop;
    const itemHeight = 200; // Approximate item height
    const containerHeight = container.clientHeight;
    
    const start = Math.floor(scrollTop / itemHeight);
    const end = Math.min(
      start + Math.ceil(containerHeight / itemHeight) + 2,
      favorites.length
    );
    
    setVisibleRange({ start, end });
  }, [enableVirtualization, favorites.length]);

  // Show loading skeleton during initial load
  if (isLoading || isInitialLoad) {
    return (
      <div className="flex-1 overflow-y-auto min-h-0">
        <AvatarGridSkeleton
          count={12}
          columns={4}
        />
      </div>
    );
  }

  // Show empty state
  if (favorites.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-white/50">
        <Heart className="w-16 h-16 mb-3 opacity-50" />
        <p className="text-lg mb-2">No favorites yet</p>
        <p className="text-sm">Generate an avatar and save it as a favorite!</p>
      </div>
    );
  }

  return (
    <div 
      className="flex-1 overflow-y-auto min-h-0"
      onScroll={handleScroll}
    >
      <div className={`${gridConfig.columns} ${gridConfig.gap} grid`}>
        {visibleFavorites.map((favorite) => (
          <OptimizedAvatarCard
            key={`${favorite.id}-pin`}
            favorite={favorite}
            isSelected={selectedAvatar?.id === favorite.id}
            activeSettingsTab="pin"
            pinPosition={pinPosition}
            pinRotation={pinRotation}
            pinScale={pinScale}
            isPinModeEnabled={isPinModeEnabled}
            onSelect={handleSelectFavorite}
            onDelete={handleDeleteFavorite}
            size={gridConfig.size}
            enableControls={true}
          />
        ))}
      </div>
      
      {/* Virtualization spacer (if needed) */}
      {enableVirtualization && favorites.length > maxVisibleItems && (
        <div 
          style={{ 
            height: `${(favorites.length - visibleFavorites.length) * 200}px` 
          }}
          className="w-full"
        />
      )}
    </div>
  );
});

OptimizedAvatarGrid.displayName = 'OptimizedAvatarGrid';

export default OptimizedAvatarGrid;
