import React, { useState, useCallback, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { toast } from 'react-hot-toast';
import {
  Heart,
  Trash2,
  AlertCircle,
  Edit2
} from 'lucide-react';
import AvatarViewer from './AvatarViewer';
import {
  selectFavorites,
  selectSelectedAvatar,
  setSelectedAvatar,
  loadAvatarSettings,
  removeFavorite,
} from '../store/avatarSlice';

const FavoritesTab = () => {
  const dispatch = useDispatch();
  const favorites = useSelector(selectFavorites);
  const selectedAvatar = useSelector(selectSelectedAvatar);
  




  const [error, setError] = useState(null);

  // Custom names storage
  const [customNames, setCustomNames] = useState(() => {
    try {
      const stored = localStorage.getItem('avatar-custom-names');
      return stored ? JSON.parse(stored) : {};
    } catch (error) {
      console.error('Error loading custom names:', error);
      return {};
    }
  });

  // Save custom names to localStorage
  const saveCustomNames = useCallback((names) => {
    try {
      localStorage.setItem('avatar-custom-names', JSON.stringify(names));
      setCustomNames(names);
    } catch (error) {
      console.error('Error saving custom names:', error);
    }
  }, []);

  // Update custom name for a favorite
  const updateCustomName = useCallback((favoriteId, newName) => {
    const updatedNames = { ...customNames, [favoriteId]: newName };
    saveCustomNames(updatedNames);
  }, [customNames, saveCustomNames]);

  // Get display name for a favorite
  const getDisplayName = useCallback((favorite, index) => {
    if (customNames[favorite.id]) {
      return customNames[favorite.id];
    }

    // Use the avatar's name property if it exists
    if (favorite.name) {
      return favorite.name;
    }

    // For Jupiter avatar, use "Brihaspati"
    if (favorite.id === 'jupiter-default' ||
        (favorite.previewUrl && favorite.previewUrl.includes('jupiter.glb'))) {
      return 'Brihaspati';
    }

    // Fallback to generic naming
    return `Guru${index + 1}`;
  }, [customNames]);

  // Handle favorite selection
  const handleSelectFavorite = async (favorite) => {
    try {
      setError(null);


      // Create avatar object with proper data structure
      let avatarWithData = { ...favorite };

      // Handle different favorite types
      if (favorite.isCustomModel && favorite.fileData) {
        // For custom models, convert base64 to blob URL
        const bytes = Uint8Array.from(atob(favorite.fileData), c => c.charCodeAt(0));
        const blob = new Blob([bytes], { type: 'application/octet-stream' });
        const blobUrl = URL.createObjectURL(blob);

        avatarWithData = {
          ...favorite,
          previewUrl: blobUrl,
          isBlobUrl: true,
        };
      }

      dispatch(setSelectedAvatar(avatarWithData));
      dispatch(loadAvatarSettings(avatarWithData));
      toast.success(`Selected ${favorite.name}`);
    } catch (error) {
      console.error('Error selecting favorite:', error);
      toast.error('Failed to load favorite');
      setError('Failed to load favorite');
    }
  };

  // Handle favorite deletion
  const handleDeleteFavorite = async (favoriteId) => {
    const favorite = favorites.find(f => f.id === favoriteId);
    if (!favorite) return;

    if (window.confirm(`Are you sure you want to remove "${favorite.name}" from favorites?`)) {
      try {
        dispatch(removeFavorite(favoriteId));

        if (selectedAvatar?.id === favoriteId) {
          dispatch(setSelectedAvatar(null));
        }

        toast.success(`${favorite.name} removed from favorites`);
      } catch (error) {
        console.error('Error removing favorite:', error);
        toast.error('Failed to remove favorite');
      }
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-white">Favorite Avatars</h2>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="text-sm text-white/60">
              {favorites.length} favorite{favorites.length !== 1 ? 's' : ''}
            </div>
          </div>
          {favorites.length > 0 && (
            <button
              onClick={() => {
                if (window.confirm('Are you sure you want to clear all favorites?')) {
                  favorites.forEach(favorite => {
                    dispatch(removeFavorite(favorite.id));
                  });
                  if (selectedAvatar && favorites.some(f => f.id === selectedAvatar.id)) {
                    dispatch(setSelectedAvatar(null));
                  }
                  toast.success('All favorites cleared');
                }
              }}
              className="px-3 py-1 bg-red-500/20 border border-red-500/30 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors text-sm"
              title="Clear all favorites"
            >
              Clear All
            </button>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-3">
          <AlertCircle className="w-4 h-4 text-red-400" />
          <span className="text-red-400 text-sm">{error}</span>
        </div>
      )}

      {/* Favorites Grid */}
      <div className="flex-1 overflow-y-auto">
        {favorites.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-white/40">
            <Heart className="w-16 h-16 mb-4" />
            <p className="text-lg">No favorite avatars yet</p>
            <p className="text-sm">Save avatars to your favorites to see them here</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {favorites.map((favorite, index) => (
              <FavoriteCard
                key={favorite.id}
                favorite={favorite}
                displayName={getDisplayName(favorite, index)}
                isSelected={selectedAvatar?.id === favorite.id}
                onSelect={handleSelectFavorite}
                onDelete={handleDeleteFavorite}
                onUpdateName={updateCustomName}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Individual Favorite Card Component
const FavoriteCard = ({
  favorite,
  displayName,
  isSelected,
  onSelect,
  onDelete,
  onUpdateName
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(displayName);

  // Calculate avatar path based on favorite data
  const avatarPath = React.useMemo(() => {
    let path;
    if (favorite.isDefault) {
      // For default avatars, use their specific previewUrl (jupiter.glb)
      path = favorite.previewUrl || '/avatar/jupiter.glb';
    } else if (favorite.isCustomModel) {
      path = favorite.previewUrl || favorite.fileData || '/avatar/jupiter.glb';
    } else {
      path = favorite.previewUrl || favorite.fileData || '/avatar/jupiter.glb';
    }

    return path;
  }, [favorite.isDefault, favorite.isCustomModel, favorite.previewUrl, favorite.fileData]);

  // Handle name editing
  const handleStartEdit = useCallback((e) => {
    e.stopPropagation();
    setIsEditing(true);
    setEditName(displayName);
  }, [displayName]);

  const handleSaveName = useCallback((e) => {
    e.stopPropagation();
    const trimmedName = editName.trim();
    if (trimmedName && trimmedName !== displayName) {
      onUpdateName(favorite.id, trimmedName);
    }
    setIsEditing(false);
  }, [editName, displayName, favorite.id, onUpdateName]);

  const handleCancelEdit = useCallback((e) => {
    e.stopPropagation();
    setIsEditing(false);
    setEditName(displayName);
  }, [displayName]);

  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter') {
      handleSaveName(e);
    } else if (e.key === 'Escape') {
      handleCancelEdit(e);
    }
  }, [handleSaveName, handleCancelEdit]);

  // Block body interactions when modal is open
  React.useEffect(() => {
    if (isEditing) {
      // Prevent scrolling and interactions on body
      document.body.style.overflow = 'hidden';
      document.body.style.pointerEvents = 'none';

      // Allow interactions only on the modal
      const modalElement = document.querySelector('[data-modal="edit-name"]');
      if (modalElement) {
        modalElement.style.pointerEvents = 'auto';
      }

      return () => {
        document.body.style.overflow = '';
        document.body.style.pointerEvents = '';
      };
    }
  }, [isEditing]);

  // Update edit name when display name changes
  useEffect(() => {
    if (!isEditing) {
      setEditName(displayName);
    }
  }, [displayName, isEditing]);

  return (
    <div
      className={`bg-white/5 rounded-lg border overflow-hidden hover:border-orange-500/30 transition-all group cursor-pointer relative ${
        isSelected
          ? 'border-orange-500/50 bg-orange-500/10'
          : 'border-white/10'
      }`}
      onClick={() => onSelect(favorite)}
      style={{ height: '320px', zIndex: 1 }}
    >
      {/* Avatar Preview */}
      <div className={`h-64 relative overflow-hidden ${
        isSelected
          ? 'bg-orange-500/20 border-2 border-orange-400/50'
          : 'bg-black/20'
      }`}>
        {/* Show pin mode preview for selected avatar (always pin mode now) */}
        {isSelected ? (
          <div
            className="w-full h-full relative bg-orange-500/10"
            data-avatar-selection-pin-preview
          >
            {/* This container will be used by GlobalPinnedAvatar for the contained avatar */}
          </div>
        ) : (
          <div className="w-full h-full">
            <AvatarViewer
              avatarPath={avatarPath}
              position={[0, -0.8, 0]}
              rotation={[0, Math.PI, 0]}
              scale={1}
              cameraPosition={[0, 0, 3]}
              enableControls={false}
            />
          </div>
        )}

        {/* Hover overlay */}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all flex items-center justify-center opacity-0 group-hover:opacity-100 z-10">
          <div className="text-white text-sm font-medium">Select</div>
        </div>

        {/* Delete button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete(favorite.id);
          }}
          className="absolute top-2 right-2 p-1 bg-black/50 backdrop-blur-sm rounded-full border border-white/20 hover:bg-red-500/50 transition-all opacity-0 group-hover:opacity-100 z-20"
        >
          <Trash2 className="w-3 h-3 text-white" />
        </button>
      </div>

      {/* Avatar Info with Editable Name */}
      <div className="p-3 relative group/name">
        <div className="text-center">
          <h4 className="font-bold text-base bg-gradient-to-r from-orange-400 to-amber-400 bg-clip-text text-transparent drop-shadow-lg">
            {displayName}
          </h4>
        </div>

        <button
          onClick={handleStartEdit}
          className="absolute top-1 right-1 p-1.5 text-white/30 hover:text-white/80 transition-all opacity-0 group-hover/name:opacity-100 hover:bg-white/10 rounded-full"
          title="Edit name"
        >
          <Edit2 className="w-3 h-3" />
        </button>
      </div>

      {/* Edit Name Modal */}
      {isEditing && (
        <div
          className="absolute inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-[9999] rounded-xl overflow-hidden"
          onClick={handleCancelEdit}
          onMouseDown={(e) => e.stopPropagation()}
          onTouchStart={(e) => e.stopPropagation()}
          style={{ pointerEvents: 'auto' }}
          data-modal="edit-name"
        >
          <div
            className="bg-white/10 backdrop-blur-md border border-orange-500/30 rounded-xl p-6 max-w-sm w-full mx-4 pointer-events-auto"
            onClick={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
            onTouchStart={(e) => e.stopPropagation()}
          >
            <h3 className="text-white text-lg font-semibold mb-4 text-center">Edit Avatar Name</h3>

            <div className="mb-6">
              <input
                type="text"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                onKeyDown={handleKeyPress}
                className="w-full bg-white/10 border border-orange-400/40 rounded-lg px-4 py-3 text-white text-center focus:outline-none focus:border-orange-500/70 focus:bg-white/15 transition-all"
                placeholder="Enter avatar name..."
                autoFocus
              />
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleCancelEdit}
                className="flex-1 px-4 py-2 bg-white/10 border border-white/20 text-white rounded-lg hover:bg-white/20 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveName}
                className="flex-1 px-4 py-2 bg-orange-500/80 border border-orange-500/50 text-white rounded-lg hover:bg-orange-500 transition-colors"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FavoritesTab;
