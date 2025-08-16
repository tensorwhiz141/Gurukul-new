/**
 * Utility functions for avatar handling
 */

/**
 * Preload an image with better error handling
 * @param {string} url - The URL of the image to preload
 * @returns {Promise<string>} - A promise that resolves with the URL if successful
 */
export const preloadImage = (url) => {
  return new Promise((resolve, reject) => {
    // Check if URL is valid
    if (!url || typeof url !== "string" || !url.trim()) {
      console.warn("Invalid image URL provided to preloadImage:", url);
      resolve(null); // Resolve with null instead of rejecting
      return;
    }

    // Validate URL format
    try {
      new URL(url); // This will throw if URL is invalid
    } catch (e) {
      console.warn("Malformed URL provided to preloadImage:", url);
      resolve(null); // Resolve with null instead of rejecting
      return;
    }

    const img = new Image();

    // Set a timeout to prevent hanging on image load
    const timeoutId = setTimeout(() => {
      console.warn("Image load timeout for URL:", url);
      resolve(null); // Resolve with null instead of rejecting
    }, 5000);

    img.onload = () => {
      clearTimeout(timeoutId);
      resolve(url);
    };

    img.onerror = (e) => {
      clearTimeout(timeoutId);
      console.warn("Failed to load image:", url, e);
      resolve(null); // Resolve with null instead of rejecting
    };

    // Set crossOrigin to anonymous to handle CORS issues
    img.crossOrigin = "anonymous";

    // Set the src after setting up event handlers
    img.src = url;
  });
};

/**
 * Get user initials from user object
 * @param {Object} user - The user object from Supabase
 * @returns {string} - The user's initials
 */
export const getUserInitials = (user) => {
  if (!user) return "U";

  // Try to get initials from full name
  if (user.user_metadata?.full_name) {
    const nameParts = user.user_metadata.full_name.split(" ");
    if (nameParts.length >= 2) {
      return (
        nameParts[0][0] + nameParts[nameParts.length - 1][0]
      ).toUpperCase();
    }
    return nameParts[0][0].toUpperCase();
  }

  // Fallback to email
  if (user.email) {
    return user.email[0].toUpperCase();
  }

  return "U";
};

/**
 * Generate a fallback avatar URL using UI Avatars
 * @param {string} initials - The user's initials
 * @returns {string} - The fallback avatar URL
 */
export const getFallbackAvatarUrl = (initials = "U") => {
  return `https://ui-avatars.com/api/?name=${encodeURIComponent(
    initials
  )}&background=FF9933&color=fff`;
};

/**
 * Handle avatar loading errors by replacing with fallback
 * @param {Event} event - The error event
 * @param {string} initials - The user's initials
 * @returns {boolean} - Whether the fallback was applied successfully
 */
export const handleAvatarError = (event, initials = "U") => {
  try {
    console.log("Avatar image error, using fallback");

    // Safety check for event and target
    if (!event || !event.target) {
      console.warn("Invalid event passed to handleAvatarError");
      return false;
    }

    // Prevent infinite error loop
    event.target.onerror = null;

    // Generate fallback URL
    const fallbackUrl = getFallbackAvatarUrl(initials);

    // Set the fallback image
    event.target.src = fallbackUrl;

    return true;
  } catch (error) {
    console.error("Error in handleAvatarError:", error);
    return false;
  }
};
