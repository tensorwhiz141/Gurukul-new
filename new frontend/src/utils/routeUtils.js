/**
 * Utility functions for route management
 */

/**
 * Saves the current path to localStorage for restoration after refresh or login
 * Excludes auth-related paths
 */
export const saveCurrentPath = () => {
  const currentPath = window.location.pathname;

  // Don't save auth-related paths
  if (
    currentPath !== "/signin" &&
    currentPath !== "/signup" &&
    currentPath !== "/forgotpassword" &&
    currentPath !== "/signout" &&
    currentPath !== "/auth/callback" &&
    currentPath !== "/"
  ) {
    localStorage.setItem("lastVisitedPath", currentPath);
  }
};

/**
 * Gets the last visited path from localStorage or returns the default path
 * @param {string} defaultPath - Default path to return if no saved path exists
 * @returns {string} - Path to navigate to
 */
export const getLastVisitedPath = (defaultPath = "/home") => {
  return localStorage.getItem("lastVisitedPath") || defaultPath;
};

/**
 * Clears the saved path from localStorage
 */
export const clearSavedPath = () => {
  localStorage.removeItem("lastVisitedPath");
};

/**
 * Clears all Supabase-related items from localStorage
 * This can help fix issues with corrupted auth state
 */
export const clearSupabaseCache = () => {
  // Find all Supabase-related items in localStorage
  const keysToRemove = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && (key.includes("supabase") || key.includes("sb-"))) {
      keysToRemove.push(key);
    }
  }

  // Remove all found keys
  keysToRemove.forEach((key) => {
    try {
      localStorage.removeItem(key);
      console.log(`Removed Supabase cache item: ${key}`);
    } catch (error) {
      console.error(`Failed to remove item ${key}:`, error);
    }
  });

  return keysToRemove.length;
};
