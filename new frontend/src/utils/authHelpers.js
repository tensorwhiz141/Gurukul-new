import { supabase } from "../supabaseClient";
import { storage } from "./storageUtils";
import { toast } from "react-hot-toast";

/**
 * Thoroughly clears authentication data from storage while preserving avatar data
 * This helps ensure a complete sign-out without losing user's avatar settings
 */
export const clearAuthData = () => {
  try {
    // Clear localStorage except persistent keys (avatar data, settings)
    storage.clearExceptPersistent();

    // Specifically target Supabase auth tokens in sessionStorage
    window.sessionStorage.removeItem("supabase.auth.token");

    // Clear any auth-related cookies
    document.cookie.split(";").forEach(function (c) {
      document.cookie = c
        .replace(/^ +/, "")
        .replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/");
    });

    console.log("Auth data cleared successfully (avatar data preserved)");
    return true;
  } catch (error) {
    console.error("Error clearing auth data:", error);
    return false;
  }
};

/**
 * Checks if the current user's account has been marked as deleted
 * and signs them out if it has.
 *
 * @returns {Promise<boolean>} True if the user is deleted and was signed out, false otherwise
 */
export const checkAndHandleDeletedAccount = async () => {
  try {
    // Get the current user
    const {
      data: { user },
    } = await supabase.auth.getUser();

    // If no user is logged in, nothing to do
    if (!user) {
      return false;
    }

    // Check if the user has been marked as deleted
    if (user.user_metadata?.deleted === true) {
      console.log("Detected deleted account, signing out user:", user.id);

      // Sign out the user from all devices
      await supabase.auth.signOut({ scope: "global" });

      // Clear all auth data
      clearAuthData();

      // Notify the user
      toast.error("This account has been deleted and cannot be used.", {
        position: "top-right",
        duration: 5000,
      });

      return true;
    }

    return false;
  } catch (error) {
    console.error("Error checking for deleted account:", error);
    return false;
  }
};

/**
 * Checks if a user's email matches a deleted account pattern
 *
 * @param {string} email The email to check
 * @returns {boolean} True if the email matches a deleted account pattern
 */
export const isDeletedAccountEmail = (email) => {
  if (!email) return false;

  // Check if the email matches our deleted account pattern
  return email.startsWith("deleted-") && email.includes("@deleted-account.com");
};
