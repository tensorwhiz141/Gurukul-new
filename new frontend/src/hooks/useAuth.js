import { useSelector, useDispatch } from "react-redux";
import { useCallback } from "react";
import {
  selectUser,
  selectAuthStatus,
  selectAuthError,
  selectIsAuthenticated,
  signIn,
  signOut,
  fetchCurrentUser,
} from "../store/authSlice";
import { toast } from "react-hot-toast";
import { supabase } from "../supabaseClient";

/**
 * Custom hook for authentication
 * Provides access to auth state and methods
 */
export const useAuth = () => {
  const dispatch = useDispatch();
  const user = useSelector(selectUser);
  const status = useSelector(selectAuthStatus);
  const error = useSelector(selectAuthError);
  const isAuthenticated = useSelector(selectIsAuthenticated);

  /**
   * Sign in with email and password
   */
  const login = useCallback(
    async (email, password) => {
      try {
        const resultAction = await dispatch(signIn({ email, password }));
        if (signIn.fulfilled.match(resultAction)) {
          toast.success("Signed in successfully", {
            id: "auth-success", // Prevent duplicate toasts
          });
          return { success: true, user: resultAction.payload };
        } else {
          const errorMessage = resultAction.payload || "Failed to sign in";
          toast.error(errorMessage);
          return { success: false, error: errorMessage };
        }
      } catch (error) {
        toast.error("An unexpected error occurred");
        return { success: false, error: error.message };
      }
    },
    [dispatch]
  );

  /**
   * Sign out the current user
   */
  const logout = useCallback(async () => {
    try {
      const resultAction = await dispatch(signOut());
      if (signOut.fulfilled.match(resultAction)) {
        toast.success("Signed out successfully");
        return { success: true };
      } else {
        const errorMessage = resultAction.payload || "Failed to sign out";
        toast.error(errorMessage);
        return { success: false, error: errorMessage };
      }
    } catch (error) {
      toast.error("An unexpected error occurred");
      return { success: false, error: error.message };
    }
  }, [dispatch]);

  /**
   * Sign up with email and password
   */
  const signup = useCallback(async (email, password) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
      });

      if (error) {
        toast.error(error.message);
        return { success: false, error: error.message };
      }

      toast.success(
        "Signed up successfully! Check your email for confirmation."
      );
      return { success: true, user: data.user };
    } catch (error) {
      toast.error("An unexpected error occurred");
      return { success: false, error: error.message };
    }
  }, []);

  /**
   * Reset password
   */
  const resetPassword = useCallback(async (email) => {
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: window.location.origin + "/reset-password",
      });

      if (error) {
        toast.error(error.message);
        return { success: false, error: error.message };
      }

      toast.success("Password reset email sent! Check your inbox.");
      return { success: true };
    } catch (error) {
      toast.error("An unexpected error occurred");
      return { success: false, error: error.message };
    }
  }, []);

  /**
   * Refresh the current user
   */
  const refreshUser = useCallback(async () => {
    try {
      const resultAction = await dispatch(fetchCurrentUser());
      if (fetchCurrentUser.fulfilled.match(resultAction)) {
        return { success: true, user: resultAction.payload };
      } else {
        const errorMessage = resultAction.payload || "Failed to refresh user";
        return { success: false, error: errorMessage };
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }, [dispatch]);

  return {
    user,
    status,
    error,
    isAuthenticated,
    isLoading: status === "loading",
    login,
    logout,
    signup,
    resetPassword,
    refreshUser,
  };
};
