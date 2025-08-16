import { createClient } from "@supabase/supabase-js";
import { storage } from "./utils/storageUtils";

// Use Vite environment variables
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// Validate environment variables
if (!supabaseUrl || !supabaseAnonKey) {
  console.error(
    "Missing Supabase environment variables. Check your .env file."
  );
}

// Create a custom storage object with error handling
const customStorage = {
  getItem: (key) => {
    try {
      return localStorage.getItem(key);
    } catch (error) {
      console.error("Error accessing localStorage:", error);
      return null;
    }
  },
  setItem: (key, value) => {
    try {
      localStorage.setItem(key, value);
    } catch (error) {
      console.error("Error writing to localStorage:", error);
    }
  },
  removeItem: (key) => {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error("Error removing from localStorage:", error);
    }
  },
  clear: () => {
    try {
      // Only clear Supabase-related items
      Object.keys(localStorage).forEach((key) => {
        if (key.includes("supabase") || key.includes("sb-")) {
          localStorage.removeItem(key);
        }
      });
    } catch (error) {
      console.error("Error clearing localStorage:", error);
    }
  },
};

// Create a single Supabase client instance for the entire application
export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
    // Set redirectTo for OAuth operations
    redirectTo: window.location.origin + "/auth/callback",
    // Use our custom storage implementation
    storage: customStorage,
    // Set shorter storage key duration to prevent stale sessions
    storageKey: `sb-auth-token-${Math.floor(
      Date.now() / (1000 * 60 * 60 * 24)
    )}`,
    flowType: "pkce",
  },
  global: {
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  },
  // Add debug option to help with troubleshooting
  debug: import.meta.env.DEV,
  // Set shorter localStorage TTL to prevent stale data
  realtime: {
    params: {
      eventsPerSecond: 10,
    },
  },
});

// Add event listener for auth state changes to help with debugging
supabase.auth.onAuthStateChange((event, session) => {
  console.log(
    `Supabase auth event: ${event}`,
    session ? "User is authenticated" : "No user"
  );

  // Handle session recovery errors
  if (event === "TOKEN_REFRESHED") {
    console.log("Token refreshed successfully");
  }

  if (event === "SIGNED_OUT") {
    // Clear only auth-related items from localStorage, preserve avatar data
    storage.clearAuthKeys();
  }
});
