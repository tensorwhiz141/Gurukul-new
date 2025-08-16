import React, { useEffect } from "react";
import { supabase } from "../supabaseClient";
import { toast } from "react-hot-toast";
import { storage } from "../utils/storageUtils";

export default function SignOut() {
  useEffect(() => {
    // Execute immediately
    const signOutAndRedirect = () => {
      // Show toast notification
      toast.success("Signing you out...", {
        icon: "👋",
        duration: 2000,
      });

      // Clear only auth-related storage, preserve avatar data
      try {
        storage.clearExceptPersistent();
        sessionStorage.clear();
      } catch (e) {
        console.error("Error clearing storage:", e);
      }

      // Force redirect to sign-in page
      window.location.href = "/signin";

      // Attempt to sign out after redirect is initiated
      // This is a "fire and forget" approach
      try {
        supabase.auth.signOut();
      } catch (e) {
        console.error("Error during sign out:", e);
        // We don't need to handle this error since we're already redirecting
      }
    };

    // Call the function immediately
    signOutAndRedirect();
  }, []);

  return (
    <main className="min-h-screen flex items-center justify-center bg-gradient-to-b from-[#1a1a2e] to-[#16213e]">
      <section className="w-full max-w-sm mx-auto rounded-3xl shadow-lg bg-[#0f3460]/30 backdrop-blur-md px-8 py-12 flex flex-col items-center text-center border border-[#e94560]/20">
        <h1
          className="text-3xl font-bold mb-6 text-white"
          style={{ fontFamily: "Nunito, sans-serif" }}
        >
          Signing Out...
        </h1>
        <p
          className="text-white/80 mb-6"
          style={{ fontFamily: "Nunito, sans-serif" }}
        >
          You are being signed out. Redirecting to sign in page.
        </p>
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white mt-4"></div>
      </section>
    </main>
  );
}
