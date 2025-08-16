import React, { useState } from "react";
import { supabase } from "../supabaseClient";
import { toast } from "react-hot-toast";
import { Mail, Lock, UserPlus, RefreshCcw } from "lucide-react";
import GlassInput from "../components/GlassInput";
import GlassButton from "../components/GlassButton";

export default function SignUp() {
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleSignUp = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    const email = e.target.email.value;
    const password = e.target.password.value;
    if (!email || !password) {
      setError("Please enter both email and password.");
      toast.error("Please enter both email and password.", {
        position: "top-right",
      });
      return;
    }
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) {
      setError(error.message);
      toast.error(error.message, { position: "top-right" });
    } else {
      setSuccess("Check your email for a confirmation link!");
      toast.success("Check your email for a confirmation link!", {
        position: "top-right",
      });
    }
  };

  return (
    <main className="min-h-screen w-full flex items-center justify-center relative overflow-hidden">
      {/* Background Image */}
      <div
        className="absolute inset-0 z-0"
        style={{
          backgroundImage: "url(/bg/bg.png)",
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
          filter: "brightness(0.7)",
          WebkitFilter: "brightness(0.7)",
        }}
      />

      <section className="glass-card relative z-10">
        <h1
          className="text-3xl font-bold mb-6"
          style={{
            color: "#FFD700",
            fontFamily: "Tiro Devanagari Hindi, serif",
            textShadow: "0 2px 4px rgba(0,0,0,0.15)",
          }}
        >
          Sign Up
        </h1>
        <form
          className="flex flex-col gap-3 w-full items-center"
          onSubmit={handleSignUp}
        >
          <GlassInput
            name="email"
            type="email"
            placeholder="Email"
            icon={Mail}
            autoComplete="username"
            required
          />
          <GlassInput
            name="password"
            type="password"
            placeholder="Password"
            icon={Lock}
            autoComplete="new-password"
            required
          />
          <GlassButton
            type="submit"
            icon={UserPlus}
            className="w-full mt-2"
            variant="primary"
          >
            Sign Up
          </GlassButton>
        </form>
        {error && (
          <div className="text-red-300 text-sm mt-4 font-semibold">{error}</div>
        )}
        {success && (
          <div className="text-green-300 text-sm mt-4 font-semibold">
            {success}
          </div>
        )}

        <div className="mt-7 flex flex-col gap-3 items-center text-sm">
          <div className="w-full border-t border-white/20 my-2"></div>
          {/* Back to Sign In link */}
          <div>
            {(() => {
              try {
                // eslint-disable-next-line
                const { Link } = require("react-router-dom");
                return (
                  <Link
                    to="/SignIn"
                    className="text-white hover:text-[#FFD700] transition-colors font-medium"
                  >
                    Already have an account? Sign In
                  </Link>
                );
              } catch {
                return (
                  <a
                    href="/SignIn"
                    className="text-white hover:text-[#FFD700] transition-colors font-medium"
                  >
                    Already have an account? Sign In
                  </a>
                );
              }
            })()}
          </div>
        </div>
      </section>
    </main>
  );
}
