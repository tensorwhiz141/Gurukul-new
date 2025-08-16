import React, { useEffect, useRef, useCallback } from "react";
import { supabase } from "../supabaseClient";
import { timeSync } from "../utils/timeSync";

const SessionTracker = ({ onTimeUpdate }) => {
  const sessionIdRef = useRef(crypto.randomUUID());
  const startTimeRef = useRef(Date.now());
  const userIdRef = useRef(null);
  const lastRecordedTimeRef = useRef(null);
  const lastUpdateTimeRef = useRef(Date.now());

  // Record time spent with throttling
  const recordTimeSpent = useCallback(async (endReason) => {
    if (!userIdRef.current) return;

    const now = Date.now();
    // Only record if at least 5 seconds have passed since last update
    if (now - lastUpdateTimeRef.current < 5000) return;

    const timeSpent = Math.floor((now - startTimeRef.current) / 1000);
    if (timeSpent < 1 || timeSpent === lastRecordedTimeRef.current) return;

    try {
      const { error } = await supabase.from("time_tracking").insert({
        user_id: userIdRef.current,
        session_id: sessionIdRef.current,
        time_spent: timeSpent,
        session_end: new Date().toISOString(),
        end_reason: endReason,
      });

      if (error) {
        console.error("Error recording time:", error);
      } else {
        lastRecordedTimeRef.current = timeSpent;
        lastUpdateTimeRef.current = now;
      }
    } catch (err) {
      console.error("Error in recordTimeSpent:", err);
    }
  }, []);

  useEffect(() => {
    const initTracking = async () => {
      const {
        data: { session },
        error: sessionError,
      } = await supabase.auth.getSession();
      if (sessionError || !session) {
        console.error("Supabase session error:", sessionError);
        return;
      }
      userIdRef.current = session.user.id;

      // Record initial session start
      const { error } = await supabase.from("time_tracking").insert({
        user_id: userIdRef.current,
        session_id: sessionIdRef.current,
        time_spent: 0,
        session_start: new Date().toISOString(),
      });
      if (error) console.error("Error recording session start:", error);
    };

    // Reset timeSync when component mounts
    timeSync.reset();

    // Subscribe to timeSync updates
    const unsubscribe = timeSync.subscribe((elapsed) => {
      if (onTimeUpdate) onTimeUpdate(elapsed);
    });

    const handleVisibilityChange = () => {
      if (document.hidden) {
        recordTimeSpent("tab_switch");
      } else {
        timeSync.reset();
      }
    };

    const handleUnload = () => {
      recordTimeSpent("page_close");
    };

    // Set up event listeners
    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("beforeunload", handleUnload);

    // Start tracking
    initTracking();

    // Cleanup
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("beforeunload", handleUnload);
      recordTimeSpent("component_unmount");
      unsubscribe();
    };
  }, [onTimeUpdate, recordTimeSpent]);

  return null;
};

export default SessionTracker;
